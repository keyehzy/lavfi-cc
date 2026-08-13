"""Parsers for FFmpeg filtergraph values.

``parse_filtergraph`` is fail-closed and intentionally not an FFmpeg
command-line parser.  It recognizes one linear ``-vf`` value, preserves source
offsets, and rejects graph features that could change how separators are
interpreted.  It is the only parser used on the fusion path.

``parse_filtergraph_script`` is the lenient counterpart used by analysis-only
tooling.  It accepts the wider syntax found in real filtergraphs -- several
chains, link labels, and surrounding whitespace -- and records per-filter option
problems instead of rejecting the whole graph, so that a scan can still report
what an unparseable neighbour prevented.
"""

from __future__ import annotations

from dataclasses import dataclass
import re


_NAME = re.compile(r"[a-z][a-z0-9_]*\Z")
_OPTION_NAME = re.compile(r"[a-z][a-z0-9_]*\Z")


class FiltergraphSyntaxError(ValueError):
    """A syntax error with an offset into the original filtergraph."""

    def __init__(self, message: str, offset: int):
        super().__init__(message)
        self.message = message
        self.offset = offset

    def __str__(self) -> str:
        return f"{self.message} at byte {self.offset}"


@dataclass(frozen=True)
class SourceSpan:
    start: int
    end: int

    def as_dict(self) -> dict[str, int]:
        return {"start": self.start, "end": self.end}


@dataclass(frozen=True)
class FilterOption:
    name: str | None
    value: str
    span: SourceSpan


@dataclass(frozen=True)
class FilterInvocation:
    name: str
    options: tuple[FilterOption, ...]
    span: SourceSpan
    raw: str
    inputs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()
    option_error: str | None = None

    def named_options(self) -> dict[str, FilterOption]:
        return {option.name: option for option in self.options if option.name is not None}

    def positional_options(self) -> tuple[FilterOption, ...]:
        return tuple(option for option in self.options if option.name is None)


@dataclass(frozen=True)
class Filtergraph:
    source: str
    filters: tuple[FilterInvocation, ...]


@dataclass(frozen=True)
class FilterChain:
    """One comma-separated run of filters inside a larger graph."""

    filters: tuple[FilterInvocation, ...]
    span: SourceSpan


@dataclass(frozen=True)
class FiltergraphScript:
    """A whole ``-filter_complex``-shaped value, parsed leniently."""

    source: str
    chains: tuple[FilterChain, ...]

    @property
    def filters(self) -> tuple[FilterInvocation, ...]:
        return tuple(
            invocation for chain in self.chains for invocation in chain.filters
        )


def _split(source: str, delimiter: str, base: int = 0) -> list[tuple[str, int, int]]:
    """Split on unescaped delimiters outside FFmpeg-style single quotes."""

    parts: list[tuple[str, int, int]] = []
    start = 0
    quote_offset: int | None = None
    escaped = False
    for index, char in enumerate(source):
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == "'":
            quote_offset = None if quote_offset is not None else index
            continue
        if quote_offset is None and char == delimiter:
            parts.append((source[start:index], base + start, base + index))
            start = index + 1
    if escaped:
        raise FiltergraphSyntaxError("trailing escape", base + len(source) - 1)
    if quote_offset is not None:
        raise FiltergraphSyntaxError("unterminated single quote", base + quote_offset)
    parts.append((source[start:], base + start, base + len(source)))
    return parts


def _find_unescaped(source: str, needle: str) -> int | None:
    quoted = False
    escaped = False
    for index, char in enumerate(source):
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == "'":
            quoted = not quoted
            continue
        if not quoted and char == needle:
            return index
    return None


def _decode(source: str, base: int) -> str:
    """Remove the graph-parser quoting layer from one token."""

    result: list[str] = []
    quoted = False
    escaped = False
    for index, char in enumerate(source):
        if escaped:
            result.append(char)
            escaped = False
        elif char == "\\":
            escaped = True
        elif char == "'":
            quoted = not quoted
        elif char == '"':
            raise FiltergraphSyntaxError(
                "double quotes are not quoting characters in FFmpeg filtergraphs; use single quotes",
                base + index,
            )
        else:
            result.append(char)
    if escaped:
        raise FiltergraphSyntaxError("trailing escape", base + len(source) - 1)
    if quoted:
        raise FiltergraphSyntaxError("unterminated single quote", base)
    return "".join(result)


def _reject_graph_syntax(source: str) -> None:
    """Reject labels, links, and multiple chains before parsing any options."""

    quoted = False
    escaped = False
    for index, char in enumerate(source):
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == "'":
            quoted = not quoted
            continue
        if not quoted and char in ";[]":
            descriptions = {
                ";": "multiple filtergraph chains are not supported",
                "[": "link labels are not supported",
                "]": "link labels are not supported",
            }
            raise FiltergraphSyntaxError(descriptions[char], index)


def _parse_options(
    name: str, raw_filter: str, equals: int | None, start: int
) -> list[FilterOption]:
    """Parse one filter's ``key=value:...`` argument list."""

    options: list[FilterOption] = []
    if equals is None:
        return options
    option_source = raw_filter[equals + 1 :]
    option_base = start + equals + 1
    if not option_source:
        raise FiltergraphSyntaxError(f"filter {name!r} has an empty option list", option_base)
    seen: set[str] = set()
    for raw_option, option_start, option_end in _split(option_source, ":", option_base):
        if not raw_option:
            raise FiltergraphSyntaxError("empty option between colons", option_start)
        option_equals = _find_unescaped(raw_option, "=")
        if option_equals is None:
            value = _decode(raw_option, option_start)
            if not value:
                raise FiltergraphSyntaxError("empty positional option", option_start)
            options.append(FilterOption(None, value, SourceSpan(option_start, option_end)))
            continue
        raw_option_name = raw_option[:option_equals]
        raw_value = raw_option[option_equals + 1 :]
        option_name = _decode(raw_option_name, option_start)
        if not _OPTION_NAME.fullmatch(option_name):
            raise FiltergraphSyntaxError(f"invalid option name {option_name!r}", option_start)
        if option_name in seen:
            raise FiltergraphSyntaxError(
                f"duplicate option {option_name!r} is ambiguous", option_start
            )
        if not raw_value:
            raise FiltergraphSyntaxError(
                f"option {option_name!r} has an empty value", option_start + option_equals + 1
            )
        seen.add(option_name)
        options.append(
            FilterOption(
                option_name,
                _decode(raw_value, option_start + option_equals + 1),
                SourceSpan(option_start, option_end),
            )
        )
    return options


def _strip_labels(token: str, base: int) -> tuple[tuple[str, ...], tuple[str, ...], str, int]:
    """Split ``[in]name=args[out]`` into its labels, body, and body offset."""

    inputs: list[str] = []
    outputs: list[str] = []
    start = 0
    end = len(token)

    while start < end and token[start] == "[":
        close = token.find("]", start)
        if close == -1 or close >= end:
            raise FiltergraphSyntaxError("unterminated link label", base + start)
        inputs.append(token[start + 1 : close])
        start = close + 1
    while end > start and token[end - 1] == "]":
        open_bracket = token.rfind("[", start, end)
        if open_bracket == -1:
            raise FiltergraphSyntaxError("unterminated link label", base + end - 1)
        outputs.insert(0, token[open_bracket + 1 : end - 1])
        end = open_bracket

    return tuple(inputs), tuple(outputs), token[start:end], base + start


def parse_filtergraph_script(source: str) -> FiltergraphScript:
    """Parse an arbitrary filtergraph for analysis, never for fusion.

    Several chains, link labels, and incidental whitespace are all accepted.  A
    filter whose options do not parse is still reported, with ``option_error``
    describing why, so a scan can explain what blocked an island instead of
    silently dropping the graph.
    """

    if not source.strip():
        raise FiltergraphSyntaxError("filtergraph is empty", 0)

    chains: list[FilterChain] = []
    for raw_chain, chain_start, chain_end in _split(source, ";"):
        if not raw_chain.strip():
            raise FiltergraphSyntaxError("empty chain between semicolons", chain_start)
        parsed: list[FilterInvocation] = []
        for raw_filter, start, end in _split(raw_chain, ",", chain_start):
            stripped = raw_filter.strip()
            if not stripped:
                raise FiltergraphSyntaxError("empty filter between commas", start)
            offset = start + raw_filter.index(stripped)
            inputs, outputs, body, body_start = _strip_labels(stripped, offset)
            if not body:
                raise FiltergraphSyntaxError("link labels without a filter", offset)

            equals = _find_unescaped(body, "=")
            raw_name = body if equals is None else body[:equals]
            name = _decode(raw_name, body_start)
            if not _NAME.fullmatch(name):
                raise FiltergraphSyntaxError(
                    f"invalid or unsupported filter name {name!r}", body_start
                )

            option_error: str | None = None
            try:
                options = _parse_options(name, body, equals, body_start)
            except FiltergraphSyntaxError as error:
                option_error = error.message
                options = []

            parsed.append(
                FilterInvocation(
                    name,
                    tuple(options),
                    SourceSpan(body_start, body_start + len(body)),
                    body,
                    inputs,
                    outputs,
                    option_error,
                )
            )
        chains.append(FilterChain(tuple(parsed), SourceSpan(chain_start, chain_end)))

    return FiltergraphScript(source, tuple(chains))


def parse_filtergraph(source: str) -> Filtergraph:
    """Parse one narrow, linear FFmpeg video filtergraph.

    Structural commas and colons may be protected with a backslash or single
    quotes.  Positional option values are represented but eligibility decides
    where they are safe to accept.
    """

    if not source:
        raise FiltergraphSyntaxError("filtergraph is empty", 0)
    _reject_graph_syntax(source)
    parsed: list[FilterInvocation] = []

    for raw_filter, start, end in _split(source, ","):
        if not raw_filter:
            raise FiltergraphSyntaxError("empty filter between commas", start)
        if raw_filter != raw_filter.strip():
            raise FiltergraphSyntaxError(
                "whitespace around a filter is outside the accepted subset", start
            )

        equals = _find_unescaped(raw_filter, "=")
        raw_name = raw_filter if equals is None else raw_filter[:equals]
        name = _decode(raw_name, start)
        if not _NAME.fullmatch(name):
            raise FiltergraphSyntaxError(f"invalid or unsupported filter name {name!r}", start)

        options = _parse_options(name, raw_filter, equals, start)

        parsed.append(
            FilterInvocation(name, tuple(options), SourceSpan(start, end), raw_filter)
        )

    return Filtergraph(source, tuple(parsed))
