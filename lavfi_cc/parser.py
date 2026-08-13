"""Fail-closed parser for the linear filtergraph subset accepted by lavfi-cc.

This is intentionally not an FFmpeg command-line parser.  It recognizes one
linear ``-vf`` value, preserves source offsets, and rejects graph features that
could change how separators are interpreted.
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

    def named_options(self) -> dict[str, FilterOption]:
        return {option.name: option for option in self.options if option.name is not None}

    def positional_options(self) -> tuple[FilterOption, ...]:
        return tuple(option for option in self.options if option.name is None)


@dataclass(frozen=True)
class Filtergraph:
    source: str
    filters: tuple[FilterInvocation, ...]


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

        options: list[FilterOption] = []
        if equals is not None:
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

        parsed.append(
            FilterInvocation(name, tuple(options), SourceSpan(start, end), raw_filter)
        )

    return Filtergraph(source, tuple(parsed))
