# Week 2 report: frontend and typed pixel IR

Date: 2026-08-13

Decision: **Week 2 exit gate passed.** Every versioned supported chain lowers
to deterministic RGBA8 IR, and every rejected graph returns a specific syntax,
boundary, filter, option, or value diagnostic. The native kernel, cache, and
executable rewrite remain later milestones; the rewrite printed by `explain`
is deliberately marked as planned.

## Command-line interface

The repository-local launcher needs only Python 3.11 or newer:

```sh
./lavfi-cc explain --vf \
  "format=rgba,negate,lutrgb=r=val*1.08+2,format=yuv420p"
```

Human-readable output contains the parsed filters and source spans, eligible
region, semantic canonicalization, pretty-printed IR, SHA-256 plan hash,
metadata effects, and planned rewrite. `--json` exposes both forms of the IR:

- `ir` includes filter and option source spans for diagnostics.
- `canonical_ir` excludes source locations and is the stable serialization
  hashed as the plan identity.

An eligible graph exits 0. A syntax or eligibility rejection exits 2 and can
therefore be used directly by tests and scripts.

## Accepted filtergraph grammar

The parser consumes one `-vf` value, not the complete FFmpeg command line. It
accepts a single linear comma-separated chain and FFmpeg's single-quote and
backslash protection for structural comma and colon characters. It rejects
link labels, branches, semicolon-separated chains, empty elements, duplicate
options, double-quote ambiguity, and whitespace around filter elements.

Exactly one candidate region must have this shape:

```text
format=rgba,<one or more supported filters>,format=<one literal pixel format>
```

`format=pix_fmts=rgba` is equivalent at a boundary. Filters before and after
the boundaries remain opaque and are preserved by the planned rewrite.
Multiple RGBA regions are rejected as ambiguous rather than choosing one.

Inside the region all filter parameters use explicit `key=value` syntax. The
accepted options are:

- `negate`: `components` and `negate_alpha`. Components are a `+`-separated
  subset of `r`, `g`, `b`, and `a`. The legacy alpha option intentionally has
  the pinned packed-RGBA behavior documented in `supported-filters.md`.
- `lutrgb`: `r`, `g`, `b`, and `a`. Defaults are identity tables.
- `colorlevels`: all RGBA `imin`, `imax`, `omin`, and `omax` points plus
  `preserve=none`. Negative input points and endpoints that quantize to the
  same byte are rejected.
- `colorchannelmixer`: the 16 RGBA matrix coefficients, `pc=none`, and `pa`.
  `pa` is range-checked but semantically irrelevant when preservation is off.

Timeline options, runtime commands, non-finite values, and unsupported
preservation modes fail closed.

### `lutrgb` expression subset

Week 2 accepts finite numeric constants; `val`, `clipval`, `negval`, `minval`,
and `maxval`; parentheses; unary signs; `+`, `-`, `*`, and `/`; and the fixed
arity functions `abs`, `min`, `max`, `clip`, and `pow`. A function containing
commas must be single-quoted or escaped at the filtergraph layer, for example:

```text
lutrgb=r='clip(val,0,240)'
```

Frame-dimension variables such as `w` and `h`, stateful/random functions, and
the wider FFmpeg expression language are not yet accepted. The wrapper cannot
silently reinterpret them: it reports `unsupported_expression`. This keeps the
Week 2 IR independent of graph-configuration state; dimension-dependent table
materialization can be added when the run wrapper supplies fixed dimensions.

## IR contract

The canonical JSON IR has a version and fixed `rgba8` format. A plan is a
straight-line sequence containing:

```text
load_rgba8
  lut8 | matrix4x4
  quantize_rgba8
  ...
store_rgba8
```

Every source filter retains its own quantization operation even though all
stages will eventually run inside one pixel loop. `negate` and `lutrgb` become
four concrete 256-entry tables. `colorlevels` becomes a diagonal matrix with
byte-quantized points, exact float32 coefficient encodings, and explicit input
and output offsets. `colorchannelmixer` retains exact float64 coefficient
encodings and the 16 tables that round each input contribution independently
before summation. The quantization operation records truncation, rounding, and
saturation mode.

IR numbers that originate as floating point are serialized as hexadecimal
strings. JSON keys are sorted and whitespace is removed. Equivalent component
orderings and legacy no-op options therefore produce the same bytes and plan
hash. Source locations never affect the hash.

A chain containing `lutrgb` also carries the
`remove_color_dependent_side_data` frame-metadata effect required by the pinned
FFmpeg behavior.

## Diagnostics

Diagnostics have a stable code, explanation, graph offset, filter index, and
option name where applicable. Covered rejection classes include malformed
syntax, absent or malformed format boundaries, ambiguous regions, unsupported
filters, positional or unknown options, timeline options, invalid ranges,
non-finite values, dynamic `colorlevels` extrema, degenerate points, and
unsupported LUT expressions.

The analyzer collects independent unsupported filters in a region so one
`explain` call can report all of them. Lowering does not emit partial IR after
any rejection.

## Verification

Run:

```sh
./scripts/test-week2.sh
```

The 27 tests cover parser escaping and fail-closed syntax, eligibility and
diagnostic precision, LUT evaluation, explicit quantization boundaries,
color-level point/coefficient representation, mixer contribution rounding,
source-map-independent serialization, CLI output and exit status, and metadata
effects.

The differential suite sends every non-control chain in
`benchmarks/chains.tsv` and `tests/corpus/cases.tsv` through both the frontend
and pinned FFmpeg `n8.1.2`. All are accepted by both parsers. A trace-level
oracle test additionally compares all 256 RGBA8 red-table entries for a quoted
comma expression against FFmpeg's configured `lutrgb` table.

The FFmpeg checks use `LAVFI_CC_FFMPEG` when set, then the local Week 1 pinned
build. They skip cleanly if neither is available, so parser and IR unit tests
remain runnable in a fresh checkout.
