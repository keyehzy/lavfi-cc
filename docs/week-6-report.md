# Week 6 report: cache and operational safety

Date: 2026-08-13

Decision: **Week 6 exit gate passed on the primary macOS arm64 development
host.** Native kernels now survive wrapper invocations in a private,
content-addressed cache. Warm lookup, checksum verification, ABI loading, and
release take a 2.849 ms median over 25 samples on this host, below the 20 ms
incremental startup-overhead target. Linux x86-64 remains the authoritative
release and performance platform.

## Cache identity and layout

The cache key is SHA-256 over canonical JSON containing:

- the canonical optimized IR and both source and optimized plan hashes;
- lavfi-cc 0.6.0, kernel ABI 1, RGBA8 identifier 1, and the compatible fused
  AVFilter ABI identifier;
- host OS, architecture, portable-baseline CPU-feature policy, target triple,
  and dynamic-library format;
- resolved Clang identity/version, code-generation flags, generated-C hash,
  and kernel ABI header hash.

The resolved toolchain identity is memoized against the compiler executable's
device, inode, size, and nanosecond mtime. This keeps the incremental warm path
below the target without allowing a compiler replacement to reuse a stale key.

Each entry has a shared library, readable generated C, and JSON manifest. The
cache root must be a current-user-owned mode-0700 directory; artifacts and lock
files are mode 0600. FFmpeg receives the cache as its trusted root and a direct
child library, preserving the Week 5 filter's independent ownership, path,
permission, ABI, pixel-format, and plan-hash checks.

## Atomicity, concurrency, and recovery

Compilation occurs in a private same-filesystem temporary directory while an
exclusive per-key `flock` is held. The compiler result is loaded and validated,
both artifacts and the manifest are checksummed and synced, then the library
and source are atomically renamed into place. The manifest is published last as
the commit marker and the cache directory is synced.

A second process requesting the same key waits on that lock and consumes the
completed entry instead of compiling. `run` retains the lock through preflight
and the user's FFmpeg process; `cache prune` takes locks non-blockingly and
skips active entries. Same-key abandoned build directories and partial
manifest-last publications are removed on the next request.

Every hit checks regular-file type, ownership, private permissions, manifest
identity, library/source SHA-256, and the native ABI. Missing, corrupt, or
mismatched entries are removed and rebuilt under the lock. Compilation and
publication failures leave no committed partial entry. Default fallback and
`--require-fusion` strict behavior remain unchanged.

## Commands

`compile` populates or validates the cache by default and reports the key and
`miss`, `hit`, or `rebuilt` status. Explicit `--output`/`--emit-c` paths preserve
the standalone export workflow. `native` and `run` use cached artifacts.
`explain` reports the computed key and `miss`, `hit`, `corrupt`, or unavailable
status without compiling. `cache list [--json]` verifies and inventories
entries; `cache prune --max-size SIZE` removes oldest unlocked entries until it
meets the requested bound.

The default root is the platform user cache. `--cache-dir` selects a root per
command, and `LAVFI_CC_CACHE_DIR` supplies a process-wide default.

## Verification

The complete exit gate is:

```sh
./scripts/test-week6.sh
```

It runs 77 tests covering all prior parser, IR, interpreter, native, FFmpeg
differential, and end-to-end integration tests plus Week 6 coverage for
deterministic identity, cold/hot reuse, concurrent same-key compilation,
private permissions, failure-injection cleanup, corruption recovery,
inspection, and pruning. It then compiles a four-stage representative kernel
with AddressSanitizer and UndefinedBehaviorSanitizer, executes 257-by-3
randomized RGBA8 input, and
compares it byte-for-byte with the interpreter.

The observed sanitizer result was clean. The separate warm-cache gate reported:

```text
median=2.849 ms  min=2.554 ms  max=4.985 ms  samples=25  target<20.000 ms
```

This satisfies the Week 6 exit condition: warm-cache execution is reliable and
the cache's incremental startup cost is below the MVP target.
