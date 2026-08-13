"""Host properties that decide what bytes the pinned oracle produces.

Almost nothing about this compiler depends on the machine it runs on: a table
is a table, and a plan hash means the same bytes anywhere.  One thing is not
like that.

A C compiler may contract ``a * b + c`` into a single fused multiply-add, which
rounds once where the written expression rounds twice.  Clang does it by
default -- ``-ffp-contract=on`` -- but only on a target whose instruction set
has the operation, so the same FFmpeg source compiled from the same revision
produces *different bytes* on AArch64 than on baseline x86-64.  For
``colorcontrast`` that is 147 bytes out of the 50,331,648 its complete RGB
domain covers.  Upstream's output is simply not portable there, and a compiler
that promises bit-exactness against the pinned oracle has to reproduce the
oracle it was pinned against.

So the lowering asks this module which kind of host it is on, and encodes the
answer in the IR as explicit fused multiply-adds or as separate operations.
Because the answer changes the program, it changes the plan hash too, and a
kernel built for one kind of host can never be served from a cache or a bundle
to the other.

Only filters whose arithmetic can actually tell the difference ask.
``colorbalance`` does not: every multiply-add it contains scales by ``4.f``,
which is exact, so both forms give the same bytes and its plan hash is the same
everywhere.
"""

from __future__ import annotations

import functools
import os
import platform


class UnknownTargetError(RuntimeError):
    """A host property this lowering depends on could not be determined."""


#: Machines whose baseline instruction set has a fused multiply-add, so Clang
#: contracts ``a * b + c`` at its default ``-ffp-contract=on``.
FUSED_MULTIPLY_ADD_MACHINES = frozenset({"arm64", "arm64e", "aarch64"})

#: Machines whose baseline does not, so the multiply and the add round
#: separately.  FFmpeg's configure passes no ``-march``, so an x86-64 build
#: targets the base ISA even on a CPU that has FMA3.
UNFUSED_MULTIPLY_ADD_MACHINES = frozenset({"x86_64", "amd64", "i386", "i686", "i86pc"})

#: Overrides the machine table, for a host that builds FFmpeg with a
#: ``-march`` its baseline does not imply.
FUSED_MULTIPLY_ADD_VARIABLE = "LAVFI_CC_FUSED_MULTIPLY_ADD"

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


@functools.lru_cache(maxsize=None)
def _machine_fuses(machine: str) -> bool | None:
    if machine in FUSED_MULTIPLY_ADD_MACHINES:
        return True
    if machine in UNFUSED_MULTIPLY_ADD_MACHINES:
        return False
    return None


def target_fuses_multiply_add() -> bool:
    """Whether this host's C compiler contracts ``a * b + c`` into one rounding.

    Raises :class:`UnknownTargetError` on a machine the table does not name,
    rather than guessing: guessing wrong is a wrong byte in a filter whose
    whole purpose here is to produce the right ones.
    """

    override = os.environ.get(FUSED_MULTIPLY_ADD_VARIABLE)
    if override is not None:
        normalized = override.strip().lower()
        if normalized in _TRUE:
            return True
        if normalized in _FALSE:
            return False
        raise UnknownTargetError(
            f"{FUSED_MULTIPLY_ADD_VARIABLE}={override!r} is not a boolean"
        )

    machine = platform.machine().lower()
    fuses = _machine_fuses(machine)
    if fuses is None:
        raise UnknownTargetError(
            f"whether {machine or 'this machine'!r} fuses a multiply-add is not "
            "recorded, and it decides which bytes the pinned oracle produces; "
            f"set {FUSED_MULTIPLY_ADD_VARIABLE}=1 or 0 after checking the host"
        )
    return fuses
