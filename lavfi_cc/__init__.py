"""Frontend, pixel IR, and reference interpreter for lavfi-cc."""

from .frontend import Analysis, analyze_filtergraph, require_ir
from .interpreter import (
    InterpreterError,
    interpret_into,
    interpret_pixel,
    interpret_rgba8,
)
from .ir import PixelIR

__all__ = [
    "Analysis",
    "InterpreterError",
    "PixelIR",
    "analyze_filtergraph",
    "interpret_into",
    "interpret_pixel",
    "interpret_rgba8",
    "require_ir",
]
__version__ = "0.3.0"
