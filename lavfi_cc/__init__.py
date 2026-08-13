"""Frontend, pixel IR, and reference interpreter for lavfi-cc."""

from .frontend import Analysis, analyze_filtergraph, require_ir
from .interpreter import (
    InterpreterError,
    interpret_into,
    interpret_pixel,
    interpret_rgba8,
    validate_ir,
)
from .ir import PixelIR
from .native import (
    CompilationError,
    KernelExecutionError,
    KernelLoadError,
    NativeKernel,
    compile_kernel,
)
from .passes import PassResult, optimize_ir

__all__ = [
    "Analysis",
    "CompilationError",
    "InterpreterError",
    "KernelExecutionError",
    "KernelLoadError",
    "NativeKernel",
    "PassResult",
    "PixelIR",
    "analyze_filtergraph",
    "compile_kernel",
    "interpret_into",
    "interpret_pixel",
    "interpret_rgba8",
    "optimize_ir",
    "require_ir",
    "validate_ir",
]
__version__ = "0.4.0"
