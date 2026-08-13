"""Frontend, pixel IR, and reference interpreter for lavfi-cc."""

from .version import VERSION as __version__

from .frontend import Analysis, analyze_filtergraph, require_ir
from .cache import CacheError, KernelCache
from .ffmpeg import FFmpegIntegrationError, run_ffmpeg
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
    "CacheError",
    "CompilationError",
    "FFmpegIntegrationError",
    "InterpreterError",
    "KernelExecutionError",
    "KernelLoadError",
    "KernelCache",
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
    "run_ffmpeg",
    "validate_ir",
]
