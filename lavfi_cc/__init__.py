"""Week 2 frontend and pixel IR for lavfi-cc."""

from .frontend import Analysis, analyze_filtergraph, require_ir
from .ir import PixelIR

__all__ = ["Analysis", "PixelIR", "analyze_filtergraph", "require_ir"]
__version__ = "0.2.0"
