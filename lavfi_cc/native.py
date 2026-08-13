"""Compile, load, validate, and invoke generated native kernels."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile
from typing import Any

from .codegen import GeneratedC, generate_c
from .ir import PixelIR
from .layouts import DEFAULT_LAYOUT, get_layout


KERNEL_ABI_VERSION = 2
PIXEL_FORMAT_RGBA8 = 1
BASE_COMPILER_FLAGS = (
    "-std=c11",
    "-O2",
    "-fPIC",
    "-fno-fast-math",
    "-ffp-contract=off",
)
_RUNTIME_DIRECTORY = Path(__file__).resolve().parents[1] / "runtime"


class NativeError(RuntimeError):
    """Base class for native-backend failures."""


class CompilationError(NativeError):
    """Clang could not compile a generated translation unit."""


class KernelLoadError(NativeError):
    """A shared library did not satisfy the kernel ABI contract."""


class KernelExecutionError(NativeError):
    """A frame layout cannot be passed to a native kernel safely."""


def library_suffix() -> str:
    system = platform.system()
    if system == "Darwin":
        return ".dylib"
    if system == "Linux":
        return ".so"
    raise CompilationError(
        f"native kernels are unsupported on {system or 'this platform'}"
    )


def _clang_path(configured: str | os.PathLike[str] | None) -> str:
    candidate = os.fspath(configured) if configured is not None else shutil.which("clang")
    if not candidate:
        raise CompilationError("Clang was not found; set LAVFI_CC_CLANG or install clang")
    path = Path(candidate).expanduser()
    if path.parent == Path("."):
        resolved = shutil.which(candidate)
        if resolved:
            path = Path(resolved)
    if not path.is_file() or not os.access(path, os.X_OK):
        raise CompilationError(f"Clang is not executable: {path}")
    return str(path.resolve())


def compiler_command(
    source_path: Path,
    output_path: Path,
    *,
    clang: str | os.PathLike[str] | None = None,
) -> tuple[str, ...]:
    compiler = _clang_path(clang or os.environ.get("LAVFI_CC_CLANG"))
    system = platform.system()
    if system == "Darwin":
        link_mode = "-dynamiclib"
    elif system == "Linux":
        link_mode = "-shared"
    else:
        raise CompilationError(
            f"native kernels are unsupported on {system or 'this platform'}"
        )
    return (
        compiler,
        *BASE_COMPILER_FLAGS,
        link_mode,
        "-I",
        str(_RUNTIME_DIRECTORY),
        str(source_path),
        "-o",
        str(output_path),
    )


@dataclass(frozen=True)
class CompiledArtifact:
    library_path: Path
    source_path: Path
    command: tuple[str, ...]
    generated: GeneratedC


def compile_generated_c(
    generated: GeneratedC,
    output_path: str | os.PathLike[str],
    *,
    source_path: str | os.PathLike[str] | None = None,
    clang: str | os.PathLike[str] | None = None,
) -> CompiledArtifact:
    """Compile generated source to one native shared library."""

    output = Path(output_path).expanduser().resolve()
    source = (
        Path(source_path).expanduser().resolve()
        if source_path is not None
        else output.with_suffix(".c")
    )
    if output == source:
        raise CompilationError("generated C and native library paths must be different")
    output.parent.mkdir(parents=True, exist_ok=True)
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(generated.source, encoding="utf-8")
    command = compiler_command(source, output, clang=clang)
    process = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if process.returncode != 0:
        try:
            output.unlink(missing_ok=True)
        except OSError:
            pass
        detail = process.stderr.strip() or process.stdout.strip() or "unknown compiler error"
        raise CompilationError(f"Clang exited with status {process.returncode}: {detail}")
    return CompiledArtifact(output, source, command, generated)


def compile_kernel(
    ir: PixelIR,
    output_path: str | os.PathLike[str],
    *,
    source_path: str | os.PathLike[str] | None = None,
    clang: str | os.PathLike[str] | None = None,
    identity_elimination: bool = True,
    lut_composition: bool = True,
) -> CompiledArtifact:
    generated = generate_c(
        ir,
        identity_elimination=identity_elimination,
        lut_composition=lut_composition,
    )
    return compile_generated_c(
        generated,
        output_path,
        source_path=source_path,
        clang=clang,
    )


MAX_PLANES = 4

_PlanePointers = ctypes.POINTER(ctypes.c_uint8) * MAX_PLANES
_PlaneStrides = ctypes.c_ssize_t * MAX_PLANES

# ABI 2 passes plane arrays shaped like AVFrame's data[] and linesize[].
_ProcessFunction = ctypes.CFUNCTYPE(
    None,
    ctypes.POINTER(ctypes.POINTER(ctypes.c_uint8)),
    ctypes.POINTER(ctypes.c_ssize_t),
    ctypes.POINTER(ctypes.POINTER(ctypes.c_uint8)),
    ctypes.POINTER(ctypes.c_ssize_t),
    ctypes.c_int,
    ctypes.c_int,
)


class _KernelABI(ctypes.Structure):
    _fields_ = (
        ("abi_version", ctypes.c_uint32),
        ("pixel_format", ctypes.c_uint32),
        ("plan_hash", ctypes.c_char_p),
        ("process", _ProcessFunction),
    )


def _byte_view(value: Any, description: str, *, writable: bool) -> memoryview:
    try:
        view = memoryview(value)
    except TypeError as error:
        raise KernelExecutionError(
            f"{description} must support the buffer protocol"
        ) from error
    if writable and view.readonly:
        raise KernelExecutionError(f"{description} must be writable")
    if not view.c_contiguous:
        raise KernelExecutionError(f"{description} must be C-contiguous")
    try:
        return view.cast("B")
    except TypeError as error:
        raise KernelExecutionError(
            f"{description} must have a byte-compatible layout"
        ) from error


def _positive_integer(value: Any, description: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise KernelExecutionError(f"{description} must be a positive integer")
    return value


def _validate_layout(
    length: int,
    offset: int,
    stride: int,
    row_bytes: int,
    height: int,
    description: str,
) -> None:
    if isinstance(offset, bool) or not isinstance(offset, int):
        raise KernelExecutionError(f"{description} offset must be an integer")
    if isinstance(stride, bool) or not isinstance(stride, int):
        raise KernelExecutionError(f"{description} stride must be an integer")
    first = offset
    last = offset + (height - 1) * stride
    lowest = min(first, last)
    highest = max(first, last) + row_bytes
    if lowest < 0 or highest > length:
        raise KernelExecutionError(
            f"{description} buffer is too small for offset={offset}, stride={stride}, "
            f"row_bytes={row_bytes}, height={height}"
        )


class NativeKernel:
    """A loaded native kernel whose ABI fields have already been checked."""

    def __init__(
        self,
        library_path: str | os.PathLike[str],
        expected_plan_hash: str,
        *,
        temporary_directory: tempfile.TemporaryDirectory[str] | None = None,
        artifact: CompiledArtifact | None = None,
        layout: str = DEFAULT_LAYOUT,
    ) -> None:
        path = Path(library_path).expanduser().resolve()
        if not path.is_file():
            raise KernelLoadError(f"kernel library does not exist: {path}")
        mode = getattr(os, "RTLD_NOW", 0) | getattr(os, "RTLD_LOCAL", 0)
        try:
            library = ctypes.CDLL(str(path), mode=mode)
            entry = _KernelABI.in_dll(library, "lavfi_compiled_kernel")
        except (OSError, ValueError) as error:
            raise KernelLoadError(f"could not load kernel {path}: {error}") from error
        if entry.abi_version != KERNEL_ABI_VERSION:
            raise KernelLoadError(
                f"kernel ABI version {entry.abi_version} does not match "
                f"{KERNEL_ABI_VERSION}"
            )
        expected_layout = get_layout(layout)
        if entry.pixel_format != expected_layout.abi_id:
            raise KernelLoadError(
                f"kernel pixel format {entry.pixel_format} is not "
                f"{expected_layout.name} ({expected_layout.abi_id})"
            )
        encoded_hash = entry.plan_hash
        if encoded_hash is None:
            raise KernelLoadError("kernel plan hash is null")
        try:
            observed_hash = encoded_hash.decode("ascii")
        except UnicodeDecodeError as error:
            raise KernelLoadError("kernel plan hash is not ASCII") from error
        if observed_hash != expected_plan_hash:
            raise KernelLoadError(
                f"kernel plan hash {observed_hash!r} does not match "
                f"{expected_plan_hash!r}"
            )
        if not entry.process:
            raise KernelLoadError("kernel process function is null")

        self.path = path
        self.plan_hash = observed_hash
        self.artifact = artifact
        self.layout = expected_layout
        self.step = expected_layout.step
        self._library: ctypes.CDLL | None = library
        self._process: _ProcessFunction | None = entry.process
        self._temporary_directory = temporary_directory

    @classmethod
    def compile(
        cls,
        ir: PixelIR,
        *,
        clang: str | os.PathLike[str] | None = None,
        identity_elimination: bool = True,
        lut_composition: bool = True,
    ) -> "NativeKernel":
        temporary = tempfile.TemporaryDirectory(prefix="lavfi-cc-week4-")
        directory = Path(temporary.name)
        output = directory / (ir.plan_hash + library_suffix())
        try:
            artifact = compile_kernel(
                ir,
                output,
                source_path=directory / (ir.plan_hash + ".c"),
                clang=clang,
                identity_elimination=identity_elimination,
                lut_composition=lut_composition,
            )
            return cls(
                artifact.library_path,
                ir.plan_hash,
                temporary_directory=temporary,
                artifact=artifact,
            )
        except Exception:
            temporary.cleanup()
            raise

    def close(self) -> None:
        self._process = None
        self._library = None
        if self._temporary_directory is not None:
            self._temporary_directory.cleanup()
            self._temporary_directory = None

    def __enter__(self) -> "NativeKernel":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def process_into(
        self,
        source: Any,
        destination: Any,
        width: int,
        height: int,
        *,
        source_stride: int | None = None,
        destination_stride: int | None = None,
        source_offset: int = 0,
        destination_offset: int = 0,
    ) -> None:
        width = _positive_integer(width, "width")
        height = _positive_integer(height, "height")
        row_bytes = self.layout.row_bytes(width)
        source_stride = row_bytes if source_stride is None else source_stride
        destination_stride = (
            row_bytes if destination_stride is None else destination_stride
        )
        source_view = _byte_view(source, "source", writable=False)
        destination_view = _byte_view(destination, "destination", writable=True)
        if source_view.obj is destination_view.obj:
            raise KernelExecutionError("source and destination must use distinct buffers")
        # Planes of a tightly packed frame sit back to back, as rawvideo writes
        # them. A packed layout has exactly one.
        origins = [
            self.layout.plane_origin(plane, width, height)
            for plane in range(self.layout.plane_count)
        ]
        for plane, origin in enumerate(origins):
            _validate_layout(
                len(source_view),
                source_offset + origin,
                source_stride,
                row_bytes,
                height,
                f"source plane {plane}",
            )
            _validate_layout(
                len(destination_view),
                destination_offset + origin,
                destination_stride,
                row_bytes,
                height,
                f"destination plane {plane}",
            )

        source_type = ctypes.c_uint8 * len(source_view)
        if source_view.readonly:
            source_array = source_type.from_buffer_copy(source_view)
        else:
            source_array = source_type.from_buffer(source_view)
        destination_type = ctypes.c_uint8 * len(destination_view)
        destination_array = destination_type.from_buffer(destination_view)

        source_planes = _PlanePointers()
        destination_planes = _PlanePointers()
        source_strides = _PlaneStrides()
        destination_strides = _PlaneStrides()
        for plane, origin in enumerate(origins):
            source_planes[plane] = ctypes.cast(
                ctypes.byref(source_array, source_offset + origin),
                ctypes.POINTER(ctypes.c_uint8),
            )
            destination_planes[plane] = ctypes.cast(
                ctypes.byref(destination_array, destination_offset + origin),
                ctypes.POINTER(ctypes.c_uint8),
            )
            source_strides[plane] = source_stride
            destination_strides[plane] = destination_stride

        process = self._process
        if process is None:
            raise KernelExecutionError("kernel is closed")
        process(
            destination_planes,
            destination_strides,
            source_planes,
            source_strides,
            width,
            height,
        )

    def process_rgba8(self, source: Any, width: int, height: int) -> bytes:
        width = _positive_integer(width, "width")
        height = _positive_integer(height, "height")
        expected = self.layout.frame_size(width, height)
        source_view = _byte_view(source, "source", writable=False)
        if len(source_view) != expected:
            raise KernelExecutionError(
                f"packed source has {len(source_view)} bytes; expected {expected}"
            )
        output = bytearray(expected)
        self.process_into(source_view, output, width, height)
        return bytes(output)
