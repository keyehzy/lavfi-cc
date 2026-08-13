"""Private, content-addressed native-kernel cache.

An entry is committed by publishing its JSON manifest last.  Per-key advisory
locks serialize builders and are also held while FFmpeg is using a library, so
pruning cannot remove a live artifact.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import stat
import subprocess
import tempfile
import time
from typing import Any, Callable, Iterator

from .codegen import GeneratedC, generate_c
from .ir import PixelIR
from .layouts import DEFAULT_LAYOUT, LAYOUTS, get_layout
from .native import (
    BASE_COMPILER_FLAGS,
    KERNEL_ABI_VERSION,
    CompiledArtifact,
    NativeKernel,
    _clang_path,
    compile_kernel,
    library_suffix,
)
from .version import VERSION


CACHE_SCHEMA_VERSION = 1
FFMPEG_FUSED_ABI_IDENTIFIER = "lavfi-fused-avfilter-v1"
_CACHE_DIRECTORY_VERSION = "kernels-v1"


class CacheError(RuntimeError):
    """The kernel cache could not be used safely."""


@dataclass(frozen=True)
class CachePlan:
    key: str
    key_inputs: dict[str, Any]
    generated: GeneratedC
    clang_path: str


@dataclass(frozen=True)
class CachedKernel:
    key: str
    library_path: Path
    source_path: Path
    manifest_path: Path
    status: str
    size: int
    generated: GeneratedC
    command: tuple[str, ...] | None = None


@dataclass(frozen=True)
class CacheEntry:
    key: str
    plan_hash: str
    optimized_plan_hash: str
    status: str
    size: int
    created_ns: int
    library_path: Path | None
    detail: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "plan_hash": self.plan_hash,
            "optimized_plan_hash": self.optimized_plan_hash,
            "status": self.status,
            "size": self.size,
            "created_ns": self.created_ns,
            "library": str(self.library_path) if self.library_path else None,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class PruneResult:
    before_size: int
    after_size: int
    removed_entries: int
    skipped_locked: int


def default_cache_directory() -> Path:
    """Return the platform cache root without creating it."""

    configured = os.environ.get("LAVFI_CC_CACHE_DIR")
    if configured:
        return Path(configured).expanduser()
    if platform.system() == "Darwin":
        base = Path.home() / "Library" / "Caches"
    else:
        xdg = os.environ.get("XDG_CACHE_HOME")
        base = Path(xdg).expanduser() if xdg else Path.home() / ".cache"
    return base / "lavfi-cc" / _CACHE_DIRECTORY_VERSION


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _regular_private_file(path: Path) -> None:
    try:
        information = path.lstat()
    except OSError as error:
        raise CacheError(f"could not inspect cache file {path}: {error}") from error
    if not stat.S_ISREG(information.st_mode):
        raise CacheError(f"cache artifact is not a regular file: {path}")
    if hasattr(os, "geteuid") and information.st_uid != os.geteuid():
        raise CacheError(f"cache artifact is not owned by the current user: {path}")
    if stat.S_IMODE(information.st_mode) & 0o077:
        raise CacheError(f"cache artifact is not private (expected mode 0600): {path}")


class KernelCache:
    """Content-addressed cache for checked native kernels."""

    def __init__(self, root: str | os.PathLike[str] | None = None) -> None:
        candidate = (
            Path(root).expanduser() if root is not None else default_cache_directory()
        )
        if not candidate.is_absolute():
            candidate = Path.cwd() / candidate
        self.root = candidate

    def _ensure_root(self) -> Path:
        created = False
        try:
            self.root.mkdir(parents=True, mode=0o700)
            created = True
        except FileExistsError:
            pass
        except OSError as error:
            raise CacheError(
                f"could not create cache directory {self.root}: {error}"
            ) from error
        if created:
            try:
                os.chmod(self.root, 0o700)
            except OSError as error:
                raise CacheError(
                    f"could not make cache directory private {self.root}: {error}"
                ) from error
        try:
            information = self.root.lstat()
        except OSError as error:
            raise CacheError(
                f"could not inspect cache directory {self.root}: {error}"
            ) from error
        if not stat.S_ISDIR(information.st_mode):
            raise CacheError(f"cache root is not a directory: {self.root}")
        if hasattr(os, "geteuid") and information.st_uid != os.geteuid():
            raise CacheError(f"cache root is not owned by the current user: {self.root}")
        mode = stat.S_IMODE(information.st_mode)
        if mode & 0o077:
            raise CacheError(
                f"cache root must be private (mode 0700); {self.root} has mode {mode:04o}"
            )
        return self.root.resolve()

    def _toolchain_identity(
        self, clang: str | os.PathLike[str] | None
    ) -> tuple[str, dict[str, Any]]:
        path = _clang_path(clang or os.environ.get("LAVFI_CC_CLANG"))
        try:
            information = os.stat(path)
        except OSError as error:
            raise CacheError(f"could not inspect Clang {path}: {error}") from error
        fingerprint = {
            "path": path,
            "device": information.st_dev,
            "inode": information.st_ino,
            "size": information.st_size,
            "mtime_ns": information.st_mtime_ns,
        }
        self._ensure_root()
        identity_path = self.root / ".toolchain.json"
        lock_path = self.root / ".toolchain.lock"

        flags = os.O_RDWR | os.O_CREAT
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            lock_descriptor = os.open(lock_path, flags, 0o600)
        except OSError as error:
            raise CacheError(f"could not open toolchain cache lock: {error}") from error
        try:
            os.fchmod(lock_descriptor, 0o600)
            fcntl.flock(lock_descriptor, fcntl.LOCK_EX)
            if identity_path.exists():
                try:
                    cached = self._read_json(identity_path)
                    if cached.get("fingerprint") == fingerprint and isinstance(
                        cached.get("identity"), dict
                    ):
                        return path, cached["identity"]
                except (CacheError, OSError):
                    pass

            def query(*arguments: str) -> str:
                try:
                    process = subprocess.run(
                        [path, *arguments],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        check=False,
                    )
                except OSError as error:
                    raise CacheError(f"could not inspect Clang {path}: {error}") from error
                if process.returncode != 0:
                    detail = process.stderr.strip() or process.stdout.strip()
                    raise CacheError(
                        f"could not inspect Clang {path} ({' '.join(arguments)}): {detail}"
                    )
                return process.stdout.strip()

            identity = {
                "path": path,
                "version": query("--version"),
                "target_triple": query("-dumpmachine"),
            }
            value = {"schema": 1, "fingerprint": fingerprint, "identity": identity}
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=".toolchain-", suffix=".json", dir=self.root
            )
            temporary_path = Path(temporary_name)
            try:
                os.fchmod(descriptor, 0o600)
                with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                    descriptor = -1
                    json.dump(value, stream, indent=2, sort_keys=True)
                    stream.write("\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary_path, identity_path)
            finally:
                if descriptor >= 0:
                    os.close(descriptor)
                temporary_path.unlink(missing_ok=True)
            return path, identity
        finally:
            fcntl.flock(lock_descriptor, fcntl.LOCK_UN)
            os.close(lock_descriptor)

    def plan(
        self,
        ir: PixelIR,
        *,
        clang: str | os.PathLike[str] | None = None,
        identity_elimination: bool = True,
        lut_composition: bool = True,
    ) -> CachePlan:
        generated = generate_c(
            ir,
            identity_elimination=identity_elimination,
            lut_composition=lut_composition,
        )
        clang_path, toolchain = self._toolchain_identity(clang)
        system = platform.system()
        link_mode = {"Darwin": "-dynamiclib", "Linux": "-shared"}.get(system)
        if link_mode is None:
            raise CacheError(
                f"native kernel caching is unsupported on {system or 'this platform'}"
            )
        key_inputs: dict[str, Any] = {
            "schema": CACHE_SCHEMA_VERSION,
            "compiler_version": VERSION,
            "kernel_abi_version": KERNEL_ABI_VERSION,
            "pixel_format": get_layout(generated.layout).abi_id,
            "ffmpeg_abi": FFMPEG_FUSED_ABI_IDENTIFIER,
            "source_plan_hash": generated.plan_hash,
            "optimized_ir": generated.passes.ir.canonical_dict(),
            "optimized_plan_hash": generated.optimized_plan_hash,
            "generated_c_sha256": hashlib.sha256(
                generated.source.encode("utf-8")
            ).hexdigest(),
            "kernel_abi_header_sha256": _sha256_file(
                Path(__file__).resolve().parents[1] / "runtime" / "kernel_abi.h"
            ),
            "host": {
                "system": system,
                "architecture": platform.machine(),
                # The Week 6 backend emits a portable baseline variant.  No
                # host feature flags are passed to Clang yet.
                "cpu_features": "baseline",
                "library_suffix": library_suffix(),
            },
            "toolchain": toolchain,
            # Link libraries are deliberately outside the key: they decide
            # whether the artifact resolves, never what Clang emits, and the
            # artifact itself is checksum-validated on every hit.
            "codegen_flags": [*BASE_COMPILER_FLAGS, link_mode],
        }
        serialized = json.dumps(
            key_inputs, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("ascii")
        # Manifests round-trip through JSON, so retain the JSON-shaped value
        # (not Python tuples) for exact validation on a later process.
        key_inputs = json.loads(serialized)
        return CachePlan(
            hashlib.sha256(serialized).hexdigest(),
            key_inputs,
            generated,
            clang_path,
        )

    def _paths(self, key: str) -> tuple[Path, Path, Path, Path]:
        suffix = library_suffix()
        return (
            self.root / f"{key}{suffix}",
            self.root / f"{key}.c",
            self.root / f"{key}.json",
            self.root / f"{key}.lock",
        )

    @contextmanager
    def _lock(self, key: str, *, blocking: bool = True) -> Iterator[None]:
        self._ensure_root()
        lock_path = self._paths(key)[3]
        flags = os.O_RDWR | os.O_CREAT
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(lock_path, flags, 0o600)
        except OSError as error:
            raise CacheError(f"could not open cache lock {lock_path}: {error}") from error
        try:
            information = os.fstat(descriptor)
            if not stat.S_ISREG(information.st_mode):
                raise CacheError(f"cache lock is not a regular file: {lock_path}")
            if hasattr(os, "geteuid") and information.st_uid != os.geteuid():
                raise CacheError(
                    f"cache lock is not owned by the current user: {lock_path}"
                )
            os.fchmod(descriptor, 0o600)
            operation = fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB)
            fcntl.flock(descriptor, operation)
            try:
                yield
            finally:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        _regular_private_file(path)
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise CacheError(f"invalid cache manifest {path}: {error}") from error
        if not isinstance(value, dict):
            raise CacheError(f"invalid cache manifest {path}: expected an object")
        return value

    def _validate_manifest_files(
        self, key: str, manifest: dict[str, Any]
    ) -> tuple[Path, Path, int]:
        library, source, manifest_path, _lock = self._paths(key)
        if manifest.get("schema") != CACHE_SCHEMA_VERSION or manifest.get("key") != key:
            raise CacheError("cache manifest schema or key does not match")
        expected_files = {
            "library": library.name,
            "source": source.name,
        }
        if manifest.get("files") != expected_files:
            raise CacheError("cache manifest filenames do not match the entry key")
        _regular_private_file(library)
        _regular_private_file(source)
        if _sha256_file(library) != manifest.get("library_sha256"):
            raise CacheError("cached native library checksum does not match")
        if _sha256_file(source) != manifest.get("source_sha256"):
            raise CacheError("cached generated-C checksum does not match")
        plan_hash = manifest.get("source_plan_hash")
        if not isinstance(plan_hash, str):
            raise CacheError("cache manifest has no source plan hash")
        layout = manifest.get("layout", DEFAULT_LAYOUT)
        if not isinstance(layout, str) or layout not in LAYOUTS:
            raise CacheError("cache manifest has no usable pixel layout")
        try:
            with NativeKernel(library, plan_hash, layout=layout):
                pass
        except Exception as error:
            raise CacheError(f"cached kernel failed ABI validation: {error}") from error
        size = library.stat().st_size + source.stat().st_size + manifest_path.stat().st_size
        return library, source, size

    def _validate_entry(
        self, plan: CachePlan
    ) -> tuple[Path, Path, Path, dict[str, Any], int]:
        library, source, manifest_path, _lock = self._paths(plan.key)
        manifest = self._read_json(manifest_path)
        if manifest.get("key_inputs") != plan.key_inputs:
            raise CacheError("cache key inputs do not match the current compiler")
        if manifest.get("source_plan_hash") != plan.generated.plan_hash:
            raise CacheError("cached source plan hash does not match")
        if manifest.get("optimized_plan_hash") != plan.generated.optimized_plan_hash:
            raise CacheError("cached optimized plan hash does not match")
        library, source, size = self._validate_manifest_files(plan.key, manifest)
        return library, source, manifest_path, manifest, size

    def _remove_entry_files(self, key: str) -> None:
        library, source, manifest, _lock = self._paths(key)
        for path in (manifest, library, source):
            try:
                path.unlink(missing_ok=True)
            except OSError as error:
                raise CacheError(f"could not remove invalid cache artifact {path}: {error}") from error

    def _cleanup_temporary(self, key: str) -> None:
        """Remove abandoned same-key build directories while its lock is held."""

        for path in self.root.glob(f".tmp-{key[:12]}-*"):
            try:
                if path.is_dir() and not path.is_symlink():
                    shutil.rmtree(path)
                else:
                    path.unlink(missing_ok=True)
            except OSError as error:
                raise CacheError(
                    f"could not remove abandoned cache build {path}: {error}"
                ) from error

    @staticmethod
    def _sync_file(path: Path) -> None:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def _compile_locked(
        self,
        ir: PixelIR,
        plan: CachePlan,
        *,
        compiler: Callable[..., CompiledArtifact | None],
        identity_elimination: bool,
        lut_composition: bool,
        recovery: bool,
    ) -> CachedKernel:
        library, source, manifest_path, _lock = self._paths(plan.key)
        temporary = Path(tempfile.mkdtemp(prefix=f".tmp-{plan.key[:12]}-", dir=self.root))
        os.chmod(temporary, 0o700)
        temporary_library = temporary / library.name
        temporary_source = temporary / source.name
        temporary_manifest = temporary / manifest_path.name
        command: tuple[str, ...] | None = None
        published = False
        try:
            artifact = compiler(
                ir,
                temporary_library,
                source_path=temporary_source,
                clang=plan.clang_path,
                identity_elimination=identity_elimination,
                lut_composition=lut_composition,
            )
            if artifact is not None:
                command = artifact.command
            for path in (temporary_library, temporary_source):
                if not path.is_file():
                    raise CacheError(f"compiler did not create expected artifact: {path}")
                os.chmod(path, 0o600)
                _regular_private_file(path)
            with NativeKernel(
                temporary_library,
                plan.generated.plan_hash,
                layout=plan.generated.layout,
            ):
                pass
            self._sync_file(temporary_library)
            self._sync_file(temporary_source)
            manifest = {
                "schema": CACHE_SCHEMA_VERSION,
                "key": plan.key,
                "key_inputs": plan.key_inputs,
                "source_plan_hash": plan.generated.plan_hash,
                "layout": plan.generated.layout,
                "optimized_plan_hash": plan.generated.optimized_plan_hash,
                "created_ns": time.time_ns(),
                "files": {"library": library.name, "source": source.name},
                "library_sha256": _sha256_file(temporary_library),
                "source_sha256": _sha256_file(temporary_source),
            }
            temporary_manifest.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            os.chmod(temporary_manifest, 0o600)
            self._sync_file(temporary_manifest)

            os.replace(temporary_library, library)
            os.replace(temporary_source, source)
            # The manifest is the commit marker and must always be last.
            os.replace(temporary_manifest, manifest_path)
            published = True
            directory_descriptor = os.open(self.root, os.O_RDONLY)
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
            checked_library, checked_source, _, _manifest, size = (
                self._validate_entry(plan)
            )
            return CachedKernel(
                plan.key,
                checked_library,
                checked_source,
                manifest_path,
                "rebuilt" if recovery else "miss",
                size,
                plan.generated,
                command,
            )
        except Exception:
            if published or not manifest_path.exists():
                self._remove_entry_files(plan.key)
            raise
        finally:
            shutil.rmtree(temporary, ignore_errors=True)

    @contextmanager
    def acquire(
        self,
        ir: PixelIR,
        *,
        clang: str | os.PathLike[str] | None = None,
        identity_elimination: bool = True,
        lut_composition: bool = True,
        compiler: Callable[..., CompiledArtifact | None] = compile_kernel,
    ) -> Iterator[CachedKernel]:
        """Yield a valid entry while holding its per-key usage lock."""

        plan = self.plan(
            ir,
            clang=clang,
            identity_elimination=identity_elimination,
            lut_composition=lut_composition,
        )
        with self._lock(plan.key):
            self._cleanup_temporary(plan.key)
            _library, _source, manifest_path, _lock = self._paths(plan.key)
            recovery = False
            if manifest_path.exists():
                try:
                    library, source, manifest, _value, size = self._validate_entry(plan)
                    yield CachedKernel(
                        plan.key,
                        library,
                        source,
                        manifest,
                        "hit",
                        size,
                        plan.generated,
                    )
                    return
                except (CacheError, OSError):
                    recovery = True
                    self._remove_entry_files(plan.key)
            else:
                # Clean an interrupted manifest-last publication.
                library, source, _manifest, _unused = self._paths(plan.key)
                if library.exists() or source.exists():
                    recovery = True
                    self._remove_entry_files(plan.key)
            cached = self._compile_locked(
                ir,
                plan,
                compiler=compiler,
                identity_elimination=identity_elimination,
                lut_composition=lut_composition,
                recovery=recovery,
            )
            yield cached

    def ensure(self, ir: PixelIR, **options: Any) -> CachedKernel:
        with self.acquire(ir, **options) as cached:
            return cached

    def probe(self, ir: PixelIR, **options: Any) -> dict[str, Any]:
        plan = self.plan(ir, **options)
        result: dict[str, Any] = {"key": plan.key, "status": "miss", "path": None}
        if not self.root.exists():
            return result
        self._ensure_root()
        library, _source, manifest, _lock = self._paths(plan.key)
        if not manifest.exists():
            if library.exists():
                result.update(status="corrupt", path=str(library))
            return result
        try:
            checked, _source, _manifest, _value, _size = self._validate_entry(plan)
            result.update(status="hit", path=str(checked))
        except (CacheError, OSError) as error:
            result.update(status="corrupt", path=str(library), detail=str(error))
        return result

    def list_entries(self) -> list[CacheEntry]:
        self._ensure_root()
        entries: list[CacheEntry] = []
        for manifest_path in sorted(self.root.glob("[0-9a-f]" * 64 + ".json")):
            key = manifest_path.stem
            try:
                manifest = self._read_json(manifest_path)
                library, _source, size = self._validate_manifest_files(key, manifest)
                entries.append(
                    CacheEntry(
                        key,
                        str(manifest.get("source_plan_hash", "")),
                        str(manifest.get("optimized_plan_hash", "")),
                        "valid",
                        size,
                        int(manifest.get("created_ns", 0)),
                        library,
                    )
                )
            except (CacheError, OSError, TypeError, ValueError) as error:
                size = 0
                for path in self._paths(key)[:3]:
                    try:
                        size += path.lstat().st_size
                    except OSError:
                        pass
                entries.append(
                    CacheEntry(key, "", "", "corrupt", size, 0, None, str(error))
                )
        return entries

    def prune(self, max_size: int) -> PruneResult:
        if isinstance(max_size, bool) or not isinstance(max_size, int) or max_size < 0:
            raise CacheError("maximum cache size must be a nonnegative integer")
        entries = self.list_entries()
        before = sum(entry.size for entry in entries)
        current = before
        removed = 0
        skipped = 0
        for entry in sorted(entries, key=lambda item: (item.created_ns, item.key)):
            if current <= max_size:
                break
            try:
                with self._lock(entry.key, blocking=False):
                    # Recalculate under the lock; an entry may have been rebuilt.
                    matching = next(
                        (item for item in self.list_entries() if item.key == entry.key),
                        None,
                    )
                    if matching is None:
                        continue
                    self._remove_entry_files(entry.key)
                    current -= matching.size
                    removed += 1
            except BlockingIOError:
                skipped += 1
        return PruneResult(before, max(0, current), removed, skipped)
