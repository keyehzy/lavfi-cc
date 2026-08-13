from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import shutil
import tempfile
import threading
import time
import unittest

from lavfi_cc.cache import CacheError, KernelCache
from lavfi_cc.frontend import require_ir
from lavfi_cc.native import CompilationError, NativeKernel, compile_kernel


HAS_CLANG = shutil.which("clang") is not None


def graph(region: str) -> str:
    return f"format=rgba,{region},format=rgba"


@unittest.skipUnless(HAS_CLANG, "Week 6 cache tests require Clang")
class KernelCacheTests(unittest.TestCase):
    def test_key_is_stable_and_covers_codegen_inputs(self) -> None:
        ir = require_ir(graph("lutrgb=r=val,negate"))
        with tempfile.TemporaryDirectory() as directory:
            cache = KernelCache(Path(directory) / "cache")
            first = cache.plan(ir)
            second = cache.plan(ir)
            disabled = cache.plan(
                ir, identity_elimination=False, lut_composition=False
            )
        self.assertEqual(first.key, second.key)
        self.assertNotEqual(first.key, disabled.key)
        self.assertEqual(first.key_inputs["compiler_version"], "0.6.0")
        self.assertIn("version", first.key_inputs["toolchain"])
        self.assertIn("target_triple", first.key_inputs["toolchain"])
        self.assertEqual(first.key_inputs["host"]["cpu_features"], "baseline")
        self.assertEqual(first.key_inputs["ffmpeg_abi"], "lavfi-fused-avfilter-v1")

    def test_miss_then_checked_hit_compiles_only_once(self) -> None:
        ir = require_ir(graph("negate"))
        calls = 0

        def counting_compiler(*arguments: object, **options: object):
            nonlocal calls
            calls += 1
            return compile_kernel(*arguments, **options)

        with tempfile.TemporaryDirectory() as directory:
            cache = KernelCache(Path(directory) / "cache")
            first = cache.ensure(ir, compiler=counting_compiler)
            second = cache.ensure(ir, compiler=counting_compiler)
            self.assertEqual(first.status, "miss")
            self.assertEqual(second.status, "hit")
            self.assertEqual(first.key, second.key)
            self.assertEqual(calls, 1)
            self.assertEqual(os.stat(first.library_path).st_mode & 0o777, 0o600)
            self.assertEqual(os.stat(cache.root).st_mode & 0o777, 0o700)
            with NativeKernel(second.library_path, ir.plan_hash) as kernel:
                self.assertEqual(
                    kernel.process_rgba8(bytes((0, 1, 2, 3)), 1, 1),
                    bytes((255, 254, 253, 3)),
                )

    def test_corrupt_library_is_deleted_and_rebuilt(self) -> None:
        ir = require_ir(graph("negate=components=a"))
        calls = 0

        def counting_compiler(*arguments: object, **options: object):
            nonlocal calls
            calls += 1
            return compile_kernel(*arguments, **options)

        with tempfile.TemporaryDirectory() as directory:
            cache = KernelCache(Path(directory) / "cache")
            first = cache.ensure(ir, compiler=counting_compiler)
            first.library_path.write_bytes(b"corrupt")
            os.chmod(first.library_path, 0o600)
            recovered = cache.ensure(ir, compiler=counting_compiler)
            self.assertEqual(recovered.status, "rebuilt")
            self.assertEqual(calls, 2)
            with NativeKernel(recovered.library_path, ir.plan_hash):
                pass

    def test_failed_compile_publishes_no_partial_entry(self) -> None:
        ir = require_ir(graph("lutrgb=r=negval"))

        def failing_compiler(
            _ir: object, output: object, *, source_path: object, **_options: object
        ) -> None:
            Path(output).write_bytes(b"partial library")
            Path(source_path).write_text("partial source")
            raise CompilationError("injected compiler failure")

        with tempfile.TemporaryDirectory() as directory:
            cache = KernelCache(Path(directory) / "cache")
            with self.assertRaisesRegex(CompilationError, "injected"):
                cache.ensure(ir, compiler=failing_compiler)
            names = {path.name for path in cache.root.iterdir()}
            self.assertFalse(any(name.endswith((".dylib", ".so", ".c")) for name in names), names)
            self.assertFalse(any(name.startswith(".tmp-") for name in names), names)
            cached = cache.ensure(ir)
            self.assertEqual(cached.status, "miss")

    def test_concurrent_requests_compile_one_artifact(self) -> None:
        ir = require_ir(graph("negate,lutrgb=r=val+1"))
        calls = 0
        guard = threading.Lock()

        def slow_compiler(*arguments: object, **options: object):
            nonlocal calls
            with guard:
                calls += 1
            time.sleep(0.1)
            return compile_kernel(*arguments, **options)

        with tempfile.TemporaryDirectory() as directory:
            cache = KernelCache(Path(directory) / "cache")
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = [
                    executor.submit(cache.ensure, ir, compiler=slow_compiler)
                    for _ in range(2)
                ]
                results = [future.result() for future in futures]
            self.assertEqual(calls, 1)
            self.assertEqual(sorted(item.status for item in results), ["hit", "miss"])
            self.assertEqual(len(cache.list_entries()), 1)

    def test_insecure_cache_root_is_rejected(self) -> None:
        ir = require_ir(graph("negate"))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "cache"
            root.mkdir(mode=0o700)
            os.chmod(root, 0o755)
            with self.assertRaisesRegex(CacheError, "mode 0700"):
                KernelCache(root).ensure(ir)

    def test_prune_removes_oldest_entries_to_the_limit(self) -> None:
        first_ir = require_ir(graph("negate"))
        second_ir = require_ir(graph("negate=components=a"))
        with tempfile.TemporaryDirectory() as directory:
            cache = KernelCache(Path(directory) / "cache")
            first = cache.ensure(first_ir)
            time.sleep(0.002)
            second = cache.ensure(second_ir)
            result = cache.prune(second.size)
            self.assertGreaterEqual(result.before_size, first.size + second.size)
            self.assertEqual(result.removed_entries, 1)
            entries = cache.list_entries()
            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0].key, second.key)

    def test_prune_skips_an_entry_while_it_is_acquired(self) -> None:
        ir = require_ir(graph("negate"))
        acquired = threading.Event()
        release = threading.Event()
        with tempfile.TemporaryDirectory() as directory:
            cache = KernelCache(Path(directory) / "cache")
            cached = cache.ensure(ir)

            def hold_entry() -> None:
                with cache.acquire(ir):
                    acquired.set()
                    release.wait(timeout=5)

            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(hold_entry)
                self.assertTrue(acquired.wait(timeout=5))
                result = cache.prune(0)
                self.assertEqual(result.removed_entries, 0)
                self.assertEqual(result.skipped_locked, 1)
                self.assertTrue(cached.library_path.exists())
                release.set()
                future.result(timeout=5)
            result = cache.prune(0)
            self.assertEqual(result.removed_entries, 1)


if __name__ == "__main__":
    unittest.main()
