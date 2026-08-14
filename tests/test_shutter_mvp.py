from __future__ import annotations

import unittest
from unittest import mock

from mvp.shutter import (
    CandidateIsland,
    EXPECTED_FFMPEG_VERSION,
    ShutterIntegrationError,
    _probe_formats,
    find_filter_complex_argument,
    normalize_shutter_chain,
    prepare_shutter_graph,
    run_shutter_ffmpeg,
)
from lavfi_cc.filters import FILTER_FORMATS
from lavfi_cc.parser import parse_filtergraph


DEMO_GRAPH = (
    "[0:v]"
    "curves=master='0/0 0.125/0.125 0.25/0.25 0.375/0.375 "
    "0.5/0.5 0.7/0.75 1/1',"
    "curves=master='0/0 0.45/0.55 1/1',"
    "curves=master='0/0 0.25/0.3 0.5/0.5 0.75/0.75 0.875/0.875 1/1',"
    "colorlevels=rimax=0.9:gimax=0.9:bimax=0.9,"
    "colorlevels=rimin=0.05:gimin=0.05:bimin=0.05,"
    "colortemperature=5200:pl=1"
    "[out]"
)


class ShutterGraphTests(unittest.TestCase):
    def test_normalizes_the_positional_options_shutter_emits(self) -> None:
        source = (
            "[0:v]colortemperature=5200:pl=1,"
            "vibrance=0.4:rbal=1.1:gbal=.9:bbal=1[out]"
        )
        self.assertEqual(
            normalize_shutter_chain(source),
            "colortemperature=temperature=5200:pl=1,"
            "vibrance=intensity=0.4:rbal=1.1:gbal=.9:bbal=1",
        )

    def test_requires_one_linear_zero_video_to_out_chain(self) -> None:
        invalid = (
            "curves=preset=darker",
            "[1:v]curves=preset=darker[out]",
            "[0:v]curves=preset=darker[mid]",
            "[0:v]curves=preset=darker[a];[a]negate[out]",
        )
        for source in invalid:
            with self.subTest(source=source):
                with self.assertRaises(ShutterIntegrationError):
                    normalize_shutter_chain(source)

    def test_finds_exactly_one_separate_filter_complex(self) -> None:
        location = find_filter_complex_argument(
            ("-i", "in.mp4", "-filter_complex", DEMO_GRAPH, "-map", "[out]")
        )
        self.assertEqual((location.option_index, location.value_index), (2, 3))
        with self.assertRaisesRegex(ShutterIntegrationError, "no -filter_complex"):
            find_filter_complex_argument(("-i", "in.mp4", "-vf", "negate"))
        with self.assertRaisesRegex(ShutterIntegrationError, "multiple"):
            find_filter_complex_argument(
                ("-filter_complex", DEMO_GRAPH, "-filter_complex", DEMO_GRAPH)
            )

    @mock.patch("mvp.shutter.subprocess.run")
    def test_export_probe_keeps_input_and_encoder_but_never_the_output(
        self, run: mock.Mock
    ) -> None:
        run.return_value = mock.Mock(
            returncode=0,
            stderr=(
                "[showinfo@lavfi_cc_probe_0 @ 0x1] n: 0 pts: 0 "
                "fmt:rgb24 s:16x16\n"
            ),
        )
        filters = parse_filtergraph(
            "curves=preset=darker,curves=preset=lighter"
        ).filters
        candidate = CandidateIsland(
            0,
            2,
            ("curves", "curves"),
            FILTER_FORMATS["curves"],
        )
        formats = _probe_formats(
            "/fake/ffmpeg",
            filters,
            (candidate,),
            ffmpeg_arguments=(
                "-ss",
                "30",
                "-i",
                "/video/input.mp4",
                "-filter_complex",
                DEMO_GRAPH,
                "-map",
                "[out]",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "/video/output.mp4",
            ),
        )
        self.assertEqual(formats, ("rgb24",))
        command = run.call_args.args[0]
        self.assertIn("/video/input.mp4", command)
        self.assertIn("libx264", command)
        self.assertIn("yuv420p", command)
        self.assertNotIn("/video/output.mp4", command)
        self.assertEqual(command[-3:], ["-f", "null", "-"])

    @mock.patch("mvp.shutter._ffmpeg_version", return_value=EXPECTED_FFMPEG_VERSION)
    @mock.patch("mvp.shutter._probe_formats", return_value=("rgb24",))
    def test_demo_becomes_one_six_filter_rgb24_island(
        self, _probe: mock.Mock, _version: mock.Mock
    ) -> None:
        prepared = prepare_shutter_graph(DEMO_GRAPH, ffmpeg="/fake/ffmpeg")
        self.assertEqual(len(prepared.pins), 1)
        self.assertEqual(prepared.pins[0].pixel_format, "rgb24")
        self.assertEqual(
            prepared.pins[0].filters,
            (
                "curves",
                "curves",
                "curves",
                "colorlevels",
                "colorlevels",
                "colortemperature",
            ),
        )
        self.assertEqual(prepared.analysis.eliminated_passes, 5)
        self.assertIn("format=rgb24,fused=", prepared.placeholder_filtergraph)
        self.assertNotIn("colortemperature", prepared.placeholder_filtergraph)

    @mock.patch("mvp.shutter._ffmpeg_version", return_value=EXPECTED_FFMPEG_VERSION)
    @mock.patch(
        "mvp.shutter._probe_formats", return_value=("rgb24", "rgb24")
    )
    def test_unsupported_filter_is_retained_between_maximal_islands(
        self, _probe: mock.Mock, _version: mock.Mock
    ) -> None:
        source = (
            "[0:v]curves=preset=darker,curves=preset=lighter,exposure=.5,"
            "colorlevels=rimin=.05,colorlevels=rimax=.95[out]"
        )
        prepared = prepare_shutter_graph(source, ffmpeg="/fake/ffmpeg")
        self.assertEqual(len(prepared.pins), 2)
        self.assertEqual(len(prepared.analysis.plans), 2)
        self.assertIn(",exposure=.5,", prepared.placeholder_filtergraph)
        self.assertEqual(prepared.placeholder_filtergraph.count("fused="), 2)

    @mock.patch("mvp.shutter._ffmpeg_version", return_value="n9.0")
    def test_refuses_an_unpinned_ffmpeg(self, _version: mock.Mock) -> None:
        with self.assertRaisesRegex(ShutterIntegrationError, "expected pinned"):
            prepare_shutter_graph(DEMO_GRAPH, ffmpeg="/fake/ffmpeg")


class ShutterFallbackTests(unittest.TestCase):
    @mock.patch("mvp.shutter.resolve_ffmpeg", return_value="/fake/ffmpeg")
    @mock.patch("mvp.shutter._execute", return_value=23)
    @mock.patch(
        "mvp.shutter.prepare_shutter_graph",
        side_effect=ShutterIntegrationError("not safe"),
    )
    def test_default_fallback_runs_the_original_arguments_only(
        self, _prepare: mock.Mock, execute: mock.Mock, _resolve: mock.Mock
    ) -> None:
        arguments = ("-filter_complex", DEMO_GRAPH, "-map", "[out]")
        self.assertEqual(run_shutter_ffmpeg(arguments), 23)
        execute.assert_called_once_with("/fake/ffmpeg", arguments)

    @mock.patch("mvp.shutter.resolve_ffmpeg", return_value="/fake/ffmpeg")
    @mock.patch("mvp.shutter._execute")
    @mock.patch(
        "mvp.shutter.prepare_shutter_graph",
        side_effect=ShutterIntegrationError("not safe"),
    )
    def test_strict_fallback_does_not_execute_ffmpeg(
        self, _prepare: mock.Mock, execute: mock.Mock, _resolve: mock.Mock
    ) -> None:
        arguments = ("-filter_complex", DEMO_GRAPH, "-map", "[out]")
        self.assertEqual(
            run_shutter_ffmpeg(arguments, require_fusion=True), 2
        )
        execute.assert_not_called()


if __name__ == "__main__":
    unittest.main()
