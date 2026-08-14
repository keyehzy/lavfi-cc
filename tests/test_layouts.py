from __future__ import annotations

import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest

from lavfi_cc.filters import filter_supports_format
from lavfi_cc.frontend import analyze_filtergraph
from lavfi_cc.interpreter import interpret_rgba8
from lavfi_cc.layouts import LAYOUTS, PixelLayout, get_layout
from lavfi_cc.native import NativeKernel, compile_kernel, library_suffix
from lavfi_cc.parser import parse_filtergraph


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FFMPEG_CANDIDATES = (
    ROOT / ".build" / "ffmpeg-macos" / "bin" / "ffmpeg",
    ROOT / ".build" / "ffmpeg-linux" / "bin" / "ffmpeg",
)
WEEK5_CANDIDATES = (
    ROOT / ".build" / "ffmpeg-week5-macos" / "bin" / "ffmpeg",
    ROOT / ".build" / "ffmpeg-week5-linux" / "bin" / "ffmpeg",
)

# One chain per accepted filter plus a combined one, so every RGB layout is
# exercised against every lowering.  The float32 filters read all three colour
# channels, while curves is channel-independent upstream and remains a table.
RGB_CHAINS = (
    "negate",
    "lutrgb=r=val*1.125-7:g=negval:b='clip(val,13,241)'",
    "colorlevels=rimin=.05:gimax=.9:preserve=none",
    "colorchannelmixer=rr=.9:rg=.1:gg=.8:gb=.2:bb=1:aa=1:pc=none",
    "colorbalance=rs=.3:gm=-.2:bh=.5:rm=.1:gh=.4:bs=-.35",
    "colorcontrast=rc=.4:gm=.25:by=-.15:rcw=.9:gmw=.7:byw=.3:pl=.35",
    "vibrance=intensity=.4:rbal=.8:gbal=1.2:bbal=-.4:alternate=1",
    "colortemperature=temperature=5500:mix=.8:pl=.3",
    "selectivecolor=reds='0.1 0 -0.1':blacks='0 .2 0 -.1'",
    "selectivecolor=correction_method=relative:blues='-.4 .1 .2 0':"
    "neutrals='0 .2 -.2 .3'",
    "selectivecolor=yellows='.1 0 -.2 .05':greens='-.2 .1 0 0':"
    "cyans='0 -.15 .2 .1':magentas='.1 .2 -.1 0':whites='-.1 0 .1 .2'",
    "curves=r='0/0 0.5/0.4 1/1':g='0/0.1 1/0.9':b='0/0 0.3/0.5 1/1'",
    "curves=preset=vintage:interp=pchip",
    "negate,lutrgb=r=val*1.08+2,colorlevels=rimin=.05,"
    "colorchannelmixer=rr=.9:rg=.1:ba=.3",
    "curves=m='0/0 0.5/0.6 1/1',colorbalance=rs=.2:bh=-.4,"
    "colorcontrast=rc=.5:rcw=1:gmw=.25,negate",
    "vibrance=intensity=-.35,colortemperature=temperature=7600:mix=.7:pl=.2,negate",
    "vibrance=intensity=.25,colortemperature=temperature=7200:mix=.6,"
    "selectivecolor=reds='.15 -.1 0 .05',negate",
)

# negate, lutyuv, eq, and hue are the accepted filters that advertise YUV
# upstream; lutrgb, colorlevels, and colorchannelmixer are RGB-only, so a YUV
# run containing one of them is refused rather than fused. Component masks and
# LUT expressions are spelled in the other family's names here. The lutyuv
# expressions exercise each layout's limited or full component range, the eq
# invocations cover each of its three upstream code paths, hue rotates Cb into Cr so its
# two chroma planes have to be walked in one loop, and on a subsampled layout
# each sampling group is walked at its own resolution.
YUV_CHAINS = (
    "negate",
    "negate=components=y",
    "negate=components=u+v",
    "negate=components=y+u+v",
    "lutyuv=y=negval:u=negval:v=negval",
    "lutyuv=y=minval+maxval-val:u='clip(val,100,180)'",
    "lutyuv=y=val*1.125-7:v=maxval-val",
    "eq=contrast=1.4:brightness=-0.15",
    "eq=saturation=0.6",
    "eq=gamma=2.2:gamma_weight=0.7",
    "eq=contrast=1.2:brightness=0.05:saturation=1.3:gamma=1.8:gamma_r=1.4",
    "hue=h=45",
    "hue=h=110:s=1.6:b=0.4",
    "hue=H=2.1:s=-1.2",
    "negate,lutyuv=y=negval:u=val*0.9+12,eq=contrast=1.3:saturation=0.8,"
    "hue=h=25:s=1.2:b=-0.3,negate=components=v",
)

# The yuva layouts run everything above plus the chains that reach alpha, which
# the alpha-less trio cannot express at all: negate=components=…a is a hard
# configuration failure there, and lutyuv's alpha table is never applied because
# upstream only builds one per component the format actually carries. Alpha is
# full resolution while chroma need not be, so these also put a stored channel
# in a sampling group that is neither luma's plane nor the chroma pair.
YUVA_CHAINS = YUV_CHAINS + (
    "negate=components=a",
    "negate=components=y+u+v+a",
    "lutyuv=y=negval:a=negval",
    "lutyuv=a='clip(val,16,235)':u=val*0.9+12",
    "eq=contrast=1.4:saturation=0.7,negate=components=a",
    "hue=h=35:s=1.3:b=0.2,lutyuv=a=negval",
    "negate,lutyuv=y=negval:a=val*0.75+16,eq=contrast=1.3:saturation=0.8,"
    "hue=h=25:s=1.2:b=-0.3,negate=components=v+a",
)


# Chains written for the wider range itself: a lutyuv expression whose bounds
# only exist above eight bits, and a curve whose key points land on entries a
# 256-entry table has not got.
HIGH_DEPTH_YUV_CHAINS = (
    "lutyuv=y='clip(val,64,940)':u='clip(val,64,960)'",
    "lutyuv=y=minval+maxval-val:v=negval",
    "negate=components=y,lutyuv=u=val*1.03125+3",
)
HIGH_DEPTH_RGB_CHAINS = (
    "curves=r='0/0.05 1/1'",
    "lutrgb=r=negval:g='clip(val,300,3000)'",
    "colorlevels=rimin=.05:gimax=.9,colorbalance=rs=.25:bh=-.3",
    "vibrance=intensity=-.45:rbal=.8:gbal=1.1,"
    "colortemperature=temperature=8400:mix=.65:pl=.25,negate",
    "vibrance=intensity=.3,colortemperature=temperature=5800:mix=.8,"
    "selectivecolor=correction_method=relative:reds='.2 -.1 0 .05',negate",
)


def filters_in(chain: str) -> tuple[str, ...]:
    return tuple(
        invocation.name for invocation in parse_filtergraph(chain).filters
    )


def chains_for(name: str) -> tuple[str, ...]:
    """Every chain above that is contiguous in *name*.

    A chain is only a differential case where every filter in it advertises the
    format, since otherwise FFmpeg converts in the middle of the run and the
    compiler refuses it -- which is the same rule the frontend applies, asked
    here rather than restated. It does most of its work above eight bits, where
    ``eq`` drops out of every chain and ``hue`` out of all but the ten-bit ones.
    """

    layout = get_layout(name)
    if layout.family != "yuv":
        candidates = RGB_CHAINS + (HIGH_DEPTH_RGB_CHAINS if layout.high_depth else ())
    else:
        candidates = YUVA_CHAINS if layout.has_alpha else YUV_CHAINS
        if layout.high_depth:
            candidates += HIGH_DEPTH_YUV_CHAINS
    return tuple(
        chain
        for chain in candidates
        if all(filter_supports_format(item, name) for item in filters_in(chain))
    )


def random_frame(
    layout: PixelLayout, width: int, height: int, generator: random.Random
) -> bytes:
    """One tightly packed frame of samples the format can actually hold.

    Above eight bits a sample is a little-endian word bounded by the format's
    own maximum rather than sixteen random bits. That bound is the domain the
    kernels are defined over, and it is not a convenience: ``vf_curves.c`` and
    ``vf_colorchannelmixer.c`` index a ``1 << depth``-entry table with the raw
    sample, so a wider value makes the oracle itself read past the end of its
    table and there is no answer to be exact against.
    """

    size = layout.frame_size(width, height)
    if not layout.high_depth:
        return bytes(generator.randrange(256) for _ in range(size))
    return b"".join(
        generator.randrange(layout.max_value + 1).to_bytes(2, "little")
        for _ in range(size // 2)
    )


def _executable(candidates: tuple[Path, ...], override: str | None) -> Path | None:
    paths = (Path(override),) if override else candidates
    return next(
        (path for path in paths if path.is_file() and os.access(path, os.X_OK)),
        None,
    )


def ffmpeg_path() -> Path | None:
    return _executable(DEFAULT_FFMPEG_CANDIDATES, os.environ.get("LAVFI_CC_FFMPEG"))


def week5_ffmpeg_path() -> Path | None:
    return _executable(
        WEEK5_CANDIDATES,
        os.environ.get("LAVFI_CC_WEEK5_FFMPEG") or os.environ.get("LAVFI_CC_WEEK6_FFMPEG"),
    )


class LayoutTableTests(unittest.TestCase):
    def test_packed_offsets_are_a_permutation_within_the_pixel(self) -> None:
        for name, layout in LAYOUTS.items():
            if layout.planar:
                continue
            with self.subTest(layout=name):
                offsets = [value for value in layout.offsets if value is not None]
                self.assertEqual(sorted(offsets), list(range(layout.step)))
                self.assertEqual(layout.plane_count, 1)

    def test_planar_layouts_use_one_plane_per_channel(self) -> None:
        for name in ("gbrp", "gbrap"):
            with self.subTest(layout=name):
                layout = get_layout(name)
                self.assertTrue(layout.planar)
                self.assertEqual(layout.step, 1)
                planes = [value for value in layout.planes if value is not None]
                self.assertEqual(sorted(planes), list(range(layout.plane_count)))
                self.assertTrue(all(offset in (0, None) for offset in layout.offsets))
        # Plane 0 is green, 1 is blue, 2 is red.
        self.assertEqual(get_layout("gbrp").planes, (2, 0, 1, None))

    def test_alpha_less_layouts_store_only_three_components(self) -> None:
        self.assertEqual(get_layout("rgb24").stored_channels, (0, 1, 2))
        self.assertEqual(get_layout("rgba").stored_channels, (0, 1, 2, 3))
        self.assertEqual(get_layout("gbrp").stored_channels, (0, 1, 2))
        self.assertFalse(get_layout("bgr24").has_alpha)
        self.assertTrue(get_layout("gbrap").has_alpha)

    def test_frame_size_covers_every_plane(self) -> None:
        self.assertEqual(get_layout("rgba").frame_size(4, 2), 32)
        self.assertEqual(get_layout("rgb24").frame_size(4, 2), 24)
        self.assertEqual(get_layout("gbrp").frame_size(4, 2), 24)
        self.assertEqual(get_layout("gbrap").frame_size(4, 2), 32)
        # Planes of a tightly packed frame sit back to back.
        self.assertEqual(get_layout("gbrp").plane_origin(2, 4, 2), 16)

    def test_yuv_layouts_use_luma_cb_cr_planes(self) -> None:
        for name in (
            "yuv444p", "yuv422p", "yuv420p", "yuv411p", "yuv410p", "yuv440p",
            "yuvj444p", "yuvj422p", "yuvj420p", "yuvj440p", "yuv440p10le",
        ):
            with self.subTest(layout=name):
                layout = get_layout(name)
                self.assertEqual(layout.family, "yuv")
                self.assertEqual(layout.planes, (0, 1, 2, None))
                self.assertEqual(layout.plane_count, 3)
                self.assertEqual(layout.step, 1)
                self.assertFalse(layout.has_alpha)
                self.assertEqual(set(layout.components), {"y", "u", "v"})

    def test_remaining_yuv_ratios_have_their_descriptor_shifts(self) -> None:
        expected = {
            "yuv411p": (2, 0),
            "yuv410p": (2, 2),
            "yuv440p": (0, 1),
            "yuv440p10le": (0, 1),
        }
        for name, chroma in expected.items():
            with self.subTest(layout=name):
                self.assertEqual(
                    get_layout(name).subsampling,
                    ((0, 0), chroma, chroma, (0, 0)),
                )

    def test_only_yuvj_layouts_are_intrinsically_full_range(self) -> None:
        for name in ("yuvj444p", "yuvj422p", "yuvj420p", "yuvj440p"):
            with self.subTest(layout=name):
                self.assertTrue(get_layout(name).full_range)
        for name in ("yuv444p", "yuv411p", "yuv410p", "yuv440p10le"):
            with self.subTest(layout=name):
                self.assertFalse(get_layout(name).full_range)

    def test_yuva_layouts_add_a_full_resolution_alpha_plane(self) -> None:
        # This table was read out of the pinned binary the way every other
        # layout's was: one negate per component over a frame of four distinct
        # plane values moves plane 0 for y, 1 for u, 2 for v, and 3 for a. The
        # differential tests below are what hold it to the oracle.
        for name, chroma in (("yuva444p", (0, 0)), ("yuva422p", (1, 0)),
                             ("yuva420p", (1, 1))):
            with self.subTest(layout=name):
                layout = get_layout(name)
                self.assertEqual(layout.family, "yuv")
                self.assertEqual(layout.planes, (0, 1, 2, 3))
                self.assertEqual(layout.plane_count, 4)
                self.assertEqual(layout.step, 1)
                self.assertTrue(layout.has_alpha)
                self.assertEqual(set(layout.components), {"y", "u", "v", "a"})
                # Only chroma shrinks; alpha is sized like luma.
                self.assertEqual(
                    layout.subsampling, ((0, 0), chroma, chroma, (0, 0))
                )
                self.assertEqual(layout.plane_shift(3), (0, 0))

    def test_yuva_alpha_shares_lumas_sampling_group(self) -> None:
        # Alpha and luma have a sample at exactly the same positions, so they
        # are one group; chroma is the other. This is the first layout where a
        # group spans two planes that are not adjacent, and the first that is
        # subsampled and carries alpha at the same time.
        self.assertEqual(get_layout("yuva420p").sampling_groups, ((0, 3), (1, 2)))
        self.assertEqual(get_layout("yuva422p").sampling_groups, ((0, 3), (1, 2)))
        # yuva444p subsamples nothing, so every channel is co-sited.
        self.assertEqual(get_layout("yuva444p").sampling_groups, ((0, 1, 2, 3),))
        self.assertFalse(get_layout("yuva444p").subsampled)
        self.assertTrue(get_layout("yuva420p").subsampled)

    def test_yuva_frame_sizes_shrink_only_the_chroma_planes(self) -> None:
        yuva420p = get_layout("yuva420p")
        self.assertEqual(
            [yuva420p.plane_width(plane, 5) for plane in range(4)], [5, 3, 3, 5]
        )
        self.assertEqual(
            [yuva420p.plane_height(plane, 3) for plane in range(4)], [3, 2, 2, 3]
        )
        self.assertEqual(yuva420p.frame_size(5, 3), 15 + 6 + 6 + 15)
        # The alpha plane sits after both chroma planes.
        self.assertEqual(yuva420p.plane_origin(3, 5, 3), 15 + 6 + 6)
        self.assertEqual(get_layout("yuva422p").frame_size(5, 3), 15 + 9 + 9 + 15)
        self.assertEqual(get_layout("yuva444p").frame_size(5, 3), 60)

    def test_chroma_planes_round_up_like_ffmpeg(self) -> None:
        # AV_CEIL_RSHIFT: an odd dimension keeps a whole trailing chroma
        # sample, so 5x3 in 4:2:0 has 3x2 chroma planes rather than 2x1.
        yuv420p = get_layout("yuv420p")
        self.assertFalse(get_layout("yuv444p").subsampled)
        self.assertTrue(get_layout("yuv422p").subsampled)
        self.assertTrue(yuv420p.subsampled)
        self.assertEqual(
            [yuv420p.plane_width(plane, 5) for plane in range(3)], [5, 3, 3]
        )
        self.assertEqual(
            [yuv420p.plane_height(plane, 3) for plane in range(3)], [3, 2, 2]
        )
        self.assertEqual(yuv420p.frame_size(5, 3), 15 + 6 + 6)
        self.assertEqual(yuv420p.plane_origin(2, 5, 3), 15 + 6)
        # 4:2:2 halves width only.
        self.assertEqual(get_layout("yuv422p").frame_size(5, 3), 15 + 9 + 9)
        self.assertEqual(get_layout("yuv444p").frame_size(5, 3), 45)

    def test_wider_subsampling_shifts_round_up_like_ffmpeg(self) -> None:
        # A 5x5 frame keeps two chroma columns at quarter width and two rows at
        # quarter height.  Vertical-only 4:4:0 keeps all five columns.
        self.assertEqual(get_layout("yuv411p").frame_size(5, 5), 25 + 10 + 10)
        self.assertEqual(get_layout("yuv410p").frame_size(5, 5), 25 + 4 + 4)
        self.assertEqual(get_layout("yuv440p").frame_size(5, 5), 25 + 15 + 15)
        self.assertEqual(
            get_layout("yuv440p10le").frame_size(5, 5), 2 * (25 + 15 + 15)
        )

    def test_unknown_layout_is_rejected(self) -> None:
        with self.assertRaises(KeyError):
            get_layout("nv12")


class LayoutAnalysisTests(unittest.TestCase):
    def graph(self, name: str) -> str:
        return f"format={name},{chains_for(name)[-1]},format={name}"

    def test_explicit_boundaries_select_the_named_layout(self) -> None:
        for name in LAYOUTS:
            with self.subTest(layout=name):
                analysis = analyze_filtergraph(self.graph(name))
                self.assertTrue(analysis.eligible, analysis.diagnostics)
                self.assertEqual(analysis.ir.layout, name)

    def test_each_layout_gets_its_own_plan_hash(self) -> None:
        hashes = {
            name: analyze_filtergraph(f"format={name},negate,format={name}").ir.plan_hash
            for name in LAYOUTS
        }
        self.assertEqual(len(set(hashes.values())), len(LAYOUTS))

    def test_component_names_are_refused_across_families(self) -> None:
        # Upstream validates the mask against the format and fails to
        # configure the graph, so the compiler must refuse it too.
        for layout, mask in (("rgba", "y+u+v"), ("yuv420p", "r+g+b")):
            with self.subTest(layout=layout):
                analysis = analyze_filtergraph(
                    f"format={layout},negate=components={mask},negate,"
                    f"format={layout}"
                )
                self.assertFalse(analysis.eligible)
                self.assertEqual(
                    analysis.diagnostics[0].code, "component_not_available"
                )

    def test_yuv_layouts_refuse_the_rgb_only_filters(self) -> None:
        for name in ("lutrgb=r=val*2", "colorlevels=rimin=.05",
                     "colorchannelmixer=rr=.9"):
            with self.subTest(filter=name):
                analysis = analyze_filtergraph(
                    f"format=yuv420p,{name},format=yuv420p"
                )
                self.assertFalse(analysis.eligible)
                self.assertEqual(
                    analysis.diagnostics[0].code, "format_not_advertised"
                )

    def test_rgb_layouts_refuse_lutyuv(self) -> None:
        # lutyuv advertises no RGB format, so FFmpeg would convert around it.
        for name in ("rgba", "rgb24", "gbrp"):
            with self.subTest(layout=name):
                analysis = analyze_filtergraph(
                    f"format={name},lutyuv=y=negval,format={name}"
                )
                self.assertFalse(analysis.eligible)
                self.assertEqual(
                    analysis.diagnostics[0].code, "format_not_advertised"
                )

    def test_lutyuv_refuses_the_rgb_component_names(self) -> None:
        # Upstream shares one options table between lutrgb and lutyuv, so
        # lutyuv=r=... silently sets the luma expression. Refusing it keeps the
        # spelling from meaning two different things.
        for graph, filter_name in (
            ("format=yuv420p,lutyuv=r=negval,format=yuv420p", "lutyuv"),
            ("format=rgba,lutrgb=y=negval,format=rgba", "lutrgb"),
        ):
            with self.subTest(filter=filter_name):
                analysis = analyze_filtergraph(graph)
                self.assertFalse(analysis.eligible)
                self.assertEqual(
                    analysis.diagnostics[0].code, "unsupported_option"
                )

    def test_lutyuv_and_lutrgb_do_not_share_a_plan(self) -> None:
        # The same expression means different bytes in the two ranges, so the
        # two filters must never collide on one cached kernel.
        yuv = analyze_filtergraph(
            "format=yuv444p,lutyuv=y=negval:u=negval:v=negval,format=yuv444p"
        )
        rgb = analyze_filtergraph(
            "format=rgba,lutrgb=r=negval:g=negval:b=negval,format=rgba"
        )
        self.assertTrue(yuv.eligible, yuv.diagnostics)
        self.assertTrue(rgb.eligible, rgb.diagnostics)
        self.assertNotEqual(yuv.ir.plan_hash, rgb.ir.plan_hash)

    def test_negate_alpha_is_inert_on_yuv(self) -> None:
        # yuv420p has no alpha plane for the plane mask to select, so the
        # option cannot change what negate does there.
        for name in ("yuv444p", "yuv422p", "yuv420p"):
            with self.subTest(layout=name):
                self.assertTrue(
                    analyze_filtergraph(
                        f"format={name},negate=negate_alpha=1,negate,format={name}"
                    ).eligible
                )

    def test_negate_alpha_is_refused_where_it_would_negate_alpha(self) -> None:
        # A plane mask means the legacy option really does negate alpha on
        # planar RGB, unlike packed RGB where the component mask wins. The
        # yuva layouts are planar with an alpha plane, so it lands on them the
        # same way -- and the spelling suggested has to be in their own family.
        for name, spelling in (
            ("gbrap", "components=r+g+b+a"),
            ("yuva444p", "components=y+u+v+a"),
            ("yuva422p", "components=y+u+v+a"),
            ("yuva420p", "components=y+u+v+a"),
        ):
            with self.subTest(layout=name):
                analysis = analyze_filtergraph(
                    f"format={name},negate=negate_alpha=1,negate,format={name}"
                )
                self.assertFalse(analysis.eligible)
                self.assertEqual(
                    analysis.diagnostics[0].code, "planar_negate_alpha"
                )
                self.assertIn(spelling, analysis.diagnostics[0].message)
        # Packed RGB ignores it, and gbrp has no alpha plane to negate.
        for layout in ("rgba", "gbrp"):
            with self.subTest(layout=layout):
                self.assertTrue(
                    analyze_filtergraph(
                        f"format={layout},negate=negate_alpha=1,negate,"
                        f"format={layout}"
                    ).eligible
                )

    def test_alpha_components_follow_the_plane_the_layout_carries(self) -> None:
        # Upstream builds the available-component mask from the descriptor, so
        # naming alpha is a hard configuration failure exactly where there is
        # no alpha plane -- which within the YUV family is what separates
        # yuv420p from yuva420p.
        self.assertTrue(
            analyze_filtergraph(
                "format=yuva420p,negate=components=y+u+v+a,negate,format=yuva420p"
            ).eligible
        )
        analysis = analyze_filtergraph(
            "format=yuv420p,negate=components=y+u+v+a,negate,format=yuv420p"
        )
        self.assertFalse(analysis.eligible)
        self.assertEqual(analysis.diagnostics[0].code, "component_not_available")

    def test_yuva_layouts_still_refuse_the_rgb_component_names(self) -> None:
        # Carrying alpha does not make a YUV layout accept the RGB family.
        for name in ("yuva444p", "yuva422p", "yuva420p"):
            with self.subTest(layout=name):
                analysis = analyze_filtergraph(
                    f"format={name},negate=components=r+g+b,negate,format={name}"
                )
                self.assertFalse(analysis.eligible)
                self.assertEqual(
                    analysis.diagnostics[0].code, "component_not_available"
                )

    def test_yuva_layouts_refuse_the_rgb_only_filters(self) -> None:
        # The same negotiation argument as for yuv420p: FFmpeg would convert
        # around an RGB-only filter, so the run is not contiguous in one format.
        for name in ("lutrgb=r=val*2", "colorlevels=rimin=.05", "curves=r='0/0 1/1'",
                     "colorbalance=rs=.2", "colorcontrast=rc=.4:rcw=.9",
                     "vibrance=intensity=.2", "colortemperature=temperature=5500",
                     "selectivecolor=reds=.2"):
            with self.subTest(filter=name):
                analysis = analyze_filtergraph(
                    f"format=yuva420p,{name},format=yuva420p"
                )
                self.assertFalse(analysis.eligible)
                self.assertEqual(
                    analysis.diagnostics[0].code, "format_not_advertised"
                )

    def test_negate_alpha_is_refused_on_an_alpha_less_layout(self) -> None:
        # Upstream fails to configure this graph, so the compiler must not
        # accept it either.
        analysis = analyze_filtergraph(
            "format=rgb24,negate=components=r+g+b+a,negate,format=rgb24"
        )
        self.assertFalse(analysis.eligible)
        self.assertEqual(analysis.diagnostics[0].code, "component_not_available")
        # The same request is fine where alpha exists.
        self.assertTrue(
            analyze_filtergraph(
                "format=rgba,negate=components=r+g+b+a,negate,format=rgba"
            ).eligible
        )


@unittest.skipUnless(ffmpeg_path(), "set LAVFI_CC_FFMPEG to run layout differentials")
class LayoutDifferentialTests(unittest.TestCase):
    WIDTH, HEIGHT = 17, 5

    def reference(self, layout: str, graph: str, source: bytes) -> bytes:
        result = subprocess.run(
            [
                str(ffmpeg_path()), "-hide_banner", "-nostdin", "-loglevel", "error",
                "-f", "rawvideo", "-pix_fmt", layout,
                "-video_size", f"{self.WIDTH}x{self.HEIGHT}",
                "-i", "pipe:0", "-vf", graph, "-pix_fmt", layout,
                "-f", "rawvideo", "pipe:1",
            ],
            input=source,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr.decode())
        return result.stdout

    def frame(self, layout: str, seed: int) -> bytes:
        return random_frame(
            get_layout(layout), self.WIDTH, self.HEIGHT, random.Random(seed)
        )

    def test_interpreter_matches_ffmpeg_for_every_layout(self) -> None:
        for seed, name in enumerate(LAYOUTS):
            source = self.frame(name, seed)
            for chain in chains_for(name):
                graph = f"format={name},{chain},format={name}"
                with self.subTest(layout=name, chain=chain):
                    analysis = analyze_filtergraph(graph)
                    self.assertTrue(analysis.eligible, analysis.diagnostics)
                    self.assertEqual(
                        interpret_rgba8(
                            analysis.ir, source, self.WIDTH, self.HEIGHT
                        ),
                        self.reference(name, graph, source),
                    )

    def test_native_kernel_matches_ffmpeg_for_every_layout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            for seed, name in enumerate(LAYOUTS):
                source = self.frame(name, 100 + seed)
                chain = chains_for(name)[-1]
                graph = f"format={name},{chain},format={name}"
                with self.subTest(layout=name):
                    analysis = analyze_filtergraph(graph)
                    self.assertTrue(analysis.eligible, analysis.diagnostics)
                    library = Path(directory) / (name + library_suffix())
                    artifact = compile_kernel(analysis.ir, library)
                    with NativeKernel(
                        artifact.library_path, analysis.ir.plan_hash, layout=name
                    ) as kernel:
                        produced = kernel.process_rgba8(
                            source, self.WIDTH, self.HEIGHT
                        )
                    self.assertEqual(
                        produced, self.reference(name, graph, source)
                    )


@unittest.skipUnless(
    week5_ffmpeg_path() and ffmpeg_path(),
    "build scripts/build-ffmpeg-week5.sh to run layout integration tests",
)
class LayoutEndToEndTests(unittest.TestCase):
    # An odd width and height so a subsampled plane's AV_CEIL_RSHIFT rounding
    # and the chroma-aligned slice boundaries both have to be right.
    WIDTH, HEIGHT = 21, 7

    def arguments(self, layout: str, graph: str) -> list[str]:
        return [
            "-hide_banner", "-nostdin", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", layout,
            "-video_size", f"{self.WIDTH}x{self.HEIGHT}",
            "-i", "pipe:0", "-filter_threads", "4", "-vf", graph,
            "-pix_fmt", layout, "-f", "rawvideo", "pipe:1",
        ]

    def test_wrapper_is_bit_exact_for_every_layout(self) -> None:
        generator = random.Random(0xB17E)
        for name in LAYOUTS:
            layout = get_layout(name)
            source = random_frame(layout, self.WIDTH, self.HEIGHT, generator)
            graph = f"format={name},{chains_for(name)[-1]},format={name}"
            arguments = self.arguments(name, graph)
            with self.subTest(layout=name):
                baseline = subprocess.run(
                    [str(ffmpeg_path()), *arguments],
                    input=source, stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE, check=False,
                )
                fused = subprocess.run(
                    [
                        sys.executable, "-m", "lavfi_cc", "run",
                        "--ffmpeg", str(week5_ffmpeg_path()),
                        "--require-fusion", "--", *arguments,
                    ],
                    cwd=ROOT, input=source, stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE, check=False,
                )
                self.assertEqual(baseline.returncode, 0, baseline.stderr.decode())
                self.assertEqual(fused.returncode, 0, fused.stderr.decode())
                self.assertEqual(
                    len(baseline.stdout),
                    layout.frame_size(self.WIDTH, self.HEIGHT),
                )
                self.assertEqual(fused.stdout, baseline.stdout)


if __name__ == "__main__":
    unittest.main()
