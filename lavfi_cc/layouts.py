"""Sample layouts for the pixel formats a kernel can run in.

Every layout here carries the same logical pixel -- four channels, the last of
which may be absent -- so the pixel IR never mentions a layout.  Only the load
and store ends differ, plus the *names* the filters use for those channels: the
first three are red, green, and blue in an RGB layout and luma, Cb, and Cr in a
YUV one.  :attr:`PixelLayout.components` records that naming, because upstream
validates component options against it.

One addressing scheme covers both packed and planar families.  A logical
channel lives in plane :attr:`PixelLayout.planes` at sample offset
:attr:`PixelLayout.offsets` within each sample group, and consecutive sample
groups are :attr:`PixelLayout.step` samples apart::

    address = plane_base + y * plane_stride
              + (x * step + offset) * sample_bytes

A packed layout has one plane, a step of three or four, and a distinct offset
per channel.  A planar layout has one plane per channel, a step of one, and
every offset zero.

The packed offsets, the planar plane order, and the YUV plane order were taken
from the pinned FFmpeg build rather than from the format descriptors, by
converting one known pixel into each format and reading the stored samples
back.  That is where ``gbrp``'s green, blue, red plane order comes from, and it
is what confirms YUV's luma, Cb, Cr order.

**Sample depth.** :attr:`PixelLayout.depth` is the number of bits a component
carries, and :attr:`PixelLayout.sample_bytes` follows from it: one byte up to
eight bits, two above.  Offsets and steps are counted in *samples* rather than
bytes so that the two widths share one addressing scheme; at eight bits the two
counts coincide, which is why every pre-existing layout's numbers are unchanged.

A high-depth layout stores each sample as a little-endian ``uint16`` whose top
``16 - depth`` bits are zero for a well-formed frame.  Only the ``le`` members
of each family are listed: the generated kernel loads a native ``uint16``, so a
big-endian host would have to byte-swap, and it is refused there instead.  What
a sample *above* :attr:`PixelLayout.max_value` means is not decided here; see
the note on the kernel's domain in :mod:`lavfi_cc.interpreter`.

Alpha-less layouts load ``a = 0``.  That is what upstream does: the
``colorchannelmixer`` templates omit every alpha term when ``have_alpha`` is
unset, and the other accepted filters treat channels independently, so an alpha
lane that is never stored cannot affect the stored components.

**Chroma subsampling.** ``yuv420p`` and ``yuv422p`` store their chroma channels
at a fraction of the frame's resolution, so a single ``width`` and ``height``
no longer describe every plane.  :attr:`PixelLayout.subsampling` records the
``log2`` shift of each channel, and the plane helpers below derive that plane's
own dimensions with FFmpeg's ``AV_CEIL_RSHIFT`` rounding.  ``width`` and
``height`` always mean plane 0, matching what the kernel ABI passes.

A subsampled layout no longer has one loop iteration per pixel: a chroma sample
covers several luma samples, so an operation that mixes channels has no single
sample to mix.  What such a layout *can* express is a mix confined to channels
that share a sample grid, which :attr:`PixelLayout.sampling_groups` partitions
out and every consumer of the IR is checked against.

The ``yuva`` layouts are what make that partition more than a formality.  Their
alpha plane is full resolution while their chroma planes may not be, so the
groups are ``(luma, alpha)`` and ``(Cb, Cr)``: two groups whose members are not
adjacent planes, in a layout that is subsampled and carries alpha at once.
Every earlier layout had alpha only where nothing was subsampled.
"""

from __future__ import annotations

from dataclasses import dataclass


def ceil_shift(value: int, shift: int) -> int:
    """Round *value* up to the next multiple of ``1 << shift``, then shift.

    This is FFmpeg's ``AV_CEIL_RSHIFT``, which is how every filter in the tree
    sizes a subsampled plane.
    """

    return (value + (1 << shift) - 1) >> shift


@dataclass(frozen=True)
class PixelLayout:
    """One packed or planar layout, at eight bits or more per component."""

    #: FFmpeg pixel-format name.
    name: str
    #: Number of planes the frame is stored in.
    plane_count: int
    #: Samples between consecutive sample groups inside a plane.
    step: int
    #: Plane holding each logical channel; ``None`` when absent.
    planes: tuple[int | None, int | None, int | None, int | None]
    #: Sample offset of each channel within a group; ``None`` when absent.
    offsets: tuple[int | None, int | None, int | None, int | None]
    #: Identifier the compiled kernel advertises, from ``runtime/kernel_abi.h``.
    abi_id: int
    #: Macro naming ``abi_id`` in the generated C.
    abi_macro: str
    #: The name each filter option uses for this channel; ``None`` when absent.
    component_names: tuple[str | None, str | None, str | None, str | None] = (
        "r",
        "g",
        "b",
        "a",
    )
    #: Horizontal and vertical ``log2`` sampling shift of each channel.
    subsampling: tuple[tuple[int, int], ...] = ((0, 0), (0, 0), (0, 0), (0, 0))
    #: Bits each component carries.  Every component of a format shares one.
    depth: int = 8

    @property
    def planar(self) -> bool:
        return self.plane_count > 1

    @property
    def sample_bytes(self) -> int:
        """Bytes one stored sample occupies."""

        return (self.depth + 7) // 8

    @property
    def high_depth(self) -> bool:
        """Whether samples are 16-bit words rather than bytes."""

        return self.sample_bytes > 1

    @property
    def max_value(self) -> int:
        """Largest value a well-formed sample carries."""

        return (1 << self.depth) - 1

    @property
    def has_alpha(self) -> bool:
        return self.planes[3] is not None

    @property
    def subsampled(self) -> bool:
        """Whether any channel is stored at less than the frame's resolution."""

        return any(shift != (0, 0) for shift in self.subsampling)

    @property
    def family(self) -> str:
        """``"yuv"`` when the colour channels are luma and chroma, else ``"rgb"``.

        Filters name their options per family, and no format offers both.
        """

        return "yuv" if self.component_names[0] == "y" else "rgb"

    @property
    def components(self) -> dict[str, int]:
        """Upstream's component names, mapped to logical channel indices.

        Upstream refuses a component that the format does not carry, so this is
        both the accepted vocabulary and the availability check.
        """

        return {
            name: channel
            for channel, name in enumerate(self.component_names)
            if name is not None and self.planes[channel] is not None
        }

    @property
    def stored_channels(self) -> tuple[int, ...]:
        """Logical channel indices this layout writes back."""

        return tuple(
            channel for channel in range(4) if self.planes[channel] is not None
        )

    @property
    def sampling_groups(self) -> tuple[tuple[int, ...], ...]:
        """Channels this layout stores, grouped by their sampling resolution.

        Channels in one group have a sample at exactly the same set of
        positions, so one loop can visit all of them and an operation may read
        across them.  Channels in different groups have no common iteration
        space: a ``yuv420p`` chroma sample covers four luma samples, so there is
        no single pixel whose luma and chroma an operation could mix.

        A layout with no subsampling has exactly one group holding every stored
        channel, which is why a whole-pixel walk is valid there.
        """

        groups: dict[tuple[int, int], list[int]] = {}
        for channel in self.stored_channels:
            groups.setdefault(self.subsampling[channel], []).append(channel)
        return tuple(tuple(group) for group in groups.values())

    @property
    def plane_channels(self) -> tuple[tuple[int, ...], ...]:
        """The logical channels each plane stores, in channel order."""

        return tuple(
            tuple(
                channel
                for channel in range(4)
                if self.planes[channel] == plane
            )
            for plane in range(self.plane_count)
        )

    def plane(self, channel: int) -> int | None:
        return self.planes[channel]

    def offset(self, channel: int) -> int | None:
        return self.offsets[channel]

    def plane_shift(self, plane: int) -> tuple[int, int]:
        """Horizontal and vertical sampling shift of one plane."""

        channels = self.plane_channels[plane]
        if not channels:
            return (0, 0)
        return self.subsampling[channels[0]]

    def plane_width(self, plane: int, width: int) -> int:
        """Samples in one row of one plane."""

        return ceil_shift(width, self.plane_shift(plane)[0])

    def plane_height(self, plane: int, height: int) -> int:
        """Rows in one plane."""

        return ceil_shift(height, self.plane_shift(plane)[1])

    def plane_row_bytes(self, plane: int, width: int) -> int:
        """Bytes in one row of one plane."""

        return self.plane_width(plane, width) * self.step * self.sample_bytes

    def row_bytes(self, width: int) -> int:
        """Bytes in one row of plane 0, which is never subsampled."""

        return self.plane_row_bytes(0, width)

    def frame_size(self, width: int, height: int) -> int:
        """Bytes in one tightly packed frame, planes laid out back to back."""

        return sum(
            self.plane_height(plane, height) * self.plane_row_bytes(plane, width)
            for plane in range(self.plane_count)
        )

    def plane_origin(self, plane: int, width: int, height: int) -> int:
        """Offset of one plane inside a tightly packed frame."""

        return sum(
            self.plane_height(earlier, height)
            * self.plane_row_bytes(earlier, width)
            for earlier in range(plane)
        )


def _packed(
    name: str,
    step: int,
    offsets: tuple[int, int, int, int | None],
    abi_id: int,
    abi_macro: str,
    depth: int = 8,
) -> PixelLayout:
    planes: tuple[int | None, ...] = tuple(
        None if offset is None else 0 for offset in offsets
    )
    return PixelLayout(
        name,
        1,
        step,
        planes,  # type: ignore[arg-type]
        offsets,
        abi_id,
        abi_macro,
        depth=depth,
    )


def _planar(
    name: str,
    planes: tuple[int, int, int, int | None],
    abi_id: int,
    abi_macro: str,
    depth: int = 8,
) -> PixelLayout:
    plane_count = sum(1 for plane in planes if plane is not None)
    offsets: tuple[int | None, ...] = tuple(
        None if plane is None else 0 for plane in planes
    )
    return PixelLayout(
        name,
        plane_count,
        1,
        planes,  # type: ignore[arg-type]
        offsets,  # type: ignore[arg-type]
        abi_id,
        abi_macro,
        depth=depth,
    )


def _yuv(
    name: str,
    chroma_shift: tuple[int, int],
    abi_id: int,
    abi_macro: str,
    *,
    alpha: bool = False,
    depth: int = 8,
) -> PixelLayout:
    """Planar YUV: plane 0 luma, 1 Cb, 2 Cr, and on ``yuva`` 3 alpha.

    Only the chroma planes are subsampled.  Alpha keeps the frame's full
    resolution, exactly as luma does, which is what upstream's ``height[0] =
    height[3] = inlink->h`` says and what the pinned binary's frame sizes
    confirm.
    """

    return PixelLayout(
        name,
        4 if alpha else 3,
        1,
        (0, 1, 2, 3 if alpha else None),
        (0, 0, 0, 0 if alpha else None),
        abi_id,
        abi_macro,
        ("y", "u", "v", "a" if alpha else None),
        ((0, 0), chroma_shift, chroma_shift, (0, 0)),
        depth,
    )


LAYOUTS: dict[str, PixelLayout] = {
    layout.name: layout
    for layout in (
        _packed("rgba", 4, (0, 1, 2, 3), 1, "LAVFI_PIXEL_FORMAT_RGBA8"),
        _packed("bgra", 4, (2, 1, 0, 3), 2, "LAVFI_PIXEL_FORMAT_BGRA8"),
        _packed("argb", 4, (1, 2, 3, 0), 3, "LAVFI_PIXEL_FORMAT_ARGB8"),
        _packed("abgr", 4, (3, 2, 1, 0), 4, "LAVFI_PIXEL_FORMAT_ABGR8"),
        _packed("rgb24", 3, (0, 1, 2, None), 5, "LAVFI_PIXEL_FORMAT_RGB24"),
        _packed("bgr24", 3, (2, 1, 0, None), 6, "LAVFI_PIXEL_FORMAT_BGR24"),
        # Plane 0 is green, 1 is blue, 2 is red, 3 is alpha.
        _planar("gbrp", (2, 0, 1, None), 7, "LAVFI_PIXEL_FORMAT_GBRP8"),
        _planar("gbrap", (2, 0, 1, 3), 8, "LAVFI_PIXEL_FORMAT_GBRAP8"),
        _yuv("yuv444p", (0, 0), 9, "LAVFI_PIXEL_FORMAT_YUV444P8"),
        _yuv("yuv422p", (1, 0), 10, "LAVFI_PIXEL_FORMAT_YUV422P8"),
        _yuv("yuv420p", (1, 1), 11, "LAVFI_PIXEL_FORMAT_YUV420P8"),
        # Plane 3 is alpha, at the frame's full resolution in all three.
        _yuv("yuva444p", (0, 0), 12, "LAVFI_PIXEL_FORMAT_YUVA444P8", alpha=True),
        _yuv("yuva422p", (1, 0), 13, "LAVFI_PIXEL_FORMAT_YUVA422P8", alpha=True),
        _yuv("yuva420p", (1, 1), 14, "LAVFI_PIXEL_FORMAT_YUVA420P8", alpha=True),
        # Planar YUV above eight bits. The depths are the ones vf_lut.c
        # advertises; 9, 12, and 14 carry no alpha-bearing twin upstream, and
        # the yuva 10-bit trio is advertised by vf_hue.c alone.
        _yuv("yuv444p9le", (0, 0), 15, "LAVFI_PIXEL_FORMAT_YUV444P9LE", depth=9),
        _yuv("yuv422p9le", (1, 0), 16, "LAVFI_PIXEL_FORMAT_YUV422P9LE", depth=9),
        _yuv("yuv420p9le", (1, 1), 17, "LAVFI_PIXEL_FORMAT_YUV420P9LE", depth=9),
        _yuv("yuv444p10le", (0, 0), 18, "LAVFI_PIXEL_FORMAT_YUV444P10LE", depth=10),
        _yuv("yuv422p10le", (1, 0), 19, "LAVFI_PIXEL_FORMAT_YUV422P10LE", depth=10),
        _yuv("yuv420p10le", (1, 1), 20, "LAVFI_PIXEL_FORMAT_YUV420P10LE", depth=10),
        _yuv("yuv444p12le", (0, 0), 21, "LAVFI_PIXEL_FORMAT_YUV444P12LE", depth=12),
        _yuv("yuv422p12le", (1, 0), 22, "LAVFI_PIXEL_FORMAT_YUV422P12LE", depth=12),
        _yuv("yuv420p12le", (1, 1), 23, "LAVFI_PIXEL_FORMAT_YUV420P12LE", depth=12),
        _yuv("yuv444p14le", (0, 0), 24, "LAVFI_PIXEL_FORMAT_YUV444P14LE", depth=14),
        _yuv("yuv422p14le", (1, 0), 25, "LAVFI_PIXEL_FORMAT_YUV422P14LE", depth=14),
        _yuv("yuv420p14le", (1, 1), 26, "LAVFI_PIXEL_FORMAT_YUV420P14LE", depth=14),
        _yuv("yuv444p16le", (0, 0), 27, "LAVFI_PIXEL_FORMAT_YUV444P16LE", depth=16),
        _yuv("yuv422p16le", (1, 0), 28, "LAVFI_PIXEL_FORMAT_YUV422P16LE", depth=16),
        _yuv("yuv420p16le", (1, 1), 29, "LAVFI_PIXEL_FORMAT_YUV420P16LE", depth=16),
        _yuv("yuva444p10le", (0, 0), 30, "LAVFI_PIXEL_FORMAT_YUVA444P10LE",
             alpha=True, depth=10),
        _yuv("yuva422p10le", (1, 0), 31, "LAVFI_PIXEL_FORMAT_YUVA422P10LE",
             alpha=True, depth=10),
        _yuv("yuva420p10le", (1, 1), 32, "LAVFI_PIXEL_FORMAT_YUVA420P10LE",
             alpha=True, depth=10),
        _yuv("yuva444p16le", (0, 0), 33, "LAVFI_PIXEL_FORMAT_YUVA444P16LE",
             alpha=True, depth=16),
        _yuv("yuva422p16le", (1, 0), 34, "LAVFI_PIXEL_FORMAT_YUVA422P16LE",
             alpha=True, depth=16),
        _yuv("yuva420p16le", (1, 1), 35, "LAVFI_PIXEL_FORMAT_YUVA420P16LE",
             alpha=True, depth=16),
        # Planar RGB above eight bits, in the same green, blue, red, alpha
        # plane order as gbrp. gbrap14le exists as a format but no accepted
        # filter advertises it, so it is not a layout here.
        _planar("gbrp9le", (2, 0, 1, None), 36, "LAVFI_PIXEL_FORMAT_GBRP9LE", 9),
        _planar("gbrp10le", (2, 0, 1, None), 37, "LAVFI_PIXEL_FORMAT_GBRP10LE", 10),
        _planar("gbrap10le", (2, 0, 1, 3), 38, "LAVFI_PIXEL_FORMAT_GBRAP10LE", 10),
        _planar("gbrp12le", (2, 0, 1, None), 39, "LAVFI_PIXEL_FORMAT_GBRP12LE", 12),
        _planar("gbrap12le", (2, 0, 1, 3), 40, "LAVFI_PIXEL_FORMAT_GBRAP12LE", 12),
        _planar("gbrp14le", (2, 0, 1, None), 41, "LAVFI_PIXEL_FORMAT_GBRP14LE", 14),
        _planar("gbrp16le", (2, 0, 1, None), 42, "LAVFI_PIXEL_FORMAT_GBRP16LE", 16),
        _planar("gbrap16le", (2, 0, 1, 3), 43, "LAVFI_PIXEL_FORMAT_GBRAP16LE", 16),
        # Packed 16-bit RGB. The step is in samples, so rgb48le is three
        # samples of two bytes rather than six of one.
        _packed("rgb48le", 3, (0, 1, 2, None), 44, "LAVFI_PIXEL_FORMAT_RGB48LE", 16),
        _packed("rgba64le", 4, (0, 1, 2, 3), 45, "LAVFI_PIXEL_FORMAT_RGBA64LE", 16),
        _packed("bgr48le", 3, (2, 1, 0, None), 46, "LAVFI_PIXEL_FORMAT_BGR48LE", 16),
        _packed("bgra64le", 4, (2, 1, 0, 3), 47, "LAVFI_PIXEL_FORMAT_BGRA64LE", 16),
    )
}

#: Layouts whose samples are 16-bit words rather than bytes.
HIGH_DEPTH_LAYOUTS = frozenset(
    name for name, layout in LAYOUTS.items() if layout.high_depth
)

#: The layout every earlier revision of the compiler assumed.
DEFAULT_LAYOUT = "rgba"


def get_layout(name: str) -> PixelLayout:
    try:
        return LAYOUTS[name]
    except KeyError:
        raise KeyError(f"unsupported pixel layout {name!r}") from None
