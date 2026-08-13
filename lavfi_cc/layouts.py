"""Byte layouts for the 8-bit pixel formats a kernel can run in.

Every layout here carries the same logical pixel -- four channels, the last of
which may be absent -- so the pixel IR never mentions a layout.  Only the load
and store ends differ, plus the *names* the filters use for those channels: the
first three are red, green, and blue in an RGB layout and luma, Cb, and Cr in a
YUV one.  :attr:`PixelLayout.components` records that naming, because upstream
validates component options against it.

One addressing scheme covers both packed and planar families.  A logical
channel lives in plane :attr:`PixelLayout.planes` at byte offset
:attr:`PixelLayout.offsets` within each sample group, and consecutive samples
are :attr:`PixelLayout.step` bytes apart::

    address = plane_base + y * plane_stride + x * step + offset

A packed layout has one plane, a step of three or four, and a distinct offset
per channel.  A planar layout has one plane per channel, a step of one, and
every offset zero.

The packed offsets, the planar plane order, and the YUV plane order were taken
from the pinned FFmpeg build rather than from the format descriptors, by
converting one known pixel into each format and reading the bytes back.  That
is where ``gbrp``'s green, blue, red plane order comes from, and it is what
confirms YUV's luma, Cb, Cr order.

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
sample to mix.  Callers must therefore restrict subsampled layouts to
channel-independent operations, which the interpreter enforces for every
consumer of the IR.
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
    """One packed or planar 8-bit layout."""

    #: FFmpeg pixel-format name.
    name: str
    #: Number of planes the frame is stored in.
    plane_count: int
    #: Bytes between consecutive samples inside a plane.
    step: int
    #: Plane holding each logical channel; ``None`` when absent.
    planes: tuple[int | None, int | None, int | None, int | None]
    #: Byte offset of each channel within a sample group; ``None`` when absent.
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

    @property
    def planar(self) -> bool:
        return self.plane_count > 1

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

        return self.plane_width(plane, width) * self.step

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
) -> PixelLayout:
    planes: tuple[int | None, ...] = tuple(
        None if offset is None else 0 for offset in offsets
    )
    return PixelLayout(name, 1, step, planes, offsets, abi_id, abi_macro)  # type: ignore[arg-type]


def _planar(
    name: str,
    planes: tuple[int, int, int, int | None],
    abi_id: int,
    abi_macro: str,
) -> PixelLayout:
    plane_count = sum(1 for plane in planes if plane is not None)
    offsets: tuple[int | None, ...] = tuple(
        None if plane is None else 0 for plane in planes
    )
    return PixelLayout(name, plane_count, 1, planes, offsets, abi_id, abi_macro)  # type: ignore[arg-type]


def _yuv(
    name: str,
    chroma_shift: tuple[int, int],
    abi_id: int,
    abi_macro: str,
) -> PixelLayout:
    """Planar 8-bit YUV: plane 0 luma, 1 Cb, 2 Cr, no alpha."""

    return PixelLayout(
        name,
        3,
        1,
        (0, 1, 2, None),
        (0, 0, 0, None),
        abi_id,
        abi_macro,
        ("y", "u", "v", None),
        ((0, 0), chroma_shift, chroma_shift, (0, 0)),
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
    )
}

#: The layout every earlier revision of the compiler assumed.
DEFAULT_LAYOUT = "rgba"


def get_layout(name: str) -> PixelLayout:
    try:
        return LAYOUTS[name]
    except KeyError:
        raise KeyError(f"unsupported pixel layout {name!r}") from None
