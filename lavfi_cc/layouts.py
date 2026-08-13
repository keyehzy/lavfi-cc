"""Byte layouts for the 8-bit RGB pixel formats a kernel can run in.

Every layout here carries the same logical pixel -- red, green, blue, and an
alpha that may be absent -- so the pixel IR never mentions a layout.  Only the
load and store ends differ.

One addressing scheme covers both families.  A logical channel lives in plane
:attr:`PixelLayout.planes` at byte offset :attr:`PixelLayout.offsets` within
each sample group, and consecutive samples are :attr:`PixelLayout.step` bytes
apart::

    address = plane_base + y * plane_stride + x * step + offset

A packed layout has one plane, a step of three or four, and a distinct offset
per channel.  A planar layout has one plane per channel, a step of one, and
every offset zero.

The packed offsets and the planar plane order were taken from the pinned
FFmpeg build rather than from the format descriptors, by converting one known
``0x11223344`` RGBA pixel into each format and reading the bytes back.  That is
where ``gbrp``'s green, blue, red plane order comes from.

Alpha-less layouts load ``a = 0``.  That is what upstream does: the
``colorchannelmixer`` templates omit every alpha term when ``have_alpha`` is
unset, and the other accepted filters treat channels independently, so an alpha
lane that is never stored cannot affect the stored components.

No layout here is chroma-subsampled, so a kernel's ``width`` and ``height``
describe every plane it touches.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PixelLayout:
    """One packed or planar 8-bit RGB layout."""

    #: FFmpeg pixel-format name.
    name: str
    #: Number of planes the frame is stored in.
    plane_count: int
    #: Bytes between consecutive samples inside a plane.
    step: int
    #: Plane holding red, green, blue, and alpha; ``None`` when absent.
    planes: tuple[int | None, int | None, int | None, int | None]
    #: Byte offset of each channel within a sample group; ``None`` when absent.
    offsets: tuple[int | None, int | None, int | None, int | None]
    #: Identifier the compiled kernel advertises, from ``runtime/kernel_abi.h``.
    abi_id: int
    #: Macro naming ``abi_id`` in the generated C.
    abi_macro: str

    @property
    def planar(self) -> bool:
        return self.plane_count > 1

    @property
    def has_alpha(self) -> bool:
        return self.planes[3] is not None

    @property
    def stored_channels(self) -> tuple[int, ...]:
        """Logical channel indices this layout writes back."""

        return tuple(
            channel for channel in range(4) if self.planes[channel] is not None
        )

    def plane(self, channel: int) -> int | None:
        return self.planes[channel]

    def offset(self, channel: int) -> int | None:
        return self.offsets[channel]

    def row_bytes(self, width: int) -> int:
        """Bytes in one row of one plane."""

        return width * self.step

    def frame_size(self, width: int, height: int) -> int:
        """Bytes in one tightly packed frame, planes laid out back to back."""

        return self.plane_count * height * self.row_bytes(width)

    def plane_origin(self, plane: int, width: int, height: int) -> int:
        """Offset of one plane inside a tightly packed frame."""

        return plane * height * self.row_bytes(width)


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
    )
}

#: The layout every earlier revision of the compiler assumed.
DEFAULT_LAYOUT = "rgba"


def get_layout(name: str) -> PixelLayout:
    try:
        return LAYOUTS[name]
    except KeyError:
        raise KeyError(f"unsupported pixel layout {name!r}") from None
