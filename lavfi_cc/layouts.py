"""Byte layouts for the 8-bit RGB pixel formats a kernel can run in.

Every layout here carries the same logical pixel -- red, green, blue, and an
alpha that may be absent -- so the pixel IR never mentions a layout.  Only the
load and store ends differ: which byte of the packed pixel holds which
component, and whether there is an alpha byte at all.

The offsets were taken from the pinned FFmpeg build rather than from the format
descriptors, by converting one known ``0x11223344`` RGBA pixel into each format
and reading the bytes back.

Alpha-less layouts load ``a = 0``.  That is what upstream does: the packed
``colorchannelmixer`` path omits every alpha term when ``have_alpha`` is unset,
and the other accepted filters treat channels independently, so an alpha lane
that is never stored cannot affect the stored components.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PixelLayout:
    """One packed 8-bit RGB layout."""

    #: FFmpeg pixel-format name.
    name: str
    #: Bytes between consecutive pixels.
    step: int
    #: Byte offset of red, green, blue, and alpha; ``None`` when absent.
    offsets: tuple[int, int, int, int | None]
    #: Identifier the compiled kernel advertises, from ``runtime/kernel_abi.h``.
    abi_id: int = 1
    #: Macro naming ``abi_id`` in the generated C.
    abi_macro: str = "LAVFI_PIXEL_FORMAT_RGBA8"

    @property
    def has_alpha(self) -> bool:
        return self.offsets[3] is not None

    @property
    def stored_channels(self) -> tuple[int, ...]:
        """Logical channel indices this layout writes back."""

        return tuple(
            channel
            for channel in range(4)
            if self.offsets[channel] is not None
        )

    def offset(self, channel: int) -> int | None:
        return self.offsets[channel]


LAYOUTS: dict[str, PixelLayout] = {
    layout.name: layout
    for layout in (
        PixelLayout("rgba", 4, (0, 1, 2, 3), 1, "LAVFI_PIXEL_FORMAT_RGBA8"),
        PixelLayout("bgra", 4, (2, 1, 0, 3), 2, "LAVFI_PIXEL_FORMAT_BGRA8"),
        PixelLayout("argb", 4, (1, 2, 3, 0), 3, "LAVFI_PIXEL_FORMAT_ARGB8"),
        PixelLayout("abgr", 4, (3, 2, 1, 0), 4, "LAVFI_PIXEL_FORMAT_ABGR8"),
        PixelLayout("rgb24", 3, (0, 1, 2, None), 5, "LAVFI_PIXEL_FORMAT_RGB24"),
        PixelLayout("bgr24", 3, (2, 1, 0, None), 6, "LAVFI_PIXEL_FORMAT_BGR24"),
    )
}

#: The layout every earlier revision of the compiler assumed.
DEFAULT_LAYOUT = "rgba"


def get_layout(name: str) -> PixelLayout:
    try:
        return LAYOUTS[name]
    except KeyError:
        raise KeyError(f"unsupported pixel layout {name!r}") from None
