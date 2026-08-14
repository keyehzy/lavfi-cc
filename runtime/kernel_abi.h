#ifndef LAVFI_CC_KERNEL_ABI_H
#define LAVFI_CC_KERNEL_ABI_H

#include <stddef.h>
#include <stdint.h>

/* Version 2 replaced the single plane and stride with per-plane arrays, so a
 * kernel can address planar formats. The arrays are shaped like AVFrame's
 * data[] and linesize[]: a packed format uses index 0 and ignores the rest. */
#define LAVFI_KERNEL_ABI_VERSION 2u
#define LAVFI_KERNEL_MAX_PLANES 4

/* Packed 8-bit RGB layouts: one plane, one stride. */
#define LAVFI_PIXEL_FORMAT_RGBA8 1u
#define LAVFI_PIXEL_FORMAT_BGRA8 2u
#define LAVFI_PIXEL_FORMAT_ARGB8 3u
#define LAVFI_PIXEL_FORMAT_ABGR8 4u
#define LAVFI_PIXEL_FORMAT_RGB24 5u
#define LAVFI_PIXEL_FORMAT_BGR24 6u

/* Planar 8-bit RGB layouts: plane 0 green, 1 blue, 2 red, 3 alpha. */
#define LAVFI_PIXEL_FORMAT_GBRP8  7u
#define LAVFI_PIXEL_FORMAT_GBRAP8 8u

/* Planar 8-bit YUV layouts: plane 0 luma, 1 Cb, 2 Cr. The chroma planes of
 * YUV422P8 and YUV420P8 are subsampled; see the note on width and height. */
#define LAVFI_PIXEL_FORMAT_YUV444P8 9u
#define LAVFI_PIXEL_FORMAT_YUV422P8 10u
#define LAVFI_PIXEL_FORMAT_YUV420P8 11u

/* The same three with an alpha plane 3, which is never subsampled: only planes
 * 1 and 2 shrink, so plane 3 is sized like plane 0. */
#define LAVFI_PIXEL_FORMAT_YUVA444P8 12u
#define LAVFI_PIXEL_FORMAT_YUVA422P8 13u
#define LAVFI_PIXEL_FORMAT_YUVA420P8 14u

/* Formats above eight bits per component. Every one of them stores each sample
 * as a little-endian uint16 whose top 16 - depth bits are zero, so a kernel
 * built for one loads a native uint16 and is only valid on a little-endian
 * host; the generated source says so with a #error. The plane geometry is the
 * same as at eight bits -- what changes is that a row is twice as many bytes
 * as it is samples, which the caller sees only through linesize. */
#define LAVFI_PIXEL_FORMAT_YUV444P9LE  15u
#define LAVFI_PIXEL_FORMAT_YUV422P9LE  16u
#define LAVFI_PIXEL_FORMAT_YUV420P9LE  17u
#define LAVFI_PIXEL_FORMAT_YUV444P10LE 18u
#define LAVFI_PIXEL_FORMAT_YUV422P10LE 19u
#define LAVFI_PIXEL_FORMAT_YUV420P10LE 20u
#define LAVFI_PIXEL_FORMAT_YUV444P12LE 21u
#define LAVFI_PIXEL_FORMAT_YUV422P12LE 22u
#define LAVFI_PIXEL_FORMAT_YUV420P12LE 23u
#define LAVFI_PIXEL_FORMAT_YUV444P14LE 24u
#define LAVFI_PIXEL_FORMAT_YUV422P14LE 25u
#define LAVFI_PIXEL_FORMAT_YUV420P14LE 26u
#define LAVFI_PIXEL_FORMAT_YUV444P16LE 27u
#define LAVFI_PIXEL_FORMAT_YUV422P16LE 28u
#define LAVFI_PIXEL_FORMAT_YUV420P16LE 29u
#define LAVFI_PIXEL_FORMAT_YUVA444P10LE 30u
#define LAVFI_PIXEL_FORMAT_YUVA422P10LE 31u
#define LAVFI_PIXEL_FORMAT_YUVA420P10LE 32u
#define LAVFI_PIXEL_FORMAT_YUVA444P16LE 33u
#define LAVFI_PIXEL_FORMAT_YUVA422P16LE 34u
#define LAVFI_PIXEL_FORMAT_YUVA420P16LE 35u

/* Planar RGB above eight bits: plane 0 green, 1 blue, 2 red, 3 alpha. */
#define LAVFI_PIXEL_FORMAT_GBRP9LE   36u
#define LAVFI_PIXEL_FORMAT_GBRP10LE  37u
#define LAVFI_PIXEL_FORMAT_GBRAP10LE 38u
#define LAVFI_PIXEL_FORMAT_GBRP12LE  39u
#define LAVFI_PIXEL_FORMAT_GBRAP12LE 40u
#define LAVFI_PIXEL_FORMAT_GBRP14LE  41u
#define LAVFI_PIXEL_FORMAT_GBRP16LE  42u
#define LAVFI_PIXEL_FORMAT_GBRAP16LE 43u

/* Packed 16-bit RGB: one plane, three or four samples per pixel. */
#define LAVFI_PIXEL_FORMAT_RGB48LE  44u
#define LAVFI_PIXEL_FORMAT_RGBA64LE 45u
#define LAVFI_PIXEL_FORMAT_BGR48LE  46u
#define LAVFI_PIXEL_FORMAT_BGRA64LE 47u

#if defined(__GNUC__) || defined(__clang__)
#define LAVFI_KERNEL_EXPORT __attribute__((visibility("default")))
#else
#define LAVFI_KERNEL_EXPORT
#endif

/* Planes the kernel does not use may be null, and their strides are ignored.
 *
 * width and height are always plane 0's sample dimensions. A kernel compiled
 * for a subsampled layout derives each chroma plane's dimensions itself with
 * AV_CEIL_RSHIFT, because it knows the layout it was generated for; the caller
 * only has to point each plane at the row corresponding to that same slice. */
typedef void (*LavfiProcessFunction)(
    uint8_t *const *dst,
    const ptrdiff_t *dst_stride,
    const uint8_t *const *src,
    const ptrdiff_t *src_stride,
    int width,
    int height);

typedef struct {
    uint32_t abi_version;
    uint32_t pixel_format;
    const char *plan_hash;
    LavfiProcessFunction process;
} LavfiCompiledKernel;

extern LAVFI_KERNEL_EXPORT const LavfiCompiledKernel lavfi_compiled_kernel;

#endif
