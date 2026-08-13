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
