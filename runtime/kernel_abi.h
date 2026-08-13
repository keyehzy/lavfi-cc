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

#if defined(__GNUC__) || defined(__clang__)
#define LAVFI_KERNEL_EXPORT __attribute__((visibility("default")))
#else
#define LAVFI_KERNEL_EXPORT
#endif

/* Planes the kernel does not use may be null, and their strides are ignored.
 * width and height are in samples of the largest plane; no accepted layout is
 * subsampled, so every used plane has the same sample dimensions. */
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
