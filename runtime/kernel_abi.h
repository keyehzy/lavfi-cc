#ifndef LAVFI_CC_KERNEL_ABI_H
#define LAVFI_CC_KERNEL_ABI_H

#include <stddef.h>
#include <stdint.h>

#define LAVFI_KERNEL_ABI_VERSION 1u

/* Packed 8-bit RGB layouts. The process signature is identical for all of
 * them -- one plane, one stride -- so adding an identifier does not change the
 * ABI. A loader that only knows RGBA8 still refuses the others by value. */
#define LAVFI_PIXEL_FORMAT_RGBA8 1u
#define LAVFI_PIXEL_FORMAT_BGRA8 2u
#define LAVFI_PIXEL_FORMAT_ARGB8 3u
#define LAVFI_PIXEL_FORMAT_ABGR8 4u
#define LAVFI_PIXEL_FORMAT_RGB24 5u
#define LAVFI_PIXEL_FORMAT_BGR24 6u

#if defined(__GNUC__) || defined(__clang__)
#define LAVFI_KERNEL_EXPORT __attribute__((visibility("default")))
#else
#define LAVFI_KERNEL_EXPORT
#endif

typedef void (*LavfiProcessFunction)(
    uint8_t *dst,
    ptrdiff_t dst_stride,
    const uint8_t *src,
    ptrdiff_t src_stride,
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
