/*
 * Dynamic lavfi-cc 8-bit kernel filter.
 *
 * This file is part of FFmpeg.
 *
 * FFmpeg is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 */

#include <dlfcn.h>
#include <errno.h>
#include <limits.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "libavutil/avstring.h"
#include "libavutil/common.h"
#include "libavutil/error.h"
#include "libavutil/internal.h"
#include "libavutil/opt.h"
#include "libavutil/pixdesc.h"
#include "avfilter.h"
#include "filters.h"
#include "formats.h"
#include "video.h"

/* Mirrors runtime/kernel_abi.h. ABI 2 passes per-plane arrays shaped like
 * AVFrame's data[] and linesize[], so packed and planar kernels share one
 * entry point and one slice loop here. width and height describe plane 0; a
 * kernel built for a subsampled layout sizes its own chroma planes, so this
 * filter only has to point each plane at the row the slice starts on. */
#define LAVFI_KERNEL_ABI_VERSION 2u
#define LAVFI_KERNEL_MAX_PLANES 4

#define LAVFI_PIXEL_FORMAT_RGBA8  1u
#define LAVFI_PIXEL_FORMAT_BGRA8  2u
#define LAVFI_PIXEL_FORMAT_ARGB8  3u
#define LAVFI_PIXEL_FORMAT_ABGR8  4u
#define LAVFI_PIXEL_FORMAT_RGB24  5u
#define LAVFI_PIXEL_FORMAT_BGR24  6u
#define LAVFI_PIXEL_FORMAT_GBRP8  7u
#define LAVFI_PIXEL_FORMAT_GBRAP8 8u
#define LAVFI_PIXEL_FORMAT_YUV444P8  9u
#define LAVFI_PIXEL_FORMAT_YUV422P8 10u
#define LAVFI_PIXEL_FORMAT_YUV420P8 11u

static enum AVPixelFormat kernel_pixel_format(uint32_t identifier)
{
    switch (identifier) {
    case LAVFI_PIXEL_FORMAT_RGBA8:     return AV_PIX_FMT_RGBA;
    case LAVFI_PIXEL_FORMAT_BGRA8:     return AV_PIX_FMT_BGRA;
    case LAVFI_PIXEL_FORMAT_ARGB8:     return AV_PIX_FMT_ARGB;
    case LAVFI_PIXEL_FORMAT_ABGR8:     return AV_PIX_FMT_ABGR;
    case LAVFI_PIXEL_FORMAT_RGB24:     return AV_PIX_FMT_RGB24;
    case LAVFI_PIXEL_FORMAT_BGR24:     return AV_PIX_FMT_BGR24;
    case LAVFI_PIXEL_FORMAT_GBRP8:     return AV_PIX_FMT_GBRP;
    case LAVFI_PIXEL_FORMAT_GBRAP8:    return AV_PIX_FMT_GBRAP;
    case LAVFI_PIXEL_FORMAT_YUV444P8:  return AV_PIX_FMT_YUV444P;
    case LAVFI_PIXEL_FORMAT_YUV422P8:  return AV_PIX_FMT_YUV422P;
    case LAVFI_PIXEL_FORMAT_YUV420P8:  return AV_PIX_FMT_YUV420P;
    default:                           return AV_PIX_FMT_NONE;
    }
}

typedef void (*LavfiProcessFunction)(
    uint8_t *const *dst,
    const ptrdiff_t *dst_stride,
    const uint8_t *const *src,
    const ptrdiff_t *src_stride,
    int width,
    int height);

typedef struct LavfiCompiledKernel {
    uint32_t abi_version;
    uint32_t pixel_format;
    const char *plan_hash;
    LavfiProcessFunction process;
} LavfiCompiledKernel;

typedef struct FusedContext {
    const AVClass *class;
    char *kernel_path;
    char *kernel_root;
    char *plan_hash;
    int remove_color_side_data;

    void *library;
    const LavfiCompiledKernel *kernel;
    int nb_planes;
    int hsub, vsub;
} FusedContext;

typedef struct ThreadData {
    const AVFrame *in;
    AVFrame *out;
} ThreadData;

#define OFFSET(x) offsetof(FusedContext, x)
#define FLAGS (AV_OPT_FLAG_FILTERING_PARAM | AV_OPT_FLAG_VIDEO_PARAM)

static const AVOption fused_options[] = {
    { "kernel", "absolute path to a lavfi-cc kernel", OFFSET(kernel_path),
      AV_OPT_TYPE_STRING, { .str = NULL }, .flags = FLAGS },
    { "kernel_root", "trusted private directory containing the kernel",
      OFFSET(kernel_root), AV_OPT_TYPE_STRING, { .str = NULL }, .flags = FLAGS },
    { "plan_hash", "expected canonical lavfi-cc plan hash", OFFSET(plan_hash),
      AV_OPT_TYPE_STRING, { .str = NULL }, .flags = FLAGS },
    { "remove_color_side_data", "remove color-dependent frame side data",
      OFFSET(remove_color_side_data), AV_OPT_TYPE_BOOL, { .i64 = 0 }, 0, 1, FLAGS },
    { NULL }
};

AVFILTER_DEFINE_CLASS(fused);

static int valid_plan_hash(const char *hash)
{
    if (!hash || strlen(hash) != 64)
        return 0;
    for (int i = 0; i < 64; i++)
        if ((hash[i] < '0' || hash[i] > '9') &&
            (hash[i] < 'a' || hash[i] > 'f'))
            return 0;
    return 1;
}

static int trusted_kernel_path(AVFilterContext *ctx,
                               char resolved_kernel[PATH_MAX])
{
    FusedContext *s = ctx->priv;
    char resolved_root[PATH_MAX];
    struct stat kernel_stat;
    struct stat root_stat;
    const char *relative;

    if (!s->kernel_path || s->kernel_path[0] != '/' ||
        !s->kernel_root || s->kernel_root[0] != '/') {
        av_log(ctx, AV_LOG_ERROR,
               "kernel and kernel_root must both be absolute paths\n");
        return AVERROR(EINVAL);
    }
    if (!realpath(s->kernel_root, resolved_root) ||
        !realpath(s->kernel_path, resolved_kernel)) {
        av_log(ctx, AV_LOG_ERROR, "could not resolve kernel path: %s\n",
               av_err2str(AVERROR(errno)));
        return AVERROR(errno);
    }
    if (stat(resolved_root, &root_stat) < 0 ||
        stat(resolved_kernel, &kernel_stat) < 0) {
        av_log(ctx, AV_LOG_ERROR, "could not stat kernel path: %s\n",
               av_err2str(AVERROR(errno)));
        return AVERROR(errno);
    }
    if (!S_ISDIR(root_stat.st_mode) || !S_ISREG(kernel_stat.st_mode)) {
        av_log(ctx, AV_LOG_ERROR,
               "kernel_root must be a directory and kernel must be a regular file\n");
        return AVERROR(EINVAL);
    }
    if (root_stat.st_uid != geteuid() || kernel_stat.st_uid != geteuid()) {
        av_log(ctx, AV_LOG_ERROR,
               "kernel directory and file must be owned by the current user\n");
        return AVERROR(EACCES);
    }
    if ((root_stat.st_mode & (S_IWGRP | S_IWOTH)) ||
        (kernel_stat.st_mode & S_IWOTH)) {
        av_log(ctx, AV_LOG_ERROR,
               "kernel directory must be private and kernel must not be world-writable\n");
        return AVERROR(EACCES);
    }

    if (!av_strstart(resolved_kernel, resolved_root, &relative) ||
        relative[0] != '/' || relative[1] == '\0' ||
        strchr(relative + 1, '/')) {
        av_log(ctx, AV_LOG_ERROR,
               "kernel must be a direct child of the trusted kernel_root\n");
        return AVERROR(EACCES);
    }
    return 0;
}

static av_cold int init(AVFilterContext *ctx)
{
    FusedContext *s = ctx->priv;
    char resolved_kernel[PATH_MAX];
    const char *loader_error;
    int ret;

    if (!valid_plan_hash(s->plan_hash)) {
        av_log(ctx, AV_LOG_ERROR,
               "plan_hash must contain exactly 64 lowercase hexadecimal digits\n");
        return AVERROR(EINVAL);
    }
    ret = trusted_kernel_path(ctx, resolved_kernel);
    if (ret < 0)
        return ret;

    dlerror();
    s->library = dlopen(resolved_kernel, RTLD_NOW | RTLD_LOCAL);
    if (!s->library) {
        av_log(ctx, AV_LOG_ERROR, "could not load kernel: %s\n", dlerror());
        return AVERROR(EINVAL);
    }
    dlerror();
    s->kernel = dlsym(s->library, "lavfi_compiled_kernel");
    loader_error = dlerror();
    if (loader_error || !s->kernel) {
        av_log(ctx, AV_LOG_ERROR, "kernel entry point is unavailable: %s\n",
               loader_error ? loader_error : "symbol is null");
        ret = AVERROR(EINVAL);
        goto fail;
    }
    if (s->kernel->abi_version != LAVFI_KERNEL_ABI_VERSION) {
        av_log(ctx, AV_LOG_ERROR,
               "kernel ABI version %u does not match %u\n",
               s->kernel->abi_version, LAVFI_KERNEL_ABI_VERSION);
        ret = AVERROR(EINVAL);
        goto fail;
    }
    if (kernel_pixel_format(s->kernel->pixel_format) == AV_PIX_FMT_NONE) {
        av_log(ctx, AV_LOG_ERROR, "kernel pixel format %u is not supported\n",
               s->kernel->pixel_format);
        ret = AVERROR(EINVAL);
        goto fail;
    }
    if (!s->kernel->plan_hash || strcmp(s->kernel->plan_hash, s->plan_hash)) {
        av_log(ctx, AV_LOG_ERROR, "kernel plan hash does not match requested plan\n");
        ret = AVERROR(EINVAL);
        goto fail;
    }
    if (!s->kernel->process) {
        av_log(ctx, AV_LOG_ERROR, "kernel process function is null\n");
        ret = AVERROR(EINVAL);
        goto fail;
    }
    return 0;

fail:
    s->kernel = NULL;
    dlclose(s->library);
    s->library = NULL;
    return ret;
}

static av_cold void uninit(AVFilterContext *ctx)
{
    FusedContext *s = ctx->priv;

    s->kernel = NULL;
    if (s->library)
        dlclose(s->library);
    s->library = NULL;
}

/* Rows of plane 0 that job jobnr owns.
 *
 * Slices are cut on the chroma sampling grid rather than on every luma row, so
 * a subsampled row belongs to exactly one job: two jobs must never write the
 * same chroma sample. The kernel is told how many plane-0 rows it received and
 * derives each subsampled plane's row count from that with the same
 * AV_CEIL_RSHIFT the rest of libavfilter uses, which tiles exactly because
 * every boundary but the last is a multiple of 1 << vsub. */
static void slice_rows(const FusedContext *s, int height, int jobnr, int nb_jobs,
                       int *start, int *end)
{
    const int chroma_rows = AV_CEIL_RSHIFT(height, s->vsub);

    *start = FFMIN(ff_slice_pos(chroma_rows, jobnr, nb_jobs) << s->vsub, height);
    *end   = FFMIN(ff_slice_pos(chroma_rows, jobnr + 1, nb_jobs) << s->vsub, height);
}

static int filter_slice(AVFilterContext *ctx, void *arg,
                        int jobnr, int nb_jobs)
{
    FusedContext *s = ctx->priv;
    const ThreadData *td = arg;
    const uint8_t *src[LAVFI_KERNEL_MAX_PLANES] = { NULL };
    uint8_t *dst[LAVFI_KERNEL_MAX_PLANES] = { NULL };
    ptrdiff_t src_stride[LAVFI_KERNEL_MAX_PLANES] = { 0 };
    ptrdiff_t dst_stride[LAVFI_KERNEL_MAX_PLANES] = { 0 };
    int start, end;

    slice_rows(s, td->out->height, jobnr, nb_jobs, &start, &end);
    if (start >= end)
        return 0;

    for (int plane = 0; plane < s->nb_planes; plane++) {
        /* Planes 1 and 2 carry chroma; 0 and 3 are always full resolution. */
        const int vsub = (plane == 1 || plane == 2) ? s->vsub : 0;
        const int row = start >> vsub;

        src[plane] = td->in->data[plane] +
                     (ptrdiff_t)row * td->in->linesize[plane];
        dst[plane] = td->out->data[plane] +
                     (ptrdiff_t)row * td->out->linesize[plane];
        src_stride[plane] = td->in->linesize[plane];
        dst_stride[plane] = td->out->linesize[plane];
    }

    s->kernel->process(
        dst, dst_stride,
        src, src_stride,
        td->out->width,
        end - start);
    return 0;
}

static int filter_frame(AVFilterLink *inlink, AVFrame *in)
{
    AVFilterContext *ctx = inlink->dst;
    FusedContext *s = ctx->priv;
    AVFilterLink *outlink = ctx->outputs[0];
    ThreadData td;
    AVFrame *out;
    int ret;

    out = ff_get_video_buffer(outlink, outlink->w, outlink->h);
    if (!out) {
        av_frame_free(&in);
        return AVERROR(ENOMEM);
    }
    ret = av_frame_copy_props(out, in);
    if (ret < 0) {
        av_frame_free(&out);
        av_frame_free(&in);
        return ret;
    }
    if (s->remove_color_side_data)
        av_frame_side_data_remove_by_props(&out->side_data, &out->nb_side_data,
                                           AV_SIDE_DATA_PROP_COLOR_DEPENDENT);

    td.in = in;
    td.out = out;
    /* One job per chroma row at most, since that is the finest cut a slice
     * boundary can land on. */
    ff_filter_execute(ctx, filter_slice, &td, NULL,
                      FFMIN(AV_CEIL_RSHIFT(outlink->h, s->vsub),
                            ff_filter_get_nb_threads(ctx)));

    av_frame_free(&in);
    return ff_filter_frame(outlink, out);
}

/* The kernel is loaded during init, so by the time formats are negotiated the
 * filter can advertise exactly the one layout the kernel was compiled for.
 * FFmpeg then converts into it only if the surrounding graph requires it. */
static int query_formats(const AVFilterContext *ctx,
                         AVFilterFormatsConfig **cfg_in,
                         AVFilterFormatsConfig **cfg_out)
{
    const FusedContext *s = ctx->priv;
    enum AVPixelFormat pix_fmts[2] = { AV_PIX_FMT_NONE, AV_PIX_FMT_NONE };

    if (!s->kernel)
        return AVERROR(EINVAL);
    pix_fmts[0] = kernel_pixel_format(s->kernel->pixel_format);
    if (pix_fmts[0] == AV_PIX_FMT_NONE)
        return AVERROR(EINVAL);
    return ff_set_pixel_formats_from_list2(ctx, cfg_in, cfg_out, pix_fmts);
}

static int config_input(AVFilterLink *inlink)
{
    AVFilterContext *ctx = inlink->dst;
    FusedContext *s = ctx->priv;
    const AVPixFmtDescriptor *desc = av_pix_fmt_desc_get(inlink->format);

    if (!desc)
        return AVERROR(EINVAL);
    s->nb_planes = av_pix_fmt_count_planes(inlink->format);
    if (s->nb_planes < 1 || s->nb_planes > LAVFI_KERNEL_MAX_PLANES) {
        av_log(ctx, AV_LOG_ERROR, "unsupported plane count %d\n", s->nb_planes);
        return AVERROR(EINVAL);
    }
    s->hsub = desc->log2_chroma_w;
    s->vsub = desc->log2_chroma_h;
    return 0;
}

static const AVFilterPad inputs[] = {
    {
        .name         = "default",
        .type         = AVMEDIA_TYPE_VIDEO,
        .config_props = config_input,
        .filter_frame = filter_frame,
    },
};

const FFFilter ff_vf_fused = {
    .p.name        = "fused",
    .p.description = NULL_IF_CONFIG_SMALL(
        "Run one checked lavfi-cc 8-bit RGB or YUV kernel."),
    .p.priv_class  = &fused_class,
    .p.flags       = AVFILTER_FLAG_SLICE_THREADS,
    .priv_size     = sizeof(FusedContext),
    .init          = init,
    .uninit        = uninit,
    FILTER_INPUTS(inputs),
    FILTER_OUTPUTS(ff_video_default_filterpad),
    FILTER_QUERY_FUNC2(query_formats),
};
