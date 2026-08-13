/*
 * Dynamic lavfi-cc RGBA8 kernel filter.
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
#include "avfilter.h"
#include "filters.h"
#include "video.h"

#define LAVFI_KERNEL_ABI_VERSION 1u
#define LAVFI_PIXEL_FORMAT_RGBA8 1u

typedef void (*LavfiProcessFunction)(
    uint8_t *dst,
    ptrdiff_t dst_stride,
    const uint8_t *src,
    ptrdiff_t src_stride,
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
    if (s->kernel->pixel_format != LAVFI_PIXEL_FORMAT_RGBA8) {
        av_log(ctx, AV_LOG_ERROR, "kernel pixel format %u is not RGBA8 (%u)\n",
               s->kernel->pixel_format, LAVFI_PIXEL_FORMAT_RGBA8);
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

static int filter_slice(AVFilterContext *ctx, void *arg,
                        int jobnr, int nb_jobs)
{
    FusedContext *s = ctx->priv;
    const ThreadData *td = arg;
    const int start = ff_slice_pos(td->out->height, jobnr, nb_jobs);
    const int end = ff_slice_pos(td->out->height, jobnr + 1, nb_jobs);

    s->kernel->process(
        td->out->data[0] + (ptrdiff_t)start * td->out->linesize[0],
        td->out->linesize[0],
        td->in->data[0] + (ptrdiff_t)start * td->in->linesize[0],
        td->in->linesize[0],
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
    ff_filter_execute(ctx, filter_slice, &td, NULL,
                      FFMIN(outlink->h, ff_filter_get_nb_threads(ctx)));

    av_frame_free(&in);
    return ff_filter_frame(outlink, out);
}

static const AVFilterPad inputs[] = {
    {
        .name         = "default",
        .type         = AVMEDIA_TYPE_VIDEO,
        .filter_frame = filter_frame,
    },
};

const FFFilter ff_vf_fused = {
    .p.name        = "fused",
    .p.description = NULL_IF_CONFIG_SMALL(
        "Run one checked lavfi-cc RGBA8 kernel."),
    .p.priv_class  = &fused_class,
    .p.flags       = AVFILTER_FLAG_SLICE_THREADS,
    .priv_size     = sizeof(FusedContext),
    .init          = init,
    .uninit        = uninit,
    FILTER_INPUTS(inputs),
    FILTER_OUTPUTS(ff_video_default_filterpad),
    FILTER_PIXFMTS(AV_PIX_FMT_RGBA),
};
