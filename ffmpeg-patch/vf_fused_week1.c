/*
 * Hand-fused Week 1 experiment for negate + lutrgb.
 *
 * This file is part of FFmpeg.
 *
 * FFmpeg is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 */

#include "libavutil/common.h"
#include "avfilter.h"
#include "filters.h"
#include "video.h"

typedef struct FusedContext {
    uint8_t lut[3][256];
} FusedContext;

typedef struct ThreadData {
    const AVFrame *in;
    AVFrame *out;
} ThreadData;

static av_cold int init(AVFilterContext *ctx)
{
    FusedContext *s = ctx->priv;

    for (int value = 0; value < 256; value++) {
        /* Match vf_lut.c: evaluate in double, truncate to int, then clip. */
        s->lut[0][value] = av_clip((int)(value * 1.08 + 2.0), 0, 255);
        s->lut[1][value] = av_clip((int)(value * 0.94 + 4.0), 0, 255);
        s->lut[2][value] = av_clip((int)(value * 0.88 + 12.0), 0, 255);
    }

    return 0;
}

static int filter_slice(AVFilterContext *ctx, void *arg,
                        int jobnr, int nb_jobs)
{
    FusedContext *s = ctx->priv;
    const ThreadData *td = arg;
    const AVFrame *in = td->in;
    AVFrame *out = td->out;
    const int slice_start = ff_slice_pos(out->height, jobnr, nb_jobs);
    const int slice_end = ff_slice_pos(out->height, jobnr + 1, nb_jobs);

    for (int y = slice_start; y < slice_end; y++) {
        const uint8_t *src = in->data[0] + y * in->linesize[0];
        uint8_t *dst = out->data[0] + y * out->linesize[0];

        for (int x = 0; x < out->width; x++) {
            /* `negate` quantizes exactly to RGBA8 before the LUT stage. */
            const uint8_t r = 255 - src[0];
            const uint8_t g = 255 - src[1];
            const uint8_t b = 255 - src[2];

            dst[0] = s->lut[0][r];
            dst[1] = s->lut[1][g];
            dst[2] = s->lut[2][b];
            dst[3] = src[3];
            src += 4;
            dst += 4;
        }
    }

    return 0;
}

static int filter_frame(AVFilterLink *inlink, AVFrame *in)
{
    AVFilterContext *ctx = inlink->dst;
    AVFilterLink *outlink = ctx->outputs[0];
    ThreadData td;
    AVFrame *out = ff_get_video_buffer(outlink, outlink->w, outlink->h);

    if (!out) {
        av_frame_free(&in);
        return AVERROR(ENOMEM);
    }

    av_frame_copy_props(out, in);
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
        "Hand-fused Week 1 negate plus lutrgb experiment."),
    .p.flags       = AVFILTER_FLAG_SLICE_THREADS,
    .priv_size     = sizeof(FusedContext),
    .init          = init,
    FILTER_INPUTS(inputs),
    FILTER_OUTPUTS(ff_video_default_filterpad),
    FILTER_PIXFMTS(AV_PIX_FMT_RGBA),
};
