#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "kernel_abi.h"

/* Every plane's geometry is passed in rather than derived here, so this
 * harness holds no second copy of the layout table: lavfi_cc.layouts stays the
 * only place that knows how a subsampled plane is sized. Planes the layout does
 * not use stay null, so a kernel that touched one would fault under ASan. */

static unsigned char *read_file(const char *path, size_t *size)
{
    FILE *stream = fopen(path, "rb");
    unsigned char *data;
    long length;

    if (!stream || fseek(stream, 0, SEEK_END) || (length = ftell(stream)) < 0 ||
        fseek(stream, 0, SEEK_SET)) {
        fprintf(stderr, "could not inspect %s\n", path);
        exit(2);
    }
    data = malloc((size_t)length);
    if (!data || fread(data, 1, (size_t)length, stream) != (size_t)length) {
        fprintf(stderr, "could not read %s\n", path);
        exit(2);
    }
    fclose(stream);
    *size = (size_t)length;
    return data;
}

static long parse_positive(const char *text, char **end)
{
    long value = strtol(text, end, 10);

    if (*end == text || value <= 0 || value > 1 << 20) {
        fprintf(stderr, "invalid dimension %s\n", text);
        exit(2);
    }
    return value;
}

int main(int argc, char **argv)
{
    size_t source_size;
    size_t expected_size;
    size_t total = 0;
    unsigned char *source;
    unsigned char *expected;
    unsigned char *actual;
    unsigned char *destination_planes[LAVFI_KERNEL_MAX_PLANES] = { NULL };
    const unsigned char *source_planes[LAVFI_KERNEL_MAX_PLANES] = { NULL };
    ptrdiff_t strides[LAVFI_KERNEL_MAX_PLANES] = { 0 };
    long rows[LAVFI_KERNEL_MAX_PLANES] = { 0 };
    long row_bytes[LAVFI_KERNEL_MAX_PLANES] = { 0 };
    long pixel_format, width, height;
    int nb_planes;
    char *end;

    if (argc < 7 || argc > 6 + LAVFI_KERNEL_MAX_PLANES) {
        fprintf(stderr,
                "usage: %s SOURCE EXPECTED PIXEL_FORMAT WIDTH HEIGHT "
                "ROWS:ROW_BYTES...\n",
                argv[0]);
        return 2;
    }
    source = read_file(argv[1], &source_size);
    expected = read_file(argv[2], &expected_size);
    pixel_format = parse_positive(argv[3], &end);
    width = parse_positive(argv[4], &end);
    height = parse_positive(argv[5], &end);
    nb_planes = argc - 6;

    for (int plane = 0; plane < nb_planes; plane++) {
        rows[plane] = parse_positive(argv[6 + plane], &end);
        if (*end != ':')
            return 2;
        row_bytes[plane] = parse_positive(end + 1, &end);
        total += (size_t)rows[plane] * (size_t)row_bytes[plane];
    }
    if (source_size != expected_size || source_size != total) {
        fprintf(stderr, "invalid sanitizer corpus dimensions\n");
        return 2;
    }
    actual = malloc(source_size);
    if (!actual)
        return 2;
    if (lavfi_compiled_kernel.abi_version != LAVFI_KERNEL_ABI_VERSION ||
        lavfi_compiled_kernel.pixel_format != (uint32_t)pixel_format ||
        !lavfi_compiled_kernel.plan_hash || !lavfi_compiled_kernel.process)
        return 2;

    for (int plane = 0, origin = 0; plane < nb_planes; plane++) {
        source_planes[plane] = source + origin;
        destination_planes[plane] = actual + origin;
        strides[plane] = (ptrdiff_t)row_bytes[plane];
        origin += (int)(rows[plane] * row_bytes[plane]);
    }
    lavfi_compiled_kernel.process(
        destination_planes, strides,
        source_planes, strides,
        (int)width, (int)height);

    if (memcmp(actual, expected, source_size)) {
        fprintf(stderr, "sanitized kernel output differs from interpreter\n");
        return 1;
    }
    free(actual);
    free(expected);
    free(source);
    return 0;
}
