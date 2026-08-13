#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "kernel_abi.h"

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

int main(int argc, char **argv)
{
    const int width = 257;
    size_t source_size;
    size_t expected_size;
    unsigned char *source;
    unsigned char *expected;
    unsigned char *actual;
    int height;

    if (argc != 3) {
        fprintf(stderr, "usage: %s SOURCE EXPECTED\n", argv[0]);
        return 2;
    }
    source = read_file(argv[1], &source_size);
    expected = read_file(argv[2], &expected_size);
    if (source_size != expected_size || source_size % ((size_t)width * 4)) {
        fprintf(stderr, "invalid sanitizer corpus dimensions\n");
        return 2;
    }
    height = (int)(source_size / ((size_t)width * 4));
    actual = malloc(source_size);
    if (!actual)
        return 2;
    if (lavfi_compiled_kernel.abi_version != LAVFI_KERNEL_ABI_VERSION ||
        lavfi_compiled_kernel.pixel_format != LAVFI_PIXEL_FORMAT_RGBA8 ||
        !lavfi_compiled_kernel.plan_hash || !lavfi_compiled_kernel.process)
        return 2;
    {
        /* RGBA8 is packed, so only plane 0 is used; the rest stay null so a
         * kernel that touched them would fault under the sanitizers. */
        unsigned char *destination_planes[LAVFI_KERNEL_MAX_PLANES] = { actual };
        const unsigned char *source_planes[LAVFI_KERNEL_MAX_PLANES] = { source };
        ptrdiff_t strides[LAVFI_KERNEL_MAX_PLANES] = { (ptrdiff_t)width * 4 };

        lavfi_compiled_kernel.process(
            destination_planes, strides,
            source_planes, strides,
            width, height);
    }
    if (memcmp(actual, expected, source_size)) {
        fprintf(stderr, "sanitized kernel output differs from interpreter\n");
        return 1;
    }
    free(actual);
    free(expected);
    free(source);
    return 0;
}
