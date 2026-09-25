#include <assert.h>
#include <cuda_runtime.h>
#include <libgpujpeg/gpujpeg_common.h>
#include <libgpujpeg/gpujpeg_encoder.h>
#include "../../src/gpujpeg_common_internal.h"
#include "../../src/gpujpeg_encoder_internal.h"
#include "../../src/gpujpeg_huffman_optimal_tab_gen_gpu.h"
#include "../../src/gpujpeg_table.h"
#include "../../src/gpujpeg_util.h"
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TEST_IMAGE_WIDTH 1920
#define TEST_IMAGE_HEIGHT 1080

#define ASSERT_STR_EQ(expected, actual) do { if (strcmp(expected, actual) != 0) { fprintf(stderr, "Assertion failed! Expected '%s', result '%s'\n", expected, actual); abort(); } } while(0)

#define ASSERT(expr)                                                                                                   \
    do {                                                                                                               \
        if ( !(expr) ) {                                                                                               \
            fprintf(stderr, "Assertion " #expr " failed!\n");                                                          \
            abort();                                                                                                   \
        }                                                                                                              \
    } while ( 0 )

static void subsampling_name_test() {
        fprintf(stderr, "testing %s: ", __func__);
        struct {
                enum gpujpeg_pixel_format fmt;
                const char *exp_subs_name;
        } test_pairs[] = {
                { GPUJPEG_U8, "4:0:0" },
                { GPUJPEG_420_U8_P0P1P2, "4:2:0" },
                { GPUJPEG_422_U8_P1020, "4:2:2" },
                { GPUJPEG_444_U8_P0P1P2, "4:4:4" },
                { GPUJPEG_4444_U8_P0123, "4:4:4:4" },
        };
        for (size_t i = 0; i < sizeof test_pairs / sizeof test_pairs[0]; ++i) {
                const char *name =
                        gpujpeg_subsampling_get_name(gpujpeg_pixel_format_get_comp_count(test_pairs[i].fmt), gpujpeg_pixel_format_get_sampling_factor(test_pairs[i].fmt));
                ASSERT_STR_EQ(test_pairs[i].exp_subs_name, name);
        }
        fprintf(stderr, "Ok\n");
}

/*
 * Test if we can encode GPU pointer as usual CPU pointer.
 */
static void encode_gpu_mem_as_cpu() {
        fprintf(stderr, "testing %s: ", __func__);
        struct gpujpeg_encoder *encoder = gpujpeg_encoder_create(0);
        if (encoder == NULL) { // do not fail here if we do not have CUDA capable device - just skip this test
                fprintf(stderr, "--\n");
                return;
        }

        struct gpujpeg_parameters param;
        gpujpeg_set_default_parameters(&param);

        struct gpujpeg_image_parameters param_image;
        gpujpeg_image_set_default_parameters(&param_image);
        param_image.width = TEST_IMAGE_WIDTH;
        param_image.height = TEST_IMAGE_HEIGHT;
        param_image.color_space = GPUJPEG_YCBCR_BT709;
        param_image.pixel_format = GPUJPEG_420_U8_P0P1P2;

        uint8_t *image = NULL;
        size_t len = param_image.width * param_image.height * 3 / 2;
        if (cudaSuccess != cudaMalloc((void**) &image, len)) {
                abort();
        }
        cudaMemset(image, 0, len);

        struct gpujpeg_encoder_input encoder_input;
        gpujpeg_encoder_input_set_image(&encoder_input, image);

        uint8_t *image_compressed = NULL;
        size_t image_compressed_size = 0;

        if (gpujpeg_encoder_encode(encoder, &param, &param_image, &encoder_input, &image_compressed, &image_compressed_size) != 0) {
                abort();
        }

        cudaFree(image);
        gpujpeg_encoder_destroy(encoder);
        fprintf(stderr, "Ok\n");
}

// defined in test_gh_95.c
void
test_gh_95();

static void
test_format_number_with_delim() {
        fprintf(stderr, "testing %s: ", __func__);

        struct {
                size_t val;
                char exp_res[1024];
        } test_cases[] = {
            {0,                          "0"             },
            {1,                          "1"             },
            {10,                         "10"            },
            {11,                         "11"            },
            {100,                        "100"           },
            {101,                        "101"           },
            {1000,                       "1,000"         },
            {1001,                       "1,001"         },
            {1100,                       "1,100"         },
            {101100,                     "101,100"       },
            {10LLU * 1000 * 1000 * 1000, "10,000,000,000"},
        };
        char buf[1024];
        for ( unsigned i = 0; i < ARR_SIZE(test_cases); i++ ) {
            char* res = format_number_with_delim(test_cases[i].val, buf, sizeof buf);
            ASSERT_STR_EQ(test_cases[i].exp_res, res);
            // "tight" buffer
            res = format_number_with_delim(test_cases[i].val, buf, strlen(test_cases[i].exp_res) + 1);
            ASSERT_STR_EQ(test_cases[i].exp_res, res);
        }

        // test not enough space
        char* res = format_number_with_delim(1000000, buf, 4);
        ASSERT_STR_EQ(res, "ERR");
        res = format_number_with_delim(1000000, buf, 2);
        ASSERT_STR_EQ(res, "E");
        res = format_number_with_delim(1000000, buf, 1);
        ASSERT_STR_EQ(res, "");

        fprintf(stderr, "Ok\n");
}

static void
test_huffman_optimal_gen()
{
    fprintf(stderr, "testing %s: ", __func__);
    if (getenv("CI")) {
            // no CUDA device in CI
            fprintf(stderr, "--\n");
            return;
    }

    struct gpujpeg_encoder* encoder = gpujpeg_encoder_create(0);
    // gpujpeg_encoder_set_option(encoder, GPUJPEG_ENC_OPT_HUFF_OPTIMAL, GPUJPEG_VAL_TRUE);
    struct gpujpeg_parameters param;
    gpujpeg_set_default_parameters(&param);
    struct gpujpeg_image_parameters param_image;
    gpujpeg_image_set_default_parameters(&param_image);
    param_image.height = param_image.width = 8;
    param_image.pixel_format = GPUJPEG_U8;
    uint8_t image[64] = "";
    struct gpujpeg_encoder_input encoder_input;
    gpujpeg_encoder_input_set_image(&encoder_input, image);
    uint8_t* image_compressed = NULL;
    size_t image_compressed_size = 0;
    if ( gpujpeg_encoder_encode(encoder, &param, &param_image, &encoder_input, &image_compressed, &image_compressed_size) !=
         0 ) {
        abort();
    }

    struct gpujpeg_huffman_optimal_tab_gen* opt = gpujpeg_huffman_optimal_tab_gpu_create();
    int16_t data_quantized[64] = {0};
    // test 1
    data_quantized[0] = 0; // DC
    data_quantized[gpujpeg_order_natural[1]] = 1; // code 0000|0001
    data_quantized[gpujpeg_order_natural[3]] = (1<<2)-1; // code 0001|0002 == 18
    data_quantized[gpujpeg_order_natural[5]] = (1<<3)-1; // code 0001|0003 == 19
    data_quantized[gpujpeg_order_natural[7]] = (1<<4)-1; // code 0001|0004 == 20
    data_quantized[gpujpeg_order_natural[10]] = (1<<5)-1; // code 0002|0005 == 37
    data_quantized[gpujpeg_order_natural[27]] = (1<<6)-1; // ZRL + 0000|0006 (no RLE)
    cudaMemcpy(encoder->coder.d_data_quantized, data_quantized,sizeof data_quantized, cudaMemcpyDefault);
    gpujpeg_huffman_optimal_tab_gpu_generate(encoder, opt);

    ASSERT(encoder->table_huffman[0][1].size[0] != 0); // EOB
    ASSERT(encoder->table_huffman[0][1].size[0 | 1] != 0);
    ASSERT(encoder->table_huffman[0][1].size[1 << 4 | 2] != 0);
    ASSERT(encoder->table_huffman[0][1].size[1 << 4 | 3] != 0);
    ASSERT(encoder->table_huffman[0][1].size[1 << 4 | 4] != 0);
    ASSERT(encoder->table_huffman[0][1].size[2 << 4 | 5] != 0);

    ASSERT(encoder->table_huffman[0][1].size[0xF0] != 0);
    ASSERT(encoder->table_huffman[0][1].size[6] != 0);

    // test 2
    memset(data_quantized, 0, sizeof data_quantized);
    data_quantized[gpujpeg_order_natural[1]] = 1; // code 0000|0001
    data_quantized[gpujpeg_order_natural[34]] = (1<<2) - 1;
    cudaMemcpy(encoder->coder.d_data_quantized, data_quantized,sizeof data_quantized, cudaMemcpyDefault);
    gpujpeg_huffman_optimal_tab_gpu_generate(encoder, opt);
    // assure that even 32 run has code (2x 16 run)
    ASSERT(encoder->table_huffman[0][1].size[0xF0] != 0);

    gpujpeg_huffman_optimal_tab_gpu_destroy(opt);

    gpujpeg_encoder_destroy(encoder);

    fprintf(stderr, "Ok\n");
}

int main() {
        subsampling_name_test();
        encode_gpu_mem_as_cpu();
        test_gh_95();
        test_format_number_with_delim();
        test_huffman_optimal_gen();
        printf("PASSED\n");
}

