/*
 * Copyright (c) 2011-2026, CESNET
 * Copyright (c) 2011, Silicon Genome, LLC.
 *
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */
/**
 * Idea of this file comes from batchelor thesis of Patrik Radiměřský (2026),
 * although its code isn't used directly (jpeg_gen_optimal_table taken directly
 * from libjpeg-turbo; symbol couting was adapted from gpujpeg_huffman_gpu_encoder
 * because there it maps directly to the representation of GPUJPEG - restart
 * intervals, non-/interleaved. Also there is optimized cc>=2.0 variant).
 *
 * @todo
 * it could be also possible to store the input symbols for Huffman and
 * then the actuall Huffman encode would need to do less work then. Currently
 * the output is used just to count frequencies and otherwise discareded.
 */

#include "gpujpeg_common_internal.h"
#include "gpujpeg_encoder_internal.h"
#include "gpujpeg_huffman_optimal_tab_gen_gpu.h"
#include "gpujpeg_table.h"
#include "gpujpeg_util.h"
#include "utils/jchuff.h"

#define WARPS_NUM 8

/** Natural order in constant memory */
__constant__ int gpujpeg_huffman_gpu_encoder_order_natural[GPUJPEG_ORDER_NATURAL_SIZE];

struct gpujpeg_huffman_optimal_tab_gen
{
    /** Size of occupied part of output buffer */
    int* freqs; // 256 * (luma DC, luma AC, chroma DC, chroma AC)
    struct gpujpeg_timer kernel_duration;
};

// Threadblock size for CC 1.x kernel
#define THREAD_BLOCK_SIZE 48

__device__ static inline void
gpujpeg_huffman_gpu_encoder_count_code(unsigned int code, int* table)
{
    atomicAdd(&table[code], 1);
}

/**
 * Store input Huffman symbol freqs for one 8x8 block (for CC 1.x)
 *
 * destilled from gpujpeg_huffman_gpu_encoder_encode_block() for CC 1.x
 */
__device__ static void
gpujpeg_huffman_gpu_comp_freq_block_cc10(int& dc, int16_t* data, int* d_freq_dc, int* d_freq_ac)
{
    typedef uint64_t loading_t;
    const int loading_iteration_count = 64 * 2 / sizeof(loading_t);

    // Load block to shared memory
    __shared__ int16_t s_data[64 * THREAD_BLOCK_SIZE];
    for ( int i = 0; i < loading_iteration_count; i++ ) {
        ((loading_t*)s_data)[loading_iteration_count * threadIdx.x + i] = ((loading_t*)data)[i];
    }
    int data_start = 64 * threadIdx.x;

    // Encode the DC coefficient difference per section F.1.2.1
    int temp = s_data[data_start + 0] - dc;
    dc = s_data[data_start + 0];

    if ( temp < 0 ) {
        // Temp is abs value of input
        temp = -temp;
    }

    // Find the number of bits needed for the magnitude of the coefficient
    int nbits = temp ? 32 - __clz(temp) : 0;

    gpujpeg_huffman_gpu_encoder_count_code(nbits, d_freq_dc);
    // no need to  Write category offset (EmitBits rejects calls with size 0)

    // Encode the AC coefficients per section F.1.2.2 (r = run length of zeros)
    int r = 0;
    for ( int k = 1; k < 64; k++ )
    {
        temp = s_data[data_start + gpujpeg_huffman_gpu_encoder_order_natural[k]];
        if ( temp == 0 ) {
            r++;
        }
        else {
            // If run length > 15, must emit special run-length-16 codes (0xF0)
            while ( r > 15 ) {
                gpujpeg_huffman_gpu_encoder_count_code(0xF0, d_freq_ac);
                r -= 16;
            }

            if ( temp < 0 ) {
                // temp is abs value of input
                temp = -temp;
            }

            // Find the number of bits needed for the magnitude of the coefficient
            // there must be at least one 1 bit
            nbits = temp ? 32 - __clz(temp) : 0;

            // Emit Huffman symbol for run length / number of bits
            int i = (r << 4) + nbits;
            gpujpeg_huffman_gpu_encoder_count_code(i, d_freq_ac);
            // no need to Write Category offset
            r = 0;
        }
    }

    // If all the left coefs were zero, emit an end-of-block code
    if ( r > 0 ) {
        gpujpeg_huffman_gpu_encoder_count_code(0, d_freq_ac);
    }
}

/**
 * Computation of input Huffman symbols for CC 1.x (DC diffs and RRRRSSSS for AC freqencies)
 *
 * derived from gpujpeg_huffman_encoder_encode_kernel() in gpujpeg_huffman_gpu_encoder.cu
 *
 * As for 2026 this is functionally unnecessary because even it is used just for pre-Fermi cards, that date back to
 * 2011. The reason to keep is rather to have a second implementation.
 */
__global__ static void
gpujpeg_huffman_count_freqs_cc10(
    struct gpujpeg_component* d_component,
    struct gpujpeg_segment* d_segment,
    int comp_count,
    int segment_count,
    int *d_freqs
)
{
    int* d_freq_dc_luma = d_freqs;
    int* d_freq_ac_luma = d_freqs + 257;
    int* d_freq_dc_chroma = d_freqs + 2 * 257;
    int* d_freq_ac_chroma = d_freqs + 3 * 257;

    int segment_index = blockIdx.x * blockDim.x + threadIdx.x;
    if ( segment_index >= segment_count )
        return;

    struct gpujpeg_segment* segment = &d_segment[segment_index];

    // Initialize huffman coder
    int dc[GPUJPEG_MAX_COMPONENT_COUNT];
    for ( int comp = 0; comp < GPUJPEG_MAX_COMPONENT_COUNT; comp++ )
        dc[comp] = 0;

    // Non-interleaving mode
    if ( comp_count == 1 ) {
        int segment_index = segment->scan_segment_index;
        // Encode MCUs in segment
        for ( int mcu_index = 0; mcu_index < segment->mcu_count; mcu_index++ ) {
            // Get component for current scan
            struct gpujpeg_component* component = &d_component[segment->scan_index];

            // Get component data for MCU
            int16_t* block = &component->d_data_quantized[(segment_index * component->segment_mcu_count + mcu_index) * component->mcu_size];

            // Get coder parameters
            int & component_dc = dc[segment->scan_index];

            int* freq_dc = component->type == GPUJPEG_COMPONENT_LUMINANCE ? d_freq_dc_luma
                                                                          : d_freq_dc_chroma;
            int* freq_ac = component->type == GPUJPEG_COMPONENT_LUMINANCE ? d_freq_ac_luma
                                                                          : d_freq_ac_chroma;
            // Encode 8x8 block
            gpujpeg_huffman_gpu_comp_freq_block_cc10(component_dc, block, freq_dc, freq_ac);
        }
    }
    // Interleaving mode
    else {
        int segment_index = segment->scan_segment_index;
        // Encode MCUs in segment
        for ( int mcu_index = 0; mcu_index < segment->mcu_count; mcu_index++ ) {
            //assert(segment->scan_index == 0);
            for ( int comp = 0; comp < comp_count; comp++ ) {
                struct gpujpeg_component* component = &d_component[comp];

                // Prepare mcu indexes
                int mcu_index_x = (segment_index * component->segment_mcu_count + mcu_index) % component->mcu_count_x;
                int mcu_index_y = (segment_index * component->segment_mcu_count + mcu_index) / component->mcu_count_x;
                // Compute base data index
                int data_index_base = mcu_index_y * (component->mcu_size * component->mcu_count_x) + mcu_index_x * (component->mcu_size_x * GPUJPEG_BLOCK_SIZE);

                // For all vertical 8x8 blocks
                for ( int y = 0; y < component->sampling_factor.vertical; y++ ) {
                    // Compute base row data index
                    int data_index_row = data_index_base + y * (component->mcu_count_x * component->mcu_size_x * GPUJPEG_BLOCK_SIZE);
                    // For all horizontal 8x8 blocks
                    for ( int x = 0; x < component->sampling_factor.horizontal; x++ ) {
                        // Compute 8x8 block data index
                        int data_index = data_index_row + x * GPUJPEG_BLOCK_SIZE * GPUJPEG_BLOCK_SIZE;

                        // Get component data for MCU
                        int16_t* block = &component->d_data_quantized[data_index];

                        // Get coder parameters
                        int & component_dc = dc[comp];

                        int* freq_dc =
                            component->type == GPUJPEG_COMPONENT_LUMINANCE ? d_freq_dc_luma : d_freq_dc_chroma;
                        int* freq_ac =
                            component->type == GPUJPEG_COMPONENT_LUMINANCE ? d_freq_ac_luma : d_freq_ac_chroma;

                        // Encode 8x8 block
                        gpujpeg_huffman_gpu_comp_freq_block_cc10(component_dc, block, freq_dc, freq_ac);
                    }
                }
            }
        }
    }
}

#if __CUDA_ARCH__ >= 200
#ifndef FULL_MASK
#define FULL_MASK 0xffffffffu
#endif

// compat
#if CUDART_VERSION < 9000
#define __ballot_sync(set, pred) __ballot(pred)
#endif

/**
 * Store input Huffman symbol freqs for one 8x8 block (CC >= 2.0)
 *
 * destilled from gpujpeg_huffman_gpu_encoder_encode_block() for CC 2.0
 */
__device__ static void
gpujpeg_huffman_gpu_comp_freq_block_cc20(const int16_t* block, int& dc, int tid, int* freq_dc, int* freq_ac)
{
    // each thread loads a pair of values (pair after zigzag reordering)
    const int load_idx = tid * 2;
    int in_even = block[gpujpeg_huffman_gpu_encoder_order_natural[load_idx]];
    const int in_odd = block[gpujpeg_huffman_gpu_encoder_order_natural[load_idx + 1]];

    // compute preceding zero count for even coefficient (actually compute the count multiplied by 16)
    const unsigned int nonzero_mask = (1 << tid) - 1;
    const unsigned int nonzero_bitmap_0 = 1 | __ballot_sync(FULL_MASK, in_even);  // DC is always treated as nonzero
    const unsigned int nonzero_bitmap_1 = __ballot_sync(FULL_MASK, in_odd);
    const unsigned int nonzero_bitmap_pairs = nonzero_bitmap_0 | nonzero_bitmap_1;

    const int zero_pair_count = __clz(nonzero_bitmap_pairs & nonzero_mask);
    int zeros_before_even = 2 * (zero_pair_count + tid - 32);
    if((0x80000000 >> zero_pair_count) > (nonzero_bitmap_1 & nonzero_mask)) {
        zeros_before_even += 1;
    }

    // true if any nonzero pixel follows thread's odd pixel
    const bool nonzero_follows = nonzero_bitmap_pairs & ~nonzero_mask;

    // count of consecutive zeros before odd value (either one more than
    // even if even is zero or none if even value itself is nonzero)
    // (the count is actually multiplied by 16)
    int zeros_before_odd = in_even || !tid ? 0 : zeros_before_even + 1;

    // clear zero counts if no nonzero pixel follows (so that no 16-zero symbols will be emited)
    // otherwise only trim extra bits from the counts of following zeros
    const int zero_count_mask = nonzero_follows ? 0xF : 0;
    zeros_before_even &= zero_count_mask;
    zeros_before_odd &= zero_count_mask;

    // pointer to LUT for encoding thread's even value
    // (only thread #0 uses DC table, others use AC table)
    int *freq_even = freq_ac;

    // first thread handles special DC coefficient
    if(0 == tid) {
        // first thread uses DC part of the table for its even value
        freq_even = freq_dc;

        // update last DC coefficient (saved at the special place at the end of the shared bufer)
        const int original_in_even = in_even;
        in_even -= dc;
        dc = original_in_even;
    }

    int temp = in_even < 0 ? -in_even : in_even;
    // Find the number of bits needed for the magnitude of the coefficient
    int nbits = temp ? 32 - __clz(temp) : 0;

    int val_even = zeros_before_even << 4 | nbits;
    if ( 0 == tid || (in_even || zeros_before_even == 15) ) { // tid == 0 -> DC, emit always
        gpujpeg_huffman_gpu_encoder_count_code(val_even, freq_even);
    }

    // last thread handles special block-termination symbol
    if(0 == ((tid ^ 31) | in_odd)) {
        // this causes selection of huffman symbol at index 256 (which contains the termination symbol)
        gpujpeg_huffman_gpu_encoder_count_code(0, freq_ac);
    }
    else if ( in_odd || zeros_before_odd == 15 ) {
        temp = in_odd < 0 ? -in_odd : in_odd;
        // Find the number of bits needed for the magnitude of the coefficient
        nbits = nbits = temp ? 32 - __clz(temp) : 0;
        int val_odd = zeros_before_odd << 4 | nbits;
        gpujpeg_huffman_gpu_encoder_count_code(val_odd, freq_ac);
    }
}
template <bool CONTINUOUS_BLOCK_LIST>
__device__ static void
gpujpeg_huffman_count_freqs_cc20_real(
    struct gpujpeg_component* d_component,
    struct gpujpeg_segment* d_segment,
    int segment_count,
    const uint64_t* const d_block_list,
    int16_t* const d_data_quantized,
    int *s_freq_dc_luma,
    int *s_freq_ac_luma,
    int *s_freq_dc_chroma,
    int *s_freq_ac_chroma
) {
    int warpidx = threadIdx.x >> 5;
    int tid = threadIdx.x & 31;

    // Select Segment
    const int block_idx = blockIdx.x + blockIdx.y * gridDim.x;
    const int segment_index = block_idx * WARPS_NUM + warpidx;

    // // first thread initializes compact output size for next kernel
    // if(0 == tid && 0 == warpidx && 0 == block_idx) {
    //     *d_gpujpeg_huffman_output_byte_count = 0;
    // }

    // stop if out of segment bounds
    if ( segment_index >= segment_count )
        return;
    struct gpujpeg_segment* segment = &d_segment[segment_index];

    // Initialize last DC coefficients
    __shared__ int s_dc_all[GPUJPEG_MAX_COMPONENT_COUNT * WARPS_NUM];
    int* s_dc = (int*)(s_dc_all + warpidx * GPUJPEG_MAX_COMPONENT_COUNT);
    if ( tid < GPUJPEG_MAX_COMPONENT_COUNT ) {
        s_dc[tid] = 0;
    }

    // Prepare data pointers
    // unsigned int * data_compressed = (unsigned int*)(d_data_compressed + segment->data_temp_index);
    // unsigned int * data_compressed_start = data_compressed;

    // Pre-add thread ID to output pointer (it's allways used only with it)
    // data_compressed += (tid * 4);

    // Encode all block in segment
    if(CONTINUOUS_BLOCK_LIST) {
        // Get component for current scan
        const struct gpujpeg_component* component = &d_component[segment->scan_index];

        // mcu size of the component
        const int comp_mcu_size = component->mcu_size;

        // Get component data for MCU (first block)
        const int16_t* block = component->d_data_quantized + (segment->scan_segment_index * component->segment_mcu_count) * comp_mcu_size;

        // Get huffman table offset
        int* freq_dc = component->type == GPUJPEG_COMPONENT_LUMINANCE ? s_freq_dc_luma : s_freq_dc_chroma;
        int* freq_ac = component->type == GPUJPEG_COMPONENT_LUMINANCE ? s_freq_ac_luma : s_freq_ac_chroma;

        // Encode MCUs in segment
        for (int block_count = segment->mcu_count; block_count--;) {
            // Get coder parameters
            int & component_dc = s_dc[segment->scan_index];

            // Encode 8x8 block
            gpujpeg_huffman_gpu_comp_freq_block_cc20(block, component_dc, tid, freq_dc, freq_ac);

            // Advance to next block
            block += comp_mcu_size;
        }
    } else {
        // Pointer to segment's list of 8x8 blocks and their count
        const uint64_t* packed_block_info_ptr = d_block_list + segment->block_index_list_begin;

        // Encode all blocks
        for(int block_count = segment->block_count; block_count--;) {
            // Get pointer to next block input data and info about its color type
            const uint64_t packed_block_info = *(packed_block_info_ptr++);

            // Get coder parameters
            int & component_dc = s_dc[packed_block_info & 0x7f];

            // Get offset to right part of huffman table
            int* freq_dc = packed_block_info & 0x80 ? s_freq_dc_chroma : s_freq_dc_luma;
            int* freq_ac = packed_block_info & 0x80 ? s_freq_ac_chroma : s_freq_ac_luma;

            // Source data pointer
            int16_t* block = &d_data_quantized[packed_block_info >> 8];

            // Encode 8x8 block
            gpujpeg_huffman_gpu_comp_freq_block_cc20(block, component_dc, tid, freq_dc, freq_ac);
        }
    }
}
#endif // #if __CUDA_ARCH__ >= 200

/**
 * Computation of input Huffman symbols (DC diffs, RRRRSSSS for AC) freqencies (For compute capability >= 2.0)
 *
 * derived from gpujpeg_huffman_encoder_encode_kernel_warp() in gpujpeg_huffman_gpu_encoder.cu
 */
template <bool CONTINUOUS_BLOCK_LIST>
#if __CUDA_ARCH__ >= 200
__launch_bounds__(WARPS_NUM * 32, 1024 / (WARPS_NUM * 32))
#endif
__global__ static void
gpujpeg_huffman_count_freqs_cc20(
    struct gpujpeg_component* d_component,
    struct gpujpeg_segment* d_segment,
    int segment_count,
    const uint64_t* const d_block_list,
    int16_t* const d_data_quantized,
    int *d_freqs
) {
    __shared__ int s_freq_dc_luma[256];
    __shared__ int s_freq_ac_luma[256];
    __shared__ int s_freq_dc_chroma[256];
    __shared__ int s_freq_ac_chroma[256];
    static_assert(WARPS_NUM * 32 == 256, "256 threads needed to clear smem");
    s_freq_dc_luma[threadIdx.x] = 0;
    s_freq_ac_luma[threadIdx.x] = 0;
    s_freq_dc_chroma[threadIdx.x] = 0;
    s_freq_ac_chroma[threadIdx.x] = 0;
    __syncthreads();

#if __CUDA_ARCH__ >= 200
    gpujpeg_huffman_count_freqs_cc20_real<CONTINUOUS_BLOCK_LIST>(d_component, d_segment, segment_count, d_block_list,
                                                                 d_data_quantized, s_freq_dc_luma, s_freq_ac_luma,
                                                                 s_freq_dc_chroma, s_freq_ac_chroma);
#endif // #if __CUDA_ARCH__ >= 200

    __syncthreads();
    int* d_freq_dc_luma = d_freqs;
    int* d_freq_ac_luma = d_freqs + 257;
    int* d_freq_dc_chroma = d_freqs + 2 * 257;
    int* d_freq_ac_chroma = d_freqs + 3 * 257;
    atomicAdd(&d_freq_dc_luma[threadIdx.x], s_freq_dc_luma[threadIdx.x]);
    atomicAdd(&d_freq_ac_luma[threadIdx.x], s_freq_ac_luma[threadIdx.x]);
    atomicAdd(&d_freq_dc_chroma[threadIdx.x], s_freq_dc_chroma[threadIdx.x]);
    atomicAdd(&d_freq_ac_chroma[threadIdx.x], s_freq_ac_chroma[threadIdx.x]);
}

struct gpujpeg_huffman_optimal_tab_gen *
gpujpeg_huffman_optimal_tab_gpu_create() {
    auto* huffman_optimized =
        (struct gpujpeg_huffman_optimal_tab_gen*)calloc(1, sizeof(struct gpujpeg_huffman_optimal_tab_gen));
    // Copy natural order to constant device memory
    cudaMemcpyToSymbol(
        gpujpeg_huffman_gpu_encoder_order_natural,
        gpujpeg_order_natural,
        GPUJPEG_ORDER_NATURAL_SIZE * sizeof(int),
        0,
        cudaMemcpyHostToDevice
    );
    gpujpeg_cuda_check_error("Huffman encoder init (natural order copy)", return NULL);
    GPUJPEG_CUSTOM_TIMER_CREATE(huffman_optimized->kernel_duration, return NULL);
    // Configure more shared memory for all kernels
    cudaFuncSetCacheConfig(gpujpeg_huffman_count_freqs_cc20<true>, cudaFuncCachePreferShared);
    cudaFuncSetCacheConfig(gpujpeg_huffman_count_freqs_cc20<false>, cudaFuncCachePreferShared);

    cudaMalloc(&huffman_optimized->freqs, 4 * sizeof(int) * 257);


    return huffman_optimized;
}

void
gpujpeg_huffman_optimal_tab_gpu_destroy(struct gpujpeg_huffman_optimal_tab_gen* huffman_optimized) {
    if ( !huffman_optimized ) {
        return;
    }
    if ( huffman_optimized->freqs ) {
        cudaFree(huffman_optimized->freqs);
    }
    GPUJPEG_CUSTOM_TIMER_DESTROY(huffman_optimized->kernel_duration, );
    free(huffman_optimized);
}

/**
 * Get grid size for specified count of threadblocks. (Grid size is limited
 * to 65536 in both directions, so if we need more threadblocks, we must use
 * both x and y coordinates.)
 *
 * @note
 * Post-Fermi cards increased maximal value for x coordinate to 2^31-1.
 */
extern dim3
gpujpeg_huffman_gpu_encoder_grid_size(int tblock_count);

/* Documented at declaration */
int
gpujpeg_huffman_optimal_tab_gpu_generate(
    struct gpujpeg_encoder* encoder, struct gpujpeg_huffman_optimal_tab_gen* huffman_optimized)
{
    // Get coder
    struct gpujpeg_coder* coder = &encoder->coder;

    assert(coder->param.restart_interval > 0);

    // Select encoder kernel which either expects continuos segments of blocks or uses block lists
    int comp_count = 1;
    if ( coder->param.interleaved == 1 )
        comp_count = coder->param.comp_count;
    assert(comp_count >= 1 && comp_count <= GPUJPEG_MAX_COMPONENT_COUNT);

    cudaMemsetAsync(huffman_optimized->freqs, 0, 4 * 257 * sizeof(int), coder->stream);

    // GPUJPEG_CUSTOM_TIMER_START(huffman_optimized->kernel_duration, 1, coder->stream, );
    // Run kernel
    if ( encoder->coder.cuda_cc_major < 2 ) {
        dim3 thread(THREAD_BLOCK_SIZE);
        dim3 grid(gpujpeg_div_and_round_up(coder->segment_count, thread.x));
        gpujpeg_huffman_count_freqs_cc10<<<grid, thread, 0, coder->stream>>>(
            coder->d_component, coder->d_segment, comp_count, coder->segment_count, huffman_optimized->freqs);
    }
    else {
        // Run encoder kernel
        dim3 thread(32 * WARPS_NUM);
        dim3 grid =
            gpujpeg_huffman_gpu_encoder_grid_size(gpujpeg_div_and_round_up(coder->segment_count, (thread.x / 32)));
        if ( comp_count == 1 ) {
            gpujpeg_huffman_count_freqs_cc20<true><<<grid, thread, 0, coder->stream>>>(
                coder->d_component, coder->d_segment, coder->segment_count, coder->d_block_list, coder->d_data_quantized,
                huffman_optimized->freqs);
        }
        else {
            gpujpeg_huffman_count_freqs_cc20<false><<<grid, thread, 0, coder->stream>>>(
                coder->d_component, coder->d_segment, coder->segment_count, coder->d_block_list, coder->d_data_quantized,
                huffman_optimized->freqs);
        }
    }
    gpujpeg_cuda_check_error("Computing Huffman frequencies failed", return -1);
    // GPUJPEG_CUSTOM_TIMER_STOP(huffman_optimized->kernel_duration, 1, coder->stream, );
    // printf("%f ms\n", GPUJPEG_CUSTOM_TIMER_DURATION(huffman_optimized->kernel_duration));

    int freqs[4 * 257];
    cudaMemcpyAsync(freqs, huffman_optimized->freqs, 4 * 257 * sizeof(int), cudaMemcpyDefault, coder->stream);
    cudaStreamSynchronize(coder->stream);
    gpujpeg_cuda_check_error("Generating optimized Huffman tables", return -1);

    // double t0 = gpujpeg_get_time();
    jpeg_gen_optimal_table(&encoder->table_huffman[GPUJPEG_COMPONENT_LUMINANCE][GPUJPEG_HUFFMAN_DC], freqs);
    gpujpeg_table_huffman_encoder_compute(&encoder->table_huffman[GPUJPEG_COMPONENT_LUMINANCE][GPUJPEG_HUFFMAN_DC]);

    jpeg_gen_optimal_table(&encoder->table_huffman[GPUJPEG_COMPONENT_LUMINANCE][GPUJPEG_HUFFMAN_AC], freqs + 257);
    gpujpeg_table_huffman_encoder_compute(&encoder->table_huffman[GPUJPEG_COMPONENT_LUMINANCE][GPUJPEG_HUFFMAN_AC]);

    jpeg_gen_optimal_table(&encoder->table_huffman[GPUJPEG_COMPONENT_CHROMINANCE][GPUJPEG_HUFFMAN_DC], freqs + 2 * 257);
    gpujpeg_table_huffman_encoder_compute(&encoder->table_huffman[GPUJPEG_COMPONENT_CHROMINANCE][GPUJPEG_HUFFMAN_DC]);

    jpeg_gen_optimal_table(&encoder->table_huffman[GPUJPEG_COMPONENT_CHROMINANCE][GPUJPEG_HUFFMAN_AC], freqs + 3 * 257);
    gpujpeg_table_huffman_encoder_compute(&encoder->table_huffman[GPUJPEG_COMPONENT_CHROMINANCE][GPUJPEG_HUFFMAN_AC]);

    // double t1 = gpujpeg_get_time();
    // printf("%f ms\n", (t1 - t0) * 1000.0);

    // indicate success
    return 0;
}
