/*
 * File: cuda_acquisition_kernels.cu
 * Author: Haley Orr
 * GNSS-SDR Version: 0.0.21
 *
 * Purpose:
 * This file implements the CUDA kernels used by the Parallel Code Phase
 * Search (PCPS) acquisition block.
 *
 * The kernels use the GPU to search many Doppler frequencies and PRN code
 * positions in parallel. They apply Doppler corrections, multiply the
 * signal by the satellite's PRN code in the frequency domain, calculate
 * correlation strengths, and locate the strongest results.
 *
 * The functions at the bottom of the file configure and launch each
 * CUDA kernel.
 */

#include "cuda_acquisition_kernels.cuh"

#include <cuda_runtime.h>


// Apply a different Doppler correction to the input signal for every
// Doppler bin.
//
// Each GPU thread handles one sample from one Doppler bin. The corrected
// signals are stored separately so that their FFTs can be processed as
// one batch.
__global__ void mult_vectors_batched_kernel(
    const float2* input,
    const float2* wipeoffs,
    float2* dest,
    int fft_size,
    int fft_size_pow2,
    int num_doppler_bins)
{
    // Find the sample and Doppler bin assigned to this thread.
    int sample = blockIdx.x * blockDim.x + threadIdx.x;
    int bin = blockIdx.y * blockDim.y + threadIdx.y;

    // Some threads may fall outside the actual data.
    if (sample >= fft_size || bin >= num_doppler_bins)
        {
            return;
        }

    // The same input signal is used for every Doppler bin.
    int src_index = sample;

    // Each Doppler bin has its own set of wipeoff values.
    int wipeoff_index = bin * fft_size + sample;

    // Each corrected signal is stored in its own padded FFT row.
    int dest_index = bin * fft_size_pow2 + sample;

    float2 signal_sample = input[src_index];
    float2 wipeoff_sample = wipeoffs[wipeoff_index];

    // Multiply the two complex numbers.
    dest[dest_index] = make_float2(
        signal_sample.x * wipeoff_sample.x -
            signal_sample.y * wipeoff_sample.y,
        signal_sample.x * wipeoff_sample.y +
            signal_sample.y * wipeoff_sample.x);
}


// Multiply the FFT of each Doppler-corrected signal by the FFT of the local satellite PRN code.
// The code FFT is shared by every Doppler bin. Each thread processes one
// frequency sample from one bin.
__global__ void mult_code_batched_kernel(
    const float2* signal_fft,
    const float2* code_fft,
    float2* dest,
    int fft_size_pow2,
    int num_doppler_bins)
{
    // Find the FFT sample and Doppler bin assigned to this thread.
    int sample = blockIdx.x * blockDim.x + threadIdx.x;
    int bin = blockIdx.y * blockDim.y + threadIdx.y;

    if (sample >= fft_size_pow2 || bin >= num_doppler_bins)
        {
            return;
        }

    // Move to this sample inside this Doppler bin's FFT.
    int index = bin * fft_size_pow2 + sample;

    float2 signal_value = signal_fft[index];
    float2 code_value = code_fft[sample];

    // Multiply the signal FFT by the code FFT.
    dest[index] = make_float2(
        signal_value.x * code_value.x -
            signal_value.y * code_value.y,
        signal_value.x * code_value.y +
            signal_value.y * code_value.x);
}


// Calculate the squared magnitude of every complex correlation result.
// Squared magnitude is used because it gives the correlation strength
// without requiring the slower square-root operation.
__global__ void magnitude_squared_batched_kernel(
    const float2* src,
    float* dest,
    int fft_size,
    int fft_size_pow2,
    int num_doppler_bins)
{
    // Find the code-phase sample and Doppler bin for this thread.
    int sample = blockIdx.x * blockDim.x + threadIdx.x;
    int bin = blockIdx.y * blockDim.y + threadIdx.y;

    if (sample >= fft_size || bin >= num_doppler_bins)
        {
            return;
        }

    // The source uses the padded FFT size.
    int src_index = bin * fft_size_pow2 + sample;

    // The output only stores the valid, unpadded samples.
    int dest_index = bin * fft_size + sample;

    float2 correlation = src[src_index];

    // For a complex value x + jy, magnitude squared is x² + y².
    dest[dest_index] =
        correlation.x * correlation.x +
        correlation.y * correlation.y;
}


// Calculate the complex conjugate of every value in a vector.
// The conjugate keeps the real part and reverses the sign of the
// imaginary part. This is needed for frequency-domain correlation.
__global__ void conj_vector_kernel(
    const float2* src,
    float2* dest,
    int size)
{
    // Find the vector position assigned to this thread.
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= size)
        {
            return;
        }

    dest[index] = make_float2(
        src[index].x,
        -src[index].y);
}


// Find the largest correlation magnitude handled by each CUDA block.
// Every thread searches part of the full Doppler and code-phase grid.
// The threads then compare their results in shared memory. Each block
// writes one winning magnitude, Doppler bin, and code-phase index.
__global__ void argmax_magnitude_kernel(
    const float* magnitude,
    float* out_mag,
    int* out_bin,
    int* out_idx,
    int fft_size,
    int num_doppler_bins)
{
    // Shared memory lets threads in the same block compare their results.
    extern __shared__ unsigned char shared_mem[];

    float* local_mag = reinterpret_cast<float*>(shared_mem);
    int* local_bin =
        reinterpret_cast<int*>(&local_mag[blockDim.x]);
    int* local_idx =
        reinterpret_cast<int*>(&local_bin[blockDim.x]);

    // Local ID is the thread's position inside its block.
    int local_id = threadIdx.x;

    // Global ID is the thread's position across the entire grid.
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;

    // The search grid contains every Doppler-bin and code-phase pair.
    int total_values = fft_size * num_doppler_bins;

    float best_magnitude = -1.0f;
    int best_bin = 0;
    int best_index = 0;

    // Each thread checks several locations spaced across the search grid.
    for (int i = global_id;
         i < total_values;
         i += gridDim.x * blockDim.x)
        {
            float current_magnitude = magnitude[i];

            if (current_magnitude > best_magnitude)
                {
                    best_magnitude = current_magnitude;

                    // Convert the flat array position back into a
                    // Doppler bin and code-phase position.
                    best_bin = i / fft_size;
                    best_index = i % fft_size;
                }
        }

    // Save each thread's best result in shared memory.
    local_mag[local_id] = best_magnitude;
    local_bin[local_id] = best_bin;
    local_idx[local_id] = best_index;

    // Wait until every thread has written its result.
    __syncthreads();

    // Repeatedly cut the number of possible winners in half.
    for (int stride = blockDim.x / 2;
         stride > 0;
         stride >>= 1)
        {
            if (local_id < stride &&
                local_mag[local_id + stride] > local_mag[local_id])
                {
                    local_mag[local_id] =
                        local_mag[local_id + stride];

                    local_bin[local_id] =
                        local_bin[local_id + stride];

                    local_idx[local_id] =
                        local_idx[local_id + stride];
                }

            // Wait before beginning the next comparison step.
            __syncthreads();
        }

    // Thread zero now holds the best result found by this block.
    if (local_id == 0)
        {
            out_mag[blockIdx.x] = local_mag[0];
            out_bin[blockIdx.x] = local_bin[0];
            out_idx[blockIdx.x] = local_idx[0];
        }
}


// Set the GPU grid size and launch the Doppler-wipeoff kernel.
void launch_mult_vectors_batched(
    const float2* input,
    const float2* wipeoffs,
    float2* dest,
    int fft_size,
    int fft_size_pow2,
    int num_doppler_bins)
{
    // Each block handles 32 samples across 8 Doppler bins.
    dim3 block(32, 8);

    // Create enough blocks to cover every sample and Doppler bin.
    dim3 grid(
        (fft_size + block.x - 1) / block.x,
        (num_doppler_bins + block.y - 1) / block.y);

    mult_vectors_batched_kernel<<<grid, block>>>(
        input,
        wipeoffs,
        dest,
        fft_size,
        fft_size_pow2,
        num_doppler_bins);
}


// Set the GPU grid size and launch the PRN-code multiplication kernel.
void launch_mult_code_batched(
    const float2* signal_fft,
    const float2* code_fft,
    float2* dest,
    int fft_size_pow2,
    int num_doppler_bins)
{
    // Each block handles 32 FFT samples across 8 Doppler bins.
    dim3 block(32, 8);

    dim3 grid(
        (fft_size_pow2 + block.x - 1) / block.x,
        (num_doppler_bins + block.y - 1) / block.y);

    mult_code_batched_kernel<<<grid, block>>>(
        signal_fft,
        code_fft,
        dest,
        fft_size_pow2,
        num_doppler_bins);
}


// Set the GPU grid size and launch the magnitude calculation kernel.
void launch_magnitude_squared_batched(
    const float2* src,
    float* dest,
    int fft_size,
    int fft_size_pow2,
    int num_doppler_bins)
{
    // Each block handles 32 code phases across 8 Doppler bins.
    dim3 block(32, 8);

    dim3 grid(
        (fft_size + block.x - 1) / block.x,
        (num_doppler_bins + block.y - 1) / block.y);

    magnitude_squared_batched_kernel<<<grid, block>>>(
        src,
        dest,
        fft_size,
        fft_size_pow2,
        num_doppler_bins);
}


// Set the GPU grid size and launch the conjugate kernel.
void launch_conj_vector(
    const float2* src,
    float2* dest,
    int size)
{
    // Use 256 threads in each block.
    int block = 256;

    // Create enough blocks to cover the complete vector.
    int grid = (size + block - 1) / block;

    conj_vector_kernel<<<grid, block>>>(
        src,
        dest,
        size);
}


// Launch the search for the strongest correlation results.
void launch_argmax_magnitude(
    const float* magnitude,
    float* out_mag,
    int* out_bin,
    int* out_idx,
    int fft_size,
    int num_doppler_bins,
    int num_groups)
{
    // Each block uses 64 threads to search and compare values.
    int block = 64;

    // Shared memory stores one magnitude, bin, and index per thread.
    size_t shared_size =
        block * (sizeof(float) + 2 * sizeof(int));

    // Each group returns its own strongest result.
    argmax_magnitude_kernel<<<num_groups, block, shared_size>>>(
        magnitude,
        out_mag,
        out_bin,
        out_idx,
        fft_size,
        num_doppler_bins);
}