/*
 * File: pcps_cuda_acquisition_kernels.h
 * Author: Haley Orr
 * GNSS-SDR Version: 0.0.21
 *
 * Purpose:
 * This file declares the functions used to start the CUDA kernels for
 * PCPS acquisition.
 *
 * These functions move the main acquisition calculations to the GPU.
 * They apply Doppler corrections, perform frequency-domain correlation
 * with the satellite's PRN code, calculate correlation magnitudes, and
 * find the strongest acquisition result.
 *
 * The CUDA kernel implementations are located in the matching .cu file.
 */

#pragma once

#include <cuda_runtime.h>

// Apply every Doppler correction to the input samples.
// This creates a separate corrected copy of the signal for each Doppler
// bin so that the GPU can search many possible frequencies in parallel.
void launch_mult_vectors_batched(
    const float2* input,
    const float2* wipeoffs,
    float2* dest,
    int fft_size,
    int fft_size_pow2,
    int num_doppler_bins);

// Multiply the signal FFT by the local PRN-code FFT.
// This performs the frequency-domain part of the correlation for every
// Doppler bin.
void launch_mult_code_batched(
    const float2* signal_fft,
    const float2* code_fft,
    float2* dest,
    int fft_size_pow2,
    int num_doppler_bins);

// Calculate the squared magnitude of each complex correlation result.
// The squared magnitude shows the strength of the match at each Doppler
// bin and PRN code position.
void launch_magnitude_squared_batched(
    const float2* src,
    float* dest,
    int fft_size,
    int fft_size_pow2,
    int num_doppler_bins);

// Calculate the complex conjugate of every value in a vector.
// The conjugated PRN-code FFT is needed to perform correlation using
// frequency-domain multiplication.
void launch_conj_vector(
    const float2* src,
    float2* dest,
    int size);

// Find the strongest correlation result in the complete search grid.
// This returns the correlation magnitude, Doppler bin, and PRN code
// position of the strongest detected match.
void launch_argmax_magnitude(
    const float* magnitude,
    float* out_mag,
    int* out_bin,
    int* out_idx,
    int fft_size,
    int num_doppler_bins,
    int num_groups);