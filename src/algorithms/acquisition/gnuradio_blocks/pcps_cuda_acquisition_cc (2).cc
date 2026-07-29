/*
 * File: pcps_cuda_acquisition_cc.cc
 * Author: Haley Orr
 * GNSS-SDR Version: 0.0.21
 *
 * Purpose:
 * This file implements a CUDA-based Parallel Code Phase Search (PCPS)
 * acquisition block for GNSS-SDR.
 *
 * Acquisition searches the incoming signal for a satellite by testing
 * different Doppler frequencies and PRN code positions. The CUDA path
 * processes the Doppler bins in parallel on the GPU. If CUDA cannot be
 * initialized, the block uses the original CPU-based VOLK path instead.
 *
 * This file also records CUDA acquisition timing information so that
 * GPU and CPU performance can be compared.
 */

#include "pcps_cuda_acquisition_cc.h"
#include "cuda_acquisition_kernels.cuh"
#include "MATH_CONSTANTS.h"  // Provides TWO_PI.

#include <gnuradio/io_signature.h>
#include <volk/volk.h>
#include <volk_gnsssdr/volk_gnsssdr.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cuda_runtime.h>
#include <cufft.h>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <utility>

#if USE_GLOG_AND_GFLAGS
#include <glog/logging.h>
#else
#include <absl/log/log.h>
#endif


namespace
{
// Keep track of the total CUDA acquisition time across all attempts.
int cuda_acq_count = 0;
double cuda_total_time = 0.0;

// This prevents the final benchmark summary from being printed more than once.
bool cuda_printed_results = false;


// Print a readable message when a CUDA operation fails.
inline void cuda_check(cudaError_t err, const char* what)
{
    if (err != cudaSuccess)
        {
            std::cerr << "CUDA error in " << what << ": "
                      << cudaGetErrorString(err) << '\n';
        }
}


// Print a readable message when a cuFFT operation fails.
inline void cufft_check(cufftResult err, const char* what)
{
    if (err != CUFFT_SUCCESS)
        {
            std::cerr << "cuFFT error in " << what
                      << ": " << err << '\n';
        }
}
}  // namespace


// Create a shared pointer to a new CUDA acquisition block.
pcps_cuda_acquisition_cc_sptr pcps_make_cuda_acquisition_cc(
    const Acq_Conf& conf,
    uint32_t max_dwells)
{
    return pcps_cuda_acquisition_cc_sptr(
        new pcps_cuda_acquisition_cc(conf, max_dwells));
}


// Set up the acquisition block and prepare both the CUDA and CPU paths.
pcps_cuda_acquisition_cc::pcps_cuda_acquisition_cc(
    const Acq_Conf& conf,
    uint32_t max_dwells)
    : acquisition_impl_interface(
          "pcps_cuda_acquisition_cc",
          gr::io_signature::make(
              1,
              1,
              static_cast<int>(
                  sizeof(gr_complex) *
                  conf.sampled_ms *
                  conf.samples_per_ms)),
          gr::io_signature::make(
              0,
              1,
              sizeof(Gnss_Synchro))),
      d_cu_buffer_in(nullptr),
      d_cu_buffer_fft_codes(nullptr),
      d_cu_buffer_1(nullptr),
      d_cu_buffer_2(nullptr),
      d_cu_buffer_magnitude(nullptr),
      d_cu_buffer_argmax_mag(nullptr),
      d_cu_buffer_argmax_bin(nullptr),
      d_cu_buffer_argmax_idx(nullptr),
      d_cu_buffer_grid_doppler_wipeoffs(nullptr),
      d_cu_fft_plan(0),
      d_cu_fft_batch_size(1),
      d_acq_params(conf),
      d_gnss_synchro(nullptr),
      d_sample_counter(0ULL),
      d_max_doppler_indexs(nullptr),
      d_mag(0.0),
      d_input_power(0.0),
      d_test_statistics(0.0),
      d_state(0),
      d_cuda(1),
      d_max_dwells(max_dwells),
      d_well_count(0),
      d_fft_size(
          conf.sampled_ms *
          conf.samples_per_ms),
      d_fft_size_pow2(
          static_cast<uint32_t>(
              std::pow(
                  2,
                  std::ceil(
                      std::log2(2 * d_fft_size))))),
      d_num_doppler_bins(0),
      d_code_phase(0),
      d_channel(0),
      d_in_dwell_count(0),
      d_active(false),
      d_core_working(false),
      d_in_buffer(
          d_max_dwells,
          std::vector<gr_complex>(d_fft_size)),
      d_magnitude(d_fft_size),
      d_fft_codes(d_fft_size_pow2),
      d_zero_vector(
          d_fft_size_pow2 - d_fft_size,
          gr_complex(0.0, 0.0))
{
    // This port sends acquisition success or failure messages.
    this->message_port_register_out(pmt::mp("events"));

    // Count how many Doppler frequencies will be searched.
    for (int doppler = d_acq_params.doppler_min;
         doppler <= d_acq_params.doppler_max;
         doppler += d_acq_params.doppler_step)
        {
            d_num_doppler_bins++;
        }

    // One FFT is performed for each Doppler bin.
    d_cu_fft_batch_size =
        static_cast<int>(d_num_doppler_bins);

    // Store a magnitude for every Doppler and code-phase combination.
    d_magnitude.resize(
        d_fft_size * d_num_doppler_bins);

    // Try to initialize CUDA.
    d_cuda = init_cuda_environment();

    // If CUDA initialization failed, prepare the CPU FFT objects.
    if (d_cuda != 0)
        {
            d_fft_if =
                gnss_fft_fwd_make_unique(d_fft_size);

            d_ifft =
                gnss_fft_rev_make_unique(d_fft_size);
        }

    // Create one Doppler wipeoff vector for every frequency bin.
    d_grid_doppler_wipeoffs =
        std::vector<std::vector<gr_complex>>(
            d_num_doppler_bins,
            std::vector<gr_complex>(d_fft_size));

    for (uint32_t doppler_index = 0;
         doppler_index < d_num_doppler_bins;
         doppler_index++)
        {
            // Convert this bin number into its actual Doppler frequency.
            const int doppler =
                d_acq_params.doppler_min +
                static_cast<int>(
                    d_acq_params.doppler_step *
                    doppler_index);

            // Calculate how much phase changes from one sample to the next.
            const float phase_step_rad =
                static_cast<float>(TWO_PI) *
                doppler /
                static_cast<float>(d_acq_params.fs_in);

            std::array<float, 1> phase{};

            // Generate the complex sinusoid that removes this Doppler shift.
            volk_gnsssdr_s32f_sincos_32fc(
                d_grid_doppler_wipeoffs[doppler_index].data(),
                -phase_step_rad,
                phase.data(),
                d_fft_size);

            // Copy the precomputed wipeoff vector to the GPU.
            if (d_cuda == 0)
                {
                    cuda_check(
                        cudaMemcpy(
                            d_cu_buffer_grid_doppler_wipeoffs +
                                (doppler_index * d_fft_size),
                            reinterpret_cast<const float2*>(
                                d_grid_doppler_wipeoffs[
                                    doppler_index]
                                    .data()),
                            sizeof(float2) * d_fft_size,
                            cudaMemcpyHostToDevice),
                        "copy Doppler wipeoff grid");
                }
        }

    // Clear the padded GPU input buffer before it is used.
    if (d_cuda == 0)
        {
            cuda_check(
                cudaMemset(
                    d_cu_buffer_1,
                    0,
                    sizeof(float2) *
                        d_fft_size_pow2 *
                        d_num_doppler_bins),
                "zero padded CUDA FFT input buffer");
        }
}


// Release GPU resources and print the final CUDA benchmark results.
pcps_cuda_acquisition_cc::~pcps_cuda_acquisition_cc()
{
    // Print the combined benchmark once when the block is destroyed.
    if (!cuda_printed_results && cuda_acq_count > 0)
        {
            cuda_printed_results = true;

            std::cout << std::fixed
                      << std::setprecision(6);

            std::cout
                << "ACQ_BENCHMARK,"
                << "implementation=CUDA,"
                << "attempts=" << cuda_acq_count << ","
                << "total_seconds=" << cuda_total_time << ","
                << "average_seconds="
                << cuda_total_time / cuda_acq_count
                << std::endl;
        }

    // Only free CUDA resources if initialization succeeded.
    if (d_cuda == 0)
        {
            cufftDestroy(d_cu_fft_plan);

            cudaFree(d_cu_buffer_in);
            cudaFree(d_cu_buffer_fft_codes);
            cudaFree(d_cu_buffer_1);
            cudaFree(d_cu_buffer_2);
            cudaFree(d_cu_buffer_magnitude);
            cudaFree(d_cu_buffer_argmax_mag);
            cudaFree(d_cu_buffer_argmax_bin);
            cudaFree(d_cu_buffer_argmax_idx);
            cudaFree(
                d_cu_buffer_grid_doppler_wipeoffs);
        }

    // Close the optional acquisition dump file.
    try
        {
            if (d_acq_params.dump)
                {
                    d_dump_file.close();
                }
        }
    catch (const std::ofstream::failure& e)
        {
            std::cerr
                << "Problem closing Acquisition dump file: "
                << d_acq_params.dump_filename
                << '\n';
        }
    catch (const std::exception& e)
        {
            std::cerr << e.what() << '\n';
        }
}


// Select a CUDA device, allocate GPU memory, and create the FFT plan.
//
// A return value of zero means CUDA is ready. A nonzero value identifies
// the step where initialization failed.
int pcps_cuda_acquisition_cc::init_cuda_environment()
{
    // Use the first available CUDA device.
    cudaError_t cuda_err = cudaSetDevice(0);

    if (cuda_err != cudaSuccess)
        {
            std::cout
                << "Error selecting CUDA device: "
                << cudaGetErrorString(cuda_err)
                << '\n';

            return 1;
        }

    d_cu_fft_batch_size =
        static_cast<int>(d_num_doppler_bins);

    // Allocate space for one dwell of input samples.
    cuda_err = cudaMalloc(
        reinterpret_cast<void**>(&d_cu_buffer_in),
        sizeof(float2) * d_fft_size);

    if (cuda_err != cudaSuccess)
        {
            return 2;
        }

    // Allocate space for every precomputed Doppler wipeoff vector.
    cuda_err = cudaMalloc(
        reinterpret_cast<void**>(
            &d_cu_buffer_grid_doppler_wipeoffs),
        sizeof(float2) *
            d_fft_size *
            d_num_doppler_bins);

    if (cuda_err != cudaSuccess)
        {
            return 3;
        }

    // Allocate space for the FFT of the local PRN code.
    cuda_err = cudaMalloc(
        reinterpret_cast<void**>(
            &d_cu_buffer_fft_codes),
        sizeof(float2) * d_fft_size_pow2);

    if (cuda_err != cudaSuccess)
        {
            return 4;
        }

    // Allocate the first temporary batched FFT buffer.
    cuda_err = cudaMalloc(
        reinterpret_cast<void**>(&d_cu_buffer_1),
        sizeof(float2) *
            d_fft_size_pow2 *
            d_num_doppler_bins);

    if (cuda_err != cudaSuccess)
        {
            return 5;
        }

    // Allocate the second temporary batched FFT buffer.
    cuda_err = cudaMalloc(
        reinterpret_cast<void**>(&d_cu_buffer_2),
        sizeof(float2) *
            d_fft_size_pow2 *
            d_num_doppler_bins);

    if (cuda_err != cudaSuccess)
        {
            return 6;
        }

    // Allocate one magnitude for each location in the search grid.
    cuda_err = cudaMalloc(
        reinterpret_cast<void**>(
            &d_cu_buffer_magnitude),
        sizeof(float) *
            d_fft_size *
            d_num_doppler_bins);

    if (cuda_err != cudaSuccess)
        {
            return 7;
        }

    // The argmax kernel returns one partial winner from each group.
    const int local_size = 64;

    const int total =
        static_cast<int>(
            d_fft_size *
            d_num_doppler_bins);

    const int num_groups =
        (total + local_size - 1) /
        local_size;

    // Allocate space for each group's strongest magnitude.
    cuda_err = cudaMalloc(
        reinterpret_cast<void**>(
            &d_cu_buffer_argmax_mag),
        sizeof(float) * num_groups);

    if (cuda_err != cudaSuccess)
        {
            return 8;
        }

    // Allocate space for each group's strongest Doppler bin.
    cuda_err = cudaMalloc(
        reinterpret_cast<void**>(
            &d_cu_buffer_argmax_bin),
        sizeof(int) * num_groups);

    if (cuda_err != cudaSuccess)
        {
            return 9;
        }

    // Allocate space for each group's strongest code-phase index.
    cuda_err = cudaMalloc(
        reinterpret_cast<void**>(
            &d_cu_buffer_argmax_idx),
        sizeof(int) * num_groups);

    if (cuda_err != cudaSuccess)
        {
            return 10;
        }

    int n[1] = {
        static_cast<int>(d_fft_size_pow2)};

    // Create one FFT plan that processes every Doppler bin as a batch.
    cufftResult cufft_err = cufftPlanMany(
        &d_cu_fft_plan,
        1,
        n,
        nullptr,
        1,
        static_cast<int>(d_fft_size_pow2),
        nullptr,
        1,
        static_cast<int>(d_fft_size_pow2),
        CUFFT_C2C,
        static_cast<int>(d_num_doppler_bins));

    if (cufft_err != CUFFT_SUCCESS)
        {
            std::cout
                << "Error creating CUDA FFT plan.\n";

            return 11;
        }

    return 0;
}


// Prepare the local PRN code that will be correlated with the signal.
void pcps_cuda_acquisition_cc::set_local_code(
    std::complex<float>* code)
{
    if (d_cuda == 0)
        {
            // Pad the code to the FFT size used by the CUDA search.
            std::vector<gr_complex> code_padded(
                d_fft_size_pow2,
                gr_complex(0.0, 0.0));

            std::copy(
                code,
                code + d_fft_size,
                code_padded.begin());

            // Place another copy at the end to preserve the required
            // code relationship inside the larger padded FFT.
            if (d_fft_size_pow2 >= 2 * d_fft_size)
                {
                    std::copy(
                        code,
                        code + d_fft_size,
                        code_padded.begin() +
                            (d_fft_size_pow2 -
                             d_fft_size));
                }

            // Copy the padded code to the GPU.
            cuda_check(
                cudaMemcpy(
                    d_cu_buffer_2,
                    reinterpret_cast<const float2*>(
                        code_padded.data()),
                    sizeof(float2) *
                        d_fft_size_pow2,
                    cudaMemcpyHostToDevice),
                "copy local code to CUDA");

            // Transform the code into the frequency domain.
            cufft_check(
                cufftExecC2C(
                    d_cu_fft_plan,
                    reinterpret_cast<cufftComplex*>(
                        d_cu_buffer_2),
                    reinterpret_cast<cufftComplex*>(
                        d_cu_buffer_2),
                    CUFFT_FORWARD),
                "FFT local code");

            // Conjugating the code FFT prepares it for correlation.
            launch_conj_vector(
                d_cu_buffer_2,
                d_cu_buffer_fft_codes,
                static_cast<int>(
                    d_fft_size_pow2));

            cuda_check(
                cudaDeviceSynchronize(),
                "synchronize local code setup");
        }
    else
        {
            // Prepare the same code using the CPU FFT path.
            std::copy(
                code,
                code + d_fft_size,
                d_fft_if->get_inbuf());

            d_fft_if->execute();

            volk_32fc_conjugate_32fc(
                d_fft_codes.data(),
                d_fft_if->get_outbuf(),
                d_fft_size);
        }
}


// Run one PCPS acquisition dwell using the CPU and VOLK.
void pcps_cuda_acquisition_cc::acquisition_core_volk()
{
    int doppler;
    uint32_t indext = 0;
    float magt = 0.0;

    float fft_normalization_factor =
        static_cast<float>(d_fft_size) *
        static_cast<float>(d_fft_size);

    uint64_t samplestamp =
        d_sample_counter_buffer[d_well_count];

    d_input_power = 0.0;
    d_mag = 0.0;

    d_well_count++;

    DLOG(INFO)
        << "Channel: " << d_channel
        << " , doing acquisition of satellite: "
        << d_gnss_synchro->System
        << " " << d_gnss_synchro->PRN
        << " ,sample stamp: " << d_sample_counter
        << ", threshold: "
        << d_acq_params.threshold
        << ", doppler_max: "
        << d_acq_params.doppler_max
        << ", doppler_step: "
        << d_acq_params.doppler_step;

    // Estimate the average power of the input signal.
    volk_32fc_magnitude_squared_32f(
        d_magnitude.data(),
        d_in_buffer[d_well_count].data(),
        d_fft_size);

    volk_32f_accumulator_s32f(
        &d_input_power,
        d_magnitude.data(),
        d_fft_size);

    d_input_power /=
        static_cast<float>(d_fft_size);

    // Test each Doppler frequency one at a time on the CPU.
    for (uint32_t doppler_index = 0;
         doppler_index < d_num_doppler_bins;
         doppler_index++)
        {
            doppler =
                d_acq_params.doppler_min +
                d_acq_params.doppler_step *
                    doppler_index;

            // Remove the Doppler shift being tested.
            volk_32fc_x2_multiply_32fc(
                d_fft_if->get_inbuf(),
                d_in_buffer[d_well_count].data(),
                d_grid_doppler_wipeoffs[
                    doppler_index]
                    .data(),
                d_fft_size);

            // Move the corrected signal into the frequency domain.
            d_fft_if->execute();

            // Multiply by the prepared PRN code FFT.
            volk_32fc_x2_multiply_32fc(
                d_ifft->get_inbuf(),
                d_fft_if->get_outbuf(),
                d_fft_codes.data(),
                d_fft_size);

            // Return to the time domain to obtain code-phase matches.
            d_ifft->execute();

            // Calculate the correlation strength at each code phase.
            volk_32fc_magnitude_squared_32f(
                d_magnitude.data(),
                d_ifft->get_outbuf(),
                d_fft_size);

            // Find the strongest code-phase match for this Doppler bin.
            volk_gnsssdr_32f_index_max_32u(
                &indext,
                d_magnitude.data(),
                d_fft_size);

            magt =
                d_magnitude[indext] /
                (fft_normalization_factor *
                 fft_normalization_factor);

            // Keep the strongest result found in the entire search.
            if (d_mag < magt)
                {
                    d_mag = magt;

                    if (d_test_statistics <
                            (d_mag / d_input_power) ||
                        !d_acq_params.bit_transition_flag)
                        {
                            d_gnss_synchro->
                                Acq_delay_samples =
                                static_cast<double>(
                                    indext %
                                    static_cast<int32_t>(
                                        d_acq_params
                                            .samples_per_code));

                            d_gnss_synchro->
                                Acq_doppler_hz =
                                static_cast<double>(
                                    doppler);

                            d_gnss_synchro->
                                Acq_samplestamp_samples =
                                samplestamp;

                            d_gnss_synchro->
                                Acq_doppler_step =
                                d_acq_params
                                    .doppler_step;

                            d_test_statistics =
                                d_mag /
                                d_input_power;
                        }
                }

            // Save correlation data when acquisition dumping is enabled.
            if (d_acq_params.dump)
                {
                    std::stringstream filename;

                    std::streamsize nbytes =
                        2 *
                        sizeof(float) *
                        d_fft_size;

                    filename
                        << "./test_statistics_"
                        << d_gnss_synchro->System
                        << "_"
                        << d_gnss_synchro->Signal[0]
                        << d_gnss_synchro->Signal[1]
                        << "_sat_"
                        << d_gnss_synchro->PRN
                        << "_doppler_"
                        << doppler
                        << ".dat";

                    d_dump_file.open(
                        filename.str().c_str(),
                        std::ios::out |
                            std::ios::binary);

                    d_dump_file.write(
                        reinterpret_cast<char*>(
                            d_ifft->get_outbuf()),
                        nbytes);

                    d_dump_file.close();
                }
        }

    // Decide whether acquisition succeeded or another dwell is needed.
    if (!d_acq_params.bit_transition_flag)
        {
            if (d_test_statistics >
                d_acq_params.threshold)
                {
                    d_state = 2;
                }
            else if (d_well_count ==
                     d_max_dwells)
                {
                    d_state = 3;
                }
        }
    else
        {
            // When checking for a bit transition, wait until all dwells
            // have been tested before making the final decision.
            if (d_well_count ==
                d_max_dwells)
                {
                    d_state =
                        (d_test_statistics >
                         d_acq_params.threshold)
                            ? 2
                            : 3;
                }
        }

    d_core_working = false;
}


// Run one PCPS acquisition dwell using CUDA.
void pcps_cuda_acquisition_cc::acquisition_core_cuda()
{
    int doppler;

    float fft_normalization_factor =
        static_cast<float>(d_fft_size);

    // Save the current dwell before advancing the dwell counter.
    uint32_t current_well = d_well_count;

    uint64_t samplestamp =
        d_sample_counter_buffer[current_well];

    d_input_power = 0.0;
    d_mag = 0.0;

    // Copy this dwell of input samples to the GPU.
    cuda_check(
        cudaMemcpy(
            d_cu_buffer_in,
            reinterpret_cast<const float2*>(
                d_in_buffer[current_well].data()),
            sizeof(float2) * d_fft_size,
            cudaMemcpyHostToDevice),
        "copy acquisition input to CUDA");

    d_well_count++;

    DLOG(INFO)
        << "Channel: " << d_channel
        << " , doing CUDA acquisition of satellite: "
        << d_gnss_synchro->System
        << " " << d_gnss_synchro->PRN
        << " , sample stamp: "
        << d_sample_counter
        << ", threshold: "
        << d_acq_params.threshold
        << ", doppler_min: "
        << d_acq_params.doppler_min
        << ", doppler_max: "
        << d_acq_params.doppler_max
        << ", doppler_step: "
        << d_acq_params.doppler_step;

    // Input power is still calculated on the CPU because it only
    // requires one short pass through the original signal samples.
    volk_32fc_magnitude_squared_32f(
        d_magnitude.data(),
        d_in_buffer[current_well].data(),
        d_fft_size);

    volk_32f_accumulator_s32f(
        &d_input_power,
        d_magnitude.data(),
        d_fft_size);

    d_input_power /=
        static_cast<float>(d_fft_size);

    // Begin timing the main CUDA acquisition work.
    auto start =
        std::chrono::steady_clock::now();

    // Make a Doppler-corrected copy of the signal for every bin.
    launch_mult_vectors_batched(
        d_cu_buffer_in,
        d_cu_buffer_grid_doppler_wipeoffs,
        d_cu_buffer_1,
        static_cast<int>(d_fft_size),
        static_cast<int>(d_fft_size_pow2),
        static_cast<int>(d_num_doppler_bins));

    // Transform every corrected signal into the frequency domain.
    cufft_check(
        cufftExecC2C(
            d_cu_fft_plan,
            reinterpret_cast<cufftComplex*>(
                d_cu_buffer_1),
            reinterpret_cast<cufftComplex*>(
                d_cu_buffer_2),
            CUFFT_FORWARD),
        "batched forward FFT");

    // Correlate every signal FFT with the local PRN code FFT.
    launch_mult_code_batched(
        d_cu_buffer_2,
        d_cu_buffer_fft_codes,
        d_cu_buffer_2,
        static_cast<int>(d_fft_size_pow2),
        static_cast<int>(d_num_doppler_bins));

    // Return the correlation results to the time domain.
    cufft_check(
        cufftExecC2C(
            d_cu_fft_plan,
            reinterpret_cast<cufftComplex*>(
                d_cu_buffer_2),
            reinterpret_cast<cufftComplex*>(
                d_cu_buffer_2),
            CUFFT_INVERSE),
        "batched inverse FFT");

    // Calculate the correlation strength at every code phase.
    launch_magnitude_squared_batched(
        d_cu_buffer_2,
        d_cu_buffer_magnitude,
        static_cast<int>(d_fft_size),
        static_cast<int>(d_fft_size_pow2),
        static_cast<int>(d_num_doppler_bins));

    // Divide the full search grid into groups for the maximum search.
    const int local_size = 64;

    const int total =
        static_cast<int>(
            d_fft_size *
            d_num_doppler_bins);

    const int num_groups =
        (total + local_size - 1) /
        local_size;

    // Each CUDA group reports its strongest result.
    launch_argmax_magnitude(
        d_cu_buffer_magnitude,
        d_cu_buffer_argmax_mag,
        d_cu_buffer_argmax_bin,
        d_cu_buffer_argmax_idx,
        static_cast<int>(d_fft_size),
        static_cast<int>(d_num_doppler_bins),
        num_groups);

    // Wait until all acquisition calculations are finished.
    cuda_check(
        cudaDeviceSynchronize(),
        "CUDA acquisition synchronize");

    // Copy the partial winners back to the CPU.
    std::vector<float> h_mag(num_groups);
    std::vector<int> h_bin(num_groups);
    std::vector<int> h_idx(num_groups);

    cuda_check(
        cudaMemcpy(
            h_mag.data(),
            d_cu_buffer_argmax_mag,
            sizeof(float) * num_groups,
            cudaMemcpyDeviceToHost),
        "copy argmax magnitudes");

    cuda_check(
        cudaMemcpy(
            h_bin.data(),
            d_cu_buffer_argmax_bin,
            sizeof(int) * num_groups,
            cudaMemcpyDeviceToHost),
        "copy argmax bins");

    cuda_check(
        cudaMemcpy(
            h_idx.data(),
            d_cu_buffer_argmax_idx,
            sizeof(int) * num_groups,
            cudaMemcpyDeviceToHost),
        "copy argmax indices");

    // Compare the partial winners to find the overall best result.
    float best_mag = -1.0f;
    int best_bin = 0;
    int best_idx = 0;

    for (int i = 0; i < num_groups; i++)
        {
            if (h_mag[i] > best_mag)
                {
                    best_mag = h_mag[i];
                    best_bin = h_bin[i];
                    best_idx = h_idx[i];
                }
        }

    // Normalize the correlation magnitude returned by the inverse FFT.
    d_mag =
        best_mag /
        (fft_normalization_factor *
         fft_normalization_factor);

    // Convert the winning bin number back into a Doppler frequency.
    doppler =
        d_acq_params.doppler_min +
        d_acq_params.doppler_step *
            best_bin;

    // Store the winning code phase, Doppler, and sample time.
    d_gnss_synchro->Acq_delay_samples =
        static_cast<double>(
            best_idx %
            static_cast<int32_t>(
                d_acq_params.samples_per_code));

    d_gnss_synchro->Acq_doppler_hz =
        static_cast<double>(doppler);

    d_gnss_synchro->
        Acq_samplestamp_samples =
        samplestamp;

    d_gnss_synchro->Acq_doppler_step =
        d_acq_params.doppler_step;

    // The detection statistic compares the correlation peak with
    // the average power of the input signal.
    d_test_statistics =
        d_mag /
        d_input_power;

    auto end =
        std::chrono::steady_clock::now();

    double elapsed =
        std::chrono::duration<double>(
            end - start)
            .count();

    cuda_acq_count++;
    cuda_total_time += elapsed;

    // Decide whether acquisition succeeded or another dwell is needed.
    if (!d_acq_params.bit_transition_flag)
        {
            if (d_test_statistics >
                d_acq_params.threshold)
                {
                    d_state = 2;
                }
            else if (d_well_count ==
                     d_max_dwells)
                {
                    d_state = 3;
                }
        }
    else
        {
            if (d_well_count ==
                d_max_dwells)
                {
                    d_state =
                        (d_test_statistics >
                         d_acq_params.threshold)
                            ? 2
                            : 3;
                }
        }

    // Print the time for this attempt and the running average.
    std::cout
        << std::fixed
        << std::setprecision(6);

    std::cout
        << "CUDA acquisition attempt "
        << cuda_acq_count
        << " time: "
        << elapsed
        << " s, average: "
        << cuda_total_time / cuda_acq_count
        << " s\n";

    d_core_working = false;
}


// GNU Radio calls this function when it has samples for the block.
//
// The block moves through four states:
//   0: waiting for acquisition to start
//   1: collecting and searching sample dwells
//   2: acquisition succeeded
//   3: acquisition failed
int pcps_cuda_acquisition_cc::general_work(
    int noutput_items,
    gr_vector_int& ninput_items,
    gr_vector_const_void_star& input_items,
    gr_vector_void_star& output_items)
{
    // Event values used by the GNSS-SDR channel state machine:
    // 0 = stop channel, 1 = acquisition success, 2 = acquisition failure.
    int acquisition_message = -1;

    switch (d_state)
        {
        case 0:
            {
                // Wait here until the channel activates acquisition.
                if (d_active)
                    {
                        // Clear results left over from an earlier search.
                        d_gnss_synchro->
                            Acq_delay_samples = 0.0;

                        d_gnss_synchro->
                            Acq_doppler_hz = 0.0;

                        d_gnss_synchro->
                            Acq_samplestamp_samples =
                            0ULL;

                        d_gnss_synchro->
                            Acq_doppler_step = 0U;

                        d_well_count = 0;
                        d_mag = 0.0;
                        d_input_power = 0.0;
                        d_test_statistics = 0.0;
                        d_in_dwell_count = 0;
                        d_sample_counter_buffer.clear();

                        // Begin collecting samples.
                        d_state = 1;
                    }

                // Samples are still counted while the block is idle.
                d_sample_counter +=
                    static_cast<uint64_t>(
                        d_fft_size) *
                    ninput_items[0];

                break;
            }

        case 1:
            {
                // Save input dwells until the requested number is reached.
                if (d_in_dwell_count <
                    d_max_dwells)
                    {
                        uint32_t num_dwells =
                            std::min(
                                static_cast<int>(
                                    d_max_dwells -
                                    d_in_dwell_count),
                                ninput_items[0]);

                        for (uint32_t i = 0;
                             i < num_dwells;
                             i++)
                            {
                                const auto* in =
                                    reinterpret_cast<
                                        const gr_complex*>(
                                        input_items[0]);

                                std::copy(
                                    in,
                                    in + d_fft_size,
                                    d_in_buffer[
                                        d_in_dwell_count++]
                                        .data());

                                d_sample_counter +=
                                    static_cast<uint64_t>(
                                        d_fft_size);

                                d_sample_counter_buffer
                                    .push_back(
                                        d_sample_counter);
                            }

                        // Count any samples that could not be stored.
                        if (ninput_items[0] >
                            static_cast<int>(
                                num_dwells))
                            {
                                d_sample_counter +=
                                    static_cast<uint64_t>(
                                        d_fft_size *
                                        (ninput_items[0] -
                                         num_dwells));
                            }
                    }
                else
                    {
                        // All dwell buffers are full, but incoming
                        // samples still need to be counted.
                        d_sample_counter +=
                            static_cast<uint64_t>(
                                d_fft_size) *
                            ninput_items[0];
                    }

                // Start a search when an unprocessed dwell is available.
                if ((d_well_count <
                     d_in_dwell_count) &&
                    !d_core_working &&
                    d_state == 1)
                    {
                        d_core_working = true;

                        // Use CUDA when initialization succeeded.
                        if (d_cuda == 0)
                            {
                                boost::thread(
                                    &pcps_cuda_acquisition_cc::
                                        acquisition_core_cuda,
                                    this);
                            }
                        else
                            {
                                // Fall back to the original CPU search.
                                boost::thread(
                                    &pcps_cuda_acquisition_cc::
                                        acquisition_core_volk,
                                    this);
                            }
                    }

                break;
            }

        case 2:
            {
                // The correlation peak passed the detection threshold.
                DLOG(INFO)
                    << "positive acquisition";

                DLOG(INFO)
                    << "satellite "
                    << d_gnss_synchro->System
                    << " "
                    << d_gnss_synchro->PRN;

                DLOG(INFO)
                    << "sample_stamp "
                    << d_sample_counter;

                DLOG(INFO)
                    << "test statistics value "
                    << d_test_statistics;

                DLOG(INFO)
                    << "test statistics threshold "
                    << d_acq_params.threshold;

                DLOG(INFO)
                    << "code phase "
                    << d_gnss_synchro->
                           Acq_delay_samples;

                DLOG(INFO)
                    << "doppler "
                    << d_gnss_synchro->
                           Acq_doppler_hz;

                DLOG(INFO)
                    << "magnitude "
                    << d_mag;

                DLOG(INFO)
                    << "input signal power "
                    << d_input_power;

                // Return to the idle state after reporting success.
                d_active = false;
                d_state = 0;

                d_sample_counter +=
                    static_cast<uint64_t>(
                        d_fft_size) *
                    ninput_items[0];

                acquisition_message = 1;

                this->message_port_pub(
                    pmt::mp("events"),
                    pmt::from_long(
                        acquisition_message));

                LOG(INFO)
                    << "Successful acquisition in channel "
                    << d_channel
                    << " for satellite "
                    << d_gnss_synchro->System
                    << " "
                    << d_gnss_synchro->PRN;

                // Send the synchronization result through the optional
                // monitor output.
                if (d_acq_params
                        .enable_monitor_output)
                    {
                        auto** out =
                            reinterpret_cast<
                                Gnss_Synchro**>(
                                &output_items[0]);

                        Gnss_Synchro
                            current_synchro_data =
                                Gnss_Synchro();

                        current_synchro_data =
                            *d_gnss_synchro;

                        *out[0] =
                            std::move(
                                current_synchro_data);

                        noutput_items = 1;
                    }

                break;
            }

        case 3:
            {
                // Every allowed dwell was searched without passing
                // the detection threshold.
                DLOG(INFO)
                    << "negative acquisition";

                DLOG(INFO)
                    << "satellite "
                    << d_gnss_synchro->System
                    << " "
                    << d_gnss_synchro->PRN;

                DLOG(INFO)
                    << "sample_stamp "
                    << d_sample_counter;

                DLOG(INFO)
                    << "test statistics value "
                    << d_test_statistics;

                DLOG(INFO)
                    << "test statistics threshold "
                    << d_acq_params.threshold;

                DLOG(INFO)
                    << "code phase "
                    << d_gnss_synchro->
                           Acq_delay_samples;

                DLOG(INFO)
                    << "doppler "
                    << d_gnss_synchro->
                           Acq_doppler_hz;

                DLOG(INFO)
                    << "magnitude "
                    << d_mag;

                DLOG(INFO)
                    << "input signal power "
                    << d_input_power;

                // Return to the idle state after reporting failure.
                d_active = false;
                d_state = 0;

                d_sample_counter +=
                    static_cast<uint64_t>(
                        d_fft_size) *
                    ninput_items[0];

                acquisition_message = 2;

                this->message_port_pub(
                    pmt::mp("events"),
                    pmt::from_long(
                        acquisition_message));

                break;
            }
        }

    // Tell GNU Radio that every available input item was consumed.
    consume_each(ninput_items[0]);

    return noutput_items;
}