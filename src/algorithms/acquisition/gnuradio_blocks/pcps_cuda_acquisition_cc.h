/*
 * File: pcps_cuda_acquisition_cc.h
 * Author: Haley Orr
 * GNSS-SDR Version: 0.0.21
 *
 * Purpose:
 * This file declares the CUDA version of the Parallel Code Phase Search
 * (PCPS) acquisition block used by GNSS-SDR.
 *
 * The block searches incoming GNSS samples for a satellite signal by
 * testing possible Doppler frequencies and PRN code positions. CUDA moves
 * these calculations to the GPU so that multiple Doppler bins and code
 * phases can be searched in parallel.
 *
 * A CPU-based acquisition path is also included as a fallback when CUDA
 * is unavailable.
 */

#ifndef GNSS_SDR_PCPS_CUDA_ACQUISITION_CC_H
#define GNSS_SDR_PCPS_CUDA_ACQUISITION_CC_H

#include "acq_conf.h"
#include "acquisition_impl_interface.h"
#include "channel_fsm.h"
#include "gnss_block_interface.h"
#include "gnss_sdr_fft.h"
#include "gnss_synchro.h"

#include <cuda_runtime.h>
#include <cufft.h>

#include <gnuradio/block.h>
#include <gnuradio/gr_complex.h>

#include <cstdint>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

/** \addtogroup Acquisition
 * \{ */
/** \addtogroup Acq_gnuradio_blocks
 * \{ */

// Tell the compiler that this class will be defined below.
class pcps_cuda_acquisition_cc;

// A shared pointer used to manage the acquisition block.
using pcps_cuda_acquisition_cc_sptr =
    gnss_shared_ptr<pcps_cuda_acquisition_cc>;

// Create and return a new CUDA acquisition block.
pcps_cuda_acquisition_cc_sptr pcps_make_cuda_acquisition_cc(
    const Acq_Conf& conf,
    uint32_t max_dwells);

class pcps_cuda_acquisition_cc : public acquisition_impl_interface
{
public:
    // Release the CPU and GPU resources used by this block.
    ~pcps_cuda_acquisition_cc();

    // Give the block the object where acquisition results will be stored.
    inline void set_gnss_synchro(
        Gnss_Synchro* p_gnss_synchro) override
    {
        d_gnss_synchro = p_gnss_synchro;
    }

    // Return the strongest correlation magnitude found.
    inline uint32_t mag() const override
    {
        return d_mag;
    }

    // Store the local PRN code for the satellite being searched for.
    void set_local_code(std::complex<float>* code) override;

    // Turn acquisition on or off.
    inline void set_active(bool active) override
    {
        // Return to the idle state when acquisition is turned off.
        if (!active)
            {
                d_state = 0;
            }

        d_active = active;
    }

    // Tell the block which GNSS-SDR channel it belongs to.
    inline void set_channel(uint32_t channel) override
    {
        d_channel = channel;
    }

    // Connect this block to the channel state machine.
    inline void set_channel_fsm(
        std::weak_ptr<ChannelFsm> channel_fsm) override
    {
        d_channel_fsm = channel_fsm;
    }

    // CUDA is ready when initialization returned a success code of zero.
    inline bool cuda_ready() const
    {
        return d_cuda == 0;
    }

    // Run the acquisition search on the CPU.
    void acquisition_core_volk();

    // Run the acquisition search on the GPU.
    void acquisition_core_cuda();

    // GNU Radio calls this when new input samples are available.
    int general_work(
        int noutput_items,
        gr_vector_int& ninput_items,
        gr_vector_const_void_star& input_items,
        gr_vector_void_star& output_items) override;

private:
    // Let the creation function call the private constructor.
    friend pcps_cuda_acquisition_cc_sptr
    pcps_make_cuda_acquisition_cc(
        const Acq_Conf& conf,
        uint32_t max_dwells);

    // Objects should be created through pcps_make_cuda_acquisition_cc().
    explicit pcps_cuda_acquisition_cc(
        const Acq_Conf& conf,
        uint32_t max_dwells);

    // Calculate CPU correlation magnitudes for one Doppler bin.
    void calculate_magnitudes(
        gr_complex* fft_begin,
        int doppler_shift,
        int doppler_offset);

    // Select the CUDA device and prepare its resources.
    int init_cuda_environment();

    // GPU memory

    // Input samples copied from the CPU to the GPU.
    float2* d_cu_buffer_in;

    // FFT of the local satellite PRN code.
    float2* d_cu_buffer_fft_codes;

    // Temporary buffers used during GPU calculations.
    float2* d_cu_buffer_1;
    float2* d_cu_buffer_2;

    // Correlation magnitude for every search position.
    float* d_cu_buffer_magnitude;

    // Strongest magnitude found by the GPU.
    float* d_cu_buffer_argmax_mag;

    // Doppler bin containing the strongest result.
    int* d_cu_buffer_argmax_bin;

    // Code-phase index containing the strongest result.
    int* d_cu_buffer_argmax_idx;

    // Doppler corrections used for the parallel frequency search.
    float2* d_cu_buffer_grid_doppler_wipeoffs;

    // CUDA FFT setup

    // Describes the FFT operations performed on the GPU.
    cufftHandle d_cu_fft_plan;

    // Number of FFTs processed together.
    int d_cu_fft_batch_size;

    // Acquisition configuration and output

    // Text description of the satellite being searched for.
    std::string d_satellite_str;

    // Settings supplied by the GNSS-SDR configuration file.
    const Acq_Conf d_acq_params;

    // Optional file used to save debugging information.
    std::ofstream d_dump_file;

    // Object where the final acquisition result is reported.
    Gnss_Synchro* d_gnss_synchro;

    // Search results and counters

    // Total number of input samples processed.
    uint64_t d_sample_counter;

    // Strongest Doppler result found during each search.
    int* d_max_doppler_indexs;

    // Strongest correlation magnitude found.
    float d_mag;

    // Estimated power of the input signal.
    float d_input_power;

    // Value compared with the acquisition threshold.
    float d_test_statistics;

    // Current acquisition state.
    int d_state;

    // CUDA status. Zero means initialization succeeded.
    int d_cuda;

    // Maximum number of sample blocks that may be tested.
    uint32_t d_max_dwells;

    // Number of sample blocks tested so far.
    uint32_t d_well_count;

    // Number of samples included in each FFT.
    const uint32_t d_fft_size;

    // FFT size rounded to a power of two when needed.
    uint32_t d_fft_size_pow2;

    // Number of Doppler frequency bins in the search.
    uint32_t d_num_doppler_bins;

    // Code phase containing the strongest result.
    uint32_t d_code_phase;

    // GNSS-SDR channel using this block.
    uint32_t d_channel;

    // Current dwell position inside the input buffer.
    uint32_t d_in_dwell_count;

    // True when this block should perform acquisition.
    bool d_active;

    // Prevents the acquisition core from starting twice.
    bool d_core_working;

    // Receives acquisition success or failure events.
    std::weak_ptr<ChannelFsm> d_channel_fsm;

    // CPU fallback

    // Forward FFT used by the CPU acquisition path.
    std::unique_ptr<gnss_fft_complex_fwd> d_fft_if;

    // Inverse FFT used to calculate code-phase correlations.
    std::unique_ptr<gnss_fft_complex_rev> d_ifft;

    // Doppler corrections used by the CPU search.
    std::vector<std::vector<gr_complex>> d_grid_doppler_wipeoffs;

    // CPU input samples organized by dwell.
    std::vector<std::vector<gr_complex>> d_in_buffer;

    // FFT of the local PRN code used by the CPU path.
    std::vector<gr_complex> d_fft_codes;

    // Empty vector used to clear or initialize buffers.
    std::vector<gr_complex> d_zero_vector;

    // Sample count associated with each stored dwell.
    std::vector<uint64_t> d_sample_counter_buffer;

    // Correlation magnitudes calculated by the CPU.
    std::vector<float> d_magnitude;
};


#endif  // GNSS_SDR_PCPS_CUDA_ACQUISITION_CC_H