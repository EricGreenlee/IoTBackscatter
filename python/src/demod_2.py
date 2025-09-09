import matplotlib.pyplot as plt
import numpy as np
from IoTBSConst import bitsPerPacket, gc_len, sync_seq
from main import logger
from scipy import signal


def plot_time_psd_scat(samples, samp_rate, title):
    plt.figure()
    
    plt.subplot(3, 1, 1)
    plt.plot(np.real(samples),'.-')
    plt.plot(np.imag(samples),'.-')
    plt.title(title)
    plt.legend(['real', 'imag'])
    plt.grid('on')

    plt.subplot(3,1,2)
    plt.psd(samples, NFFT=1024, Fs=samp_rate)
    plt.title('PSD of '+title)

    plt.subplot(3, 1, 3)
    plt.scatter(np.real(samples), np.imag(samples))
    plt.title(title+' constellation')
    plt.grid('on')
    plt.axis('equal')
    
def lpf(samples, data_rate, samp_rate):
    #filter
    cutoff_freq = data_rate*1.5
    b, a = signal.butter(5, cutoff_freq, 'low', fs=samp_rate)
    out_samples = signal.filtfilt(b, a, samples)
    return(out_samples)

def agc(samples,out_amplitude):
    mag = np.sum(np.abs(samples)**2)/len(samples)
    gain = out_amplitude/np.sqrt(mag)
    samples_out = samples*gain
    return(samples_out) 

def coarse_freq_correction(input_samples, target_corr_samples, fs_hz, max_freq_dev_hz, nfreq_pts, enable_plotting=False):
    """
    Coarse frequency correction using Gold code correlation
    """
    freq_offsets_hz = np.linspace(-max_freq_dev_hz, max_freq_dev_hz, nfreq_pts)
    logger.debug("Test frequency offsets: %s", freq_offsets_hz)
    
    time_delays_sec = np.linspace(0, (len(input_samples)-1)/fs_hz, len(input_samples))
    corr_len = len(input_samples)-len(target_corr_samples)+1
    
    # Create meshgrid for time delay and frequency offset
    Z = np.zeros((len(freq_offsets_hz), corr_len))
    
    for j, freq_offset_hz in enumerate(freq_offsets_hz):
        test_samples = input_samples * np.exp(+1j*2*np.pi*freq_offset_hz*time_delays_sec)
        correlation = np.correlate(test_samples, target_corr_samples, mode='valid')
        Z[j, :] = np.abs(correlation)
    
    max_corr_idx = np.unravel_index(np.argmax(Z), Z.shape)
    freq_adj_hz = freq_offsets_hz[max_corr_idx[0]]
    
    logger.info(f"Coarse freq correction: {freq_adj_hz} Hz")
    
    # Apply frequency correction
    output_samples = input_samples * np.exp(1j*2*np.pi*freq_adj_hz*time_delays_sec)
    
    return output_samples, freq_adj_hz

def multi_hypothesis_despread(samples, goldcode, sps, num_hypotheses=8):
    """
    Despread using multiple timing hypotheses to handle timing uncertainty
    """
    gc_len = len(goldcode)
    spread_goldcode = np.repeat(goldcode.astype(np.complex64), sps)
    
    # Create timing hypotheses (fractional sample offsets)
    timing_offsets = np.linspace(0, sps-1, num_hypotheses, dtype=int)
    
    despread_outputs = []
    correlation_powers = []
    
    for offset in timing_offsets:
        # Apply timing offset
        if offset > 0:
            offset_samples = np.concatenate([np.zeros(offset, dtype=np.complex64), samples[:-offset]])
        else:
            offset_samples = samples
            
        # Despread with this timing hypothesis
        num_symbols = len(offset_samples) // (gc_len * sps)
        despread_output = np.zeros(num_symbols, dtype=np.complex64)
        
        for i in range(num_symbols):
            start_idx = i * gc_len * sps
            end_idx = start_idx + gc_len * sps
            if end_idx <= len(offset_samples):
                # Correlate with Gold code and integrate
                symbol_samples = offset_samples[start_idx:end_idx]
                despread_chips = symbol_samples * np.conj(spread_goldcode)
                # Integrate over chips (sum every sps samples)
                integrated = np.sum(despread_chips.reshape(-1, sps), axis=1)
                despread_output[i] = np.sum(integrated)
        
        despread_outputs.append(despread_output)
        # Calculate correlation power as a metric for timing quality
        correlation_powers.append(np.sum(np.abs(despread_output)**2))
    
    # Select best timing hypothesis
    best_hypothesis = np.argmax(correlation_powers)
    logger.info(f"Best timing hypothesis: {timing_offsets[best_hypothesis]} samples offset")
    logger.info(f"Correlation powers: {correlation_powers}")
    
    return despread_outputs[best_hypothesis], timing_offsets[best_hypothesis]

def adaptive_correlation_search(samples, target_pattern, threshold_factor=0.5, enable_plotting=False):
    """
    Adaptive correlation with CFAR-like threshold detection
    """
    correlation = np.correlate(samples, target_pattern, mode='valid')
    corr_mag = np.abs(correlation)
    
    # Adaptive threshold based on local statistics
    window_size = min(1000, len(corr_mag) // 10)
    if window_size > 0:
        # Calculate rolling mean and std for adaptive threshold
        padded_corr = np.pad(corr_mag, window_size//2, mode='edge')
        rolling_mean = np.convolve(padded_corr, np.ones(window_size)/window_size, mode='valid')
        rolling_std = np.sqrt(np.convolve(padded_corr**2, np.ones(window_size)/window_size, mode='valid') - rolling_mean**2)
        adaptive_threshold = rolling_mean + threshold_factor * rolling_std
    else:
        adaptive_threshold = np.mean(corr_mag) + threshold_factor * np.std(corr_mag)
    
    # Find peaks above adaptive threshold
    if isinstance(adaptive_threshold, np.ndarray):
        peaks = []
        for i in range(len(corr_mag)):
            if i > 0 and i < len(corr_mag)-1:
                if (corr_mag[i] > corr_mag[i-1] and corr_mag[i] > corr_mag[i+1] and 
                    corr_mag[i] > adaptive_threshold[i]):
                    peaks.append(i)
        peaks = np.array(peaks)
    else:
        peaks, _ = signal.find_peaks(corr_mag, height=adaptive_threshold)
    
    if enable_plotting:
        plt.figure()
        plt.plot(corr_mag, label='Correlation')
        if isinstance(adaptive_threshold, np.ndarray):
            plt.plot(adaptive_threshold, 'r--', label='Adaptive Threshold')
        else:
            plt.axhline(y=adaptive_threshold, color='r', linestyle='--', label=f'Threshold={adaptive_threshold:.1f}')
        if len(peaks) > 0:
            plt.plot(peaks, corr_mag[peaks], 'rx', markersize=10, label='Detected Peaks')
        plt.title('Adaptive Correlation Detection')
        plt.xlabel('Sample Index')
        plt.ylabel('Correlation Magnitude')
        plt.legend()
        plt.grid(True)
    
    return correlation, peaks

def robust_timing_recovery(samples, goldcode, sps, fs_hz, enable_plotting=False):
    """
    Enhanced timing recovery with coarse and fine stages
    """
    # Stage 1: Coarse timing using Gold code correlation
    resampled_gc = np.repeat(goldcode.astype(np.complex64), sps)
    correlation, peaks = adaptive_correlation_search(samples, resampled_gc, enable_plotting=enable_plotting)
    
    if len(peaks) == 0:
        logger.warning("No timing peaks found!")
        return samples, 0
        
    # Get the first strong peak
    first_peak_index = peaks[0]
    first_peak_value = correlation[first_peak_index]
    first_peak_sign = np.sign(np.real(first_peak_value))
    
    logger.info("Coarse timing peak-> sample: %s, value: %s, sign: %s", 
                first_peak_index, first_peak_value, first_peak_sign)
    
    # Roll to align with first peak
    aligned_samples = np.roll(samples, -first_peak_index)
    
    # Stage 2: Fine timing using peak spacing
    if len(peaks) > 1:
        peak_diffs = np.diff(peaks)
        avg_peak_diff = np.mean(peak_diffs)
        expected_peak_diff = gc_len * sps  # Expected samples between symbols
        
        logger.info("Found %d peaks, avg spacing: %.2f, expected: %d", 
                    len(peaks), avg_peak_diff, expected_peak_diff)
        
        # Fine resampling correction
        resample_ratio = expected_peak_diff / avg_peak_diff
        logger.info("Fine resampling ratio: %.6f", resample_ratio)
        
        if abs(resample_ratio - 1.0) > 0.001:  # Only resample if significant error
            aligned_samples = signal.resample_poly(aligned_samples, 
                                                 int(resample_ratio*1000), 1000)
            logger.info("Applied fine resampling, new length: %d", len(aligned_samples))
    
    return aligned_samples, first_peak_sign

def mm_time_recovery(samples, samps_per_symbol, mu_init=0.0):
    """
    Enhanced Mueller & Muller timing recovery with better initialization
    """
    samples_interpolated = signal.resample_poly(samples, 16, 1)
    mu = mu_init
    out = np.zeros(len(samples) + 10, dtype=np.complex64)
    out_rail = np.zeros(len(samples) + 10, dtype=np.complex64)
    i_in = 0
    i_out = 2
    
    # Adaptive loop gain
    alpha = 0.3
    
    while i_out < len(samples) and i_in+16 < len(samples):
        out[i_out] = samples_interpolated[i_in*16 + int(mu*16)]
        out_rail[i_out] = int(np.real(out[i_out]) > 0) + 1j*int(np.imag(out[i_out]) > 0)
        
        x = (out_rail[i_out] - out_rail[i_out-2]) * np.conj(out[i_out-1])
        y = (out[i_out] - out[i_out-2]) * np.conj(out_rail[i_out-1])
        mm_val = np.real(y - x)
        
        mu += samps_per_symbol + alpha * mm_val
        i_in += int(np.floor(mu))
        mu = mu - np.floor(mu)
        i_out += 1
        
    out = out[2:i_out]
    return out

def enhanced_costas_loop(samples, samp_rate, loop_bw=0.01):
    """
    Enhanced Costas loop with adaptive bandwidth
    """
    N = len(samples)
    phase = 0
    freq = 0
    
    # Convert loop bandwidth to alpha/beta
    damping = 0.707  # Critical damping
    denom = (1.0 + 2.0*damping*loop_bw + loop_bw*loop_bw)
    alpha = (4*damping*loop_bw) / denom
    beta = (4*loop_bw*loop_bw) / denom
    
    out = np.zeros(N, dtype=np.complex64)
    freq_log = []
    error_log = []
    
    for i in range(N):
        out[i] = samples[i] * np.exp(-1j*phase)
        
        # BPSK phase error detector
        error = np.real(out[i]) * np.imag(out[i])
        error_log.append(error)
        
        # Loop filter
        freq += (beta * error)
        freq_log.append(freq * samp_rate / (2*np.pi))
        phase += freq + (alpha * error)
        
        # Wrap phase
        while phase >= 2*np.pi:
            phase -= 2*np.pi
        while phase < 0:
            phase += 2*np.pi
    
    return out

def demod_bpsk(samples):
    nbits = len(samples)
    bits = np.zeros(nbits)
    for i in range(nbits):
        bits[i] = int(np.real(samples[i]) > 0)
    return bits.astype(int)

def sync_word_search_correction(in_bits, sync_seq, payload_len):
    in_bits_bip = in_bits*2-1
    sync_seq_bip = sync_seq*2-1
    
    sync_corr = np.correlate(in_bits_bip, sync_seq_bip, mode='valid')
    sync_thresh = len(sync_seq)*0.5  # More reasonable threshold
    peaks, _ = signal.find_peaks(abs(sync_corr), height=sync_thresh)
    
    if len(peaks) > 0:
        first_peak_sample_num = peaks[np.argmax(abs(sync_corr[peaks]))]
        first_peak_value = sync_corr[first_peak_sample_num]
        first_peak_sign = np.sign(first_peak_value)
        
        logger.info("Sync peak-> sample: %s, value: %s, sign: %s", 
                    first_peak_sample_num, first_peak_value, first_peak_sign)
    else:
        logger.warning("No sync peaks found!")
        first_peak_sample_num = 0
        first_peak_sign = 1
        
    out_all_bits = (in_bits_bip*first_peak_sign+1)/2
    out_payload_bits = out_all_bits[first_peak_sample_num+len(sync_seq):
                                   first_peak_sample_num+len(sync_seq)+payload_len]
    
    return out_all_bits, out_payload_bits

def demodulate_packet(input_samples, tag_params, radio_params, enable_plotting=False):
    logger.info("Processing %s samples generated by %s tags", len(input_samples), tag_params.ntags) 
    
    start_ind = 127000+124000
    num_samps = 1270*100
    subset_samples = input_samples[start_ind:start_ind+num_samps]
    
    # Mix to center frequency
    offset_freq_hz = 505
    center_freq_hz = -50e3+offset_freq_hz
    time_array_sec = np.linspace(0,(len(subset_samples)-1)/radio_params.samplerate_hz, len(subset_samples))
    mixed_samples = subset_samples*np.exp(-1j*2*np.pi*center_freq_hz*time_array_sec)
    
    logger.info("start_ind: %s, num_samps: %s, offset_freq_hz: %s", start_ind, num_samps, offset_freq_hz)
    
    # Low pass filter
    filt_samples = lpf(mixed_samples, 25e3, radio_params.samplerate_hz)
    
    if enable_plotting:
        plot_time_psd_scat(subset_samples, radio_params.samplerate_hz, "Raw Received Samples")
        plot_time_psd_scat(mixed_samples, radio_params.samplerate_hz, "Mixed Samples")
        plot_time_psd_scat(filt_samples, radio_params.samplerate_hz, "Filtered Samples")
    
    for itag in range(tag_params.ntags):
        logger.info("Demodding tag %s", itag)
        itagParams = tag_params.get_tag(itag)
        proc_samples = filt_samples
        
        # AGC
        proc_samples = agc(proc_samples, 1/np.sqrt(2))
        if enable_plotting:
            plot_time_psd_scat(proc_samples, radio_params.samplerate_hz, "Filtered & AGCed Samples")
        
        # ENHANCED PIPELINE STARTS HERE
        
        # Stage 1: Coarse frequency correction using Gold code correlation
        max_input_freq_dev_hz = 1000
        max_output_freq_dev_hz = 100
        npts_freq_search = int(np.ceil(max_input_freq_dev_hz * 2 / max_output_freq_dev_hz) + 1)
        resampled_gc = np.repeat(itagParams.goldcode.astype(np.complex64), itagParams.sps)
        
        proc_samples, freq_correction = coarse_freq_correction(
            proc_samples, resampled_gc, radio_params.samplerate_hz, 
            max_input_freq_dev_hz, npts_freq_search, enable_plotting)
        
        if enable_plotting:
            plot_time_psd_scat(proc_samples, radio_params.samplerate_hz, "After Coarse Freq Correction")
        
        # Stage 2: Robust timing recovery (coarse + fine)
        proc_samples, timing_sign = robust_timing_recovery(
            proc_samples, itagParams.goldcode, itagParams.sps, 
            radio_params.samplerate_hz, enable_plotting)
        
        if enable_plotting:
            plot_time_psd_scat(proc_samples, radio_params.samplerate_hz, "After Timing Recovery")
        
        # Stage 3: EARLY DESPREADING with multiple hypotheses
        despread_samples, best_timing_offset = multi_hypothesis_despread(
            proc_samples, itagParams.goldcode, itagParams.sps, num_hypotheses=8)
        
        if enable_plotting:
            plot_time_psd_scat(despread_samples, radio_params.samplerate_hz/itagParams.sps/gc_len, 
                             "After Early Despreading")
        
        # Stage 4: Fine timing recovery on despread signal
        # Now working with symbol-rate samples after despreading
        symbol_rate_sps = 4  # Samples per symbol after despreading
        despread_samples_upsampled = signal.resample_poly(despread_samples, symbol_rate_sps, 1)
        
        proc_samples = mm_time_recovery(despread_samples_upsampled, symbol_rate_sps)
        if enable_plotting:
            plot_time_psd_scat(proc_samples, radio_params.samplerate_hz/gc_len, "After Fine MM Timing Recovery")
        
        # Stage 5: Enhanced Costas loop for fine frequency correction
        proc_samples = enhanced_costas_loop(proc_samples, radio_params.samplerate_hz/gc_len)
        if enable_plotting:
            plot_time_psd_scat(proc_samples, radio_params.samplerate_hz/gc_len, "After Enhanced Costas Loop")
        
        # Stage 6: BPSK demodulation
        rx_bits_raw = demod_bpsk(proc_samples)
        logger.debug("rx_bits_raw length: %d", len(rx_bits_raw))
        
        # Stage 7: Sync word search and bit correction
        nbits_data = len(itagParams.actual_bits)
        
        # Apply timing sign correction if needed
        if timing_sign < 0:
            rx_bits_raw = 1 - rx_bits_raw
        
        corrected_proc_bits, rx_payload_bits = sync_word_search_correction(
            rx_bits_raw, sync_seq, bitsPerPacket)
        
        logger.info("Transmitted actual bits:  %s", itagParams.actual_bits)
        logger.info("Corrected processed bits: %s", corrected_proc_bits.astype(int)) 
        logger.info("Raw processed bits:       %s", rx_bits_raw[:len(itagParams.actual_bits)].astype(int)) 
        
        logger.info("Transmitted payload bits:  %s", itagParams.payload_bits)
        logger.info("Received payload bits:     %s", rx_payload_bits.astype(int))
        
        # Calculate BER
        num_bits = len(itagParams.payload_bits)
        try:
            num_errors = sum(abs(rx_payload_bits-itagParams.payload_bits))
            BER = num_errors/num_bits
        except:
            logger.warning("BER unable to be computed")
            num_errors = num_bits
            BER = 1.0
        
        logger.info("BER (tag %s): %s", itag, BER) 
        
        # Store results for this tag
        if itag == 0:
            results = {}
        results[itag] = {
            'num_errors': num_errors,
            'num_bits': num_bits,
            'ber': BER
        }
       
    return results