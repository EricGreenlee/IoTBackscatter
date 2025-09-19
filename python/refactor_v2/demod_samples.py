import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from main import logger

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
    
    max_corr = np.argmax(Z)
    max_corr_idx = np.unravel_index(max_corr, Z.shape)
    freq_adj_hz = freq_offsets_hz[max_corr_idx[0]]
    
    logger.info(f"Coarse freq correction: {freq_adj_hz} Hz, max correlation: {max_corr}")
    
    # Apply frequency correction
    output_samples = input_samples * np.exp(1j*2*np.pi*freq_adj_hz*time_delays_sec)
    
    return output_samples, freq_adj_hz

def coarse_freq_correction_code_search(input_samples, target_corr_samples, fs_hz, freq_offsets_hz, enable_plotting=False):
    """
    Coarse frequency correction using Gold code correlation
    """
    # freq_offsets_hz = np.linspace(-max_freq_dev_hz, max_freq_dev_hz, nfreq_pts)
    logger.debug("Test frequency offsets: %s, len: %s", freq_offsets_hz, len(freq_offsets_hz))
    
    time_delays_sec = np.linspace(0, (len(input_samples)-1)/fs_hz, len(input_samples))
    corr_len = len(input_samples)-len(target_corr_samples)+1
    
    # Create meshgrid for time delay and frequency offset
    X, Y = np.meshgrid(time_delays_sec[0:corr_len], freq_offsets_hz)
    Z = np.zeros((len(freq_offsets_hz), corr_len))
    
    for j, freq_offset_hz in enumerate(freq_offsets_hz):
        test_samples = input_samples * np.exp(+1j*2*np.pi*freq_offset_hz*time_delays_sec)
        correlation = np.correlate(test_samples, target_corr_samples, mode='valid')
        Z[j, :] = np.abs(correlation)
        
    if enable_plotting:
        fig_lab = plt.figure()
        ax = fig_lab.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, Z[:,:], cmap='viridis')
        ax.set_title(f'Correlation with Repeated Goldcode')

        # Set labels
        ax.set_xlabel('Time Delay (s)')
        ax.set_ylabel('Frequency Offset (Hz)')
        ax.set_zlabel('Signal Strength')
    
    max_corr = np.max(Z)
    max_corr_idx = np.unravel_index(np.argmax(Z), Z.shape)
    freq_adj_hz = freq_offsets_hz[max_corr_idx[0]]
    
    logger.info(f"Coarse freq correction: {freq_adj_hz} Hz, max correlation: {max_corr}")
    
    # Apply frequency correction
    output_samples = input_samples * np.exp(1j*2*np.pi*freq_adj_hz*time_delays_sec)
    
    return output_samples, freq_adj_hz, max_corr

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

def robust_timing_recovery(samples, goldcode, sps, fs_hz,  enable_plotting=False):
    """
    Enhanced timing recovery with coarse and fine stages
    """
    gc_len = len(goldcode)
    
    # Stage 1: Coarse timing using Gold code correlation
    resampled_gc = np.repeat(goldcode.astype(np.complex64), sps)
    correlation, peaks = adaptive_correlation_search(samples, resampled_gc, threshold_factor= 8, enable_plotting=enable_plotting)
    
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

def time_freq_code_search(input_samples, target_corr_samples, input_sps, fs_hz, max_freq_dev_hz, nfreq_pts : int, enable_plotting=False):
    #must get the frequency to within ~150 Hz for the costas loop to be able to correct it
    
    freq_offsets_hz = np.linspace(-1*max_freq_dev_hz, max_freq_dev_hz, nfreq_pts)#[-100,-50,0,50,100]
    logger.debug("Test frequency offsets: %s", freq_offsets_hz)
    # freq_offsets_hz = [0]
    time_delays_sec = np.linspace(0, (len(input_samples)-1)/fs_hz, len(input_samples))
    
    # target_gc_stretch = np.repeat(target_gc,input_sps)
    
    # logger.info(f"len(samples): {len(samples)}, len(target_gc_stretch): {len(target_gc_stretch)}, dif: {len(samples)-len(target_gc_stretch)}")
    corr_len = len(input_samples)-len(target_corr_samples)+1
    # logger.info(f"corr_len: {corr_len}")
    
    # Create meshgrid for time delay and frequency offset
    X, Y = np.meshgrid(time_delays_sec[0:corr_len], freq_offsets_hz)
    Z = np.zeros((len(freq_offsets_hz), corr_len))
    
    for j, freq_offset_hz in enumerate(freq_offsets_hz):
        # logger.info(f"Loop {j}, freq_offset_hz: {freq_offset_hz}")
        test_samples = input_samples * np.exp(+1j*2*np.pi*freq_offset_hz*time_delays_sec)
        
        correlation = np.correlate(test_samples, target_corr_samples, mode='valid')
        Z[j, :] = np.abs(correlation)
    
        if enable_plotting:
            plt.figure()
            plt.plot(abs(correlation))
            plt.grid("on")
            plt.title(f"Correlation with Repeated Goldcode, freq_offset_hz: {freq_offset_hz}")
    
    if enable_plotting:
        fig_lab = plt.figure()
        ax = fig_lab.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, Z[:,:], cmap='viridis')
        ax.set_title(f'Correlation with Repeated Goldcode')

        # Set labels
        ax.set_xlabel('Time Delay (s)')
        ax.set_ylabel('Frequency Offset (Hz)')
        ax.set_zlabel('Signal Strength')
    
    max_corr = np.max(Z)
    max_corr_idx = np.argmax(Z)
    max_corr_idx = np.unravel_index(max_corr_idx, Z.shape)
    freq_adj_hz = freq_offsets_hz[max_corr_idx[0]]

    logger.info(f"max_corr: {max_corr}")
    logger.info(f"freq index of max correlation: {max_corr_idx[0]}, corresponding freq_adj_hz: {freq_adj_hz}")
    # logger.info(f"time index of max correlation: {max_corr_idx[1]}")
    
    logger.info("freq_offsets_hz: %s", freq_offsets_hz)
    # logger.info("time_delays_sec[0:5]: %s", time_delays_sec[0:5])
    
    output_samples = input_samples * np.exp(1j*2*np.pi*freq_adj_hz*time_delays_sec)
    correlation = np.correlate(output_samples, target_corr_samples, mode='valid')
    
    if (enable_plotting):
            plt.figure()
            plt.plot(abs(correlation))
            # plt.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold={threshold}')
            plt.title('Correlation with Preamble Pattern directly after mixing')
            plt.xlabel('Sample Index')
            plt.ylabel('Correlation Magnitude')
            plt.legend()
            plt.grid(True)
        

    threshold = 400 #approximately sps*gclen*0.5
    peaks, properties = signal.find_peaks(abs(correlation), height=threshold, prominence=threshold*0.3)
    
    # Get the first peak above threshold
    if len(peaks) > 0:
        first_peak_index = peaks[0]
        first_peak_value = correlation[first_peak_index]
        first_peak_time_sec = first_peak_index/fs_hz
        first_peak_sign = np.sign(np.real(first_peak_value))
      
        logger.info("first peak-> sample number: %s, value: %s, time: %s, sign: %s", first_peak_index, first_peak_value, first_peak_time_sec, first_peak_sign)
    else:
        logger.warning("No peaks found!")
        first_peak_index = 0
        first_peak_sign = 1
        
    #roll to the first peak
    intermediate_samples = np.roll(output_samples, -1*first_peak_index)
    
    #find the sps and resample
    if len(peaks) > 1:
        # Compute differences between consecutive peak indexes
        peak_diffs = np.diff(peaks)
        avg_peak_diff = np.mean(peak_diffs)
        expected_peak_diff = 1270  # Expected samples between peaks
        
        logger.info("Found %d peaks at indexes: %s", len(peaks), peaks.tolist())
        logger.info("Peak differences: %s", peak_diffs.tolist())
        logger.info("Average peak difference: %.2f samples", avg_peak_diff)
        logger.info("Expected peak difference: %d samples", expected_peak_diff)
        # logger.info("Average peak difference: %.3f ms", avg_peak_diff/samplerate_hz*1000)
        
        # Resample signal to correct timing
        resample_ratio = expected_peak_diff / avg_peak_diff
        logger.info("Resampling ratio: %.6f", resample_ratio)
        
        output_samples = signal.resample_poly(intermediate_samples, int(resample_ratio*1000), 1000)
        logger.info("Resampled signal length: %d samples", len(output_samples))
        
        output_samples = output_samples[0:1270*(96+4)]
        logger.info("Truncated signal to %d samples", len(output_samples))
        
        # Plot correlation and peaks
        if enable_plotting:
            plt.figure(figsize=(12, 6))
            plt.subplot(2, 1, 1)
            plt.plot(abs(correlation))
            plt.plot(peaks, abs(correlation)[peaks], 'rx', markersize=10, label='Detected Peaks')
            plt.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold={threshold}')
            plt.title('Correlation with Preamble Pattern')
            plt.xlabel('Sample Index')
            plt.ylabel('Correlation Magnitude')
            plt.legend()
            plt.grid(True)
            
            plt.subplot(2, 1, 2)
            plt.plot(range(len(peak_diffs)), peak_diffs, 'bo-')
            plt.axhline(y=avg_peak_diff, color='r', linestyle='--', label=f'Average={avg_peak_diff:.1f}')
            plt.title('Peak Spacing')
            plt.xlabel('Peak Pair Index')
            plt.ylabel('Sample Difference')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
    
    
    return output_samples


def time_code_resamp_search(input_samples, target_corr_samples, input_sps, fs_hz, enable_plotting=False):
    
    proc_samples = input_samples
    correlation = np.correlate(proc_samples, target_corr_samples, mode='valid')        

    threshold = 400 #approximately sps*gclen*0.5
    peaks, properties = signal.find_peaks(abs(correlation), height=threshold, prominence=threshold*0.3)
    
    # Get the first peak above threshold
    if len(peaks) > 0:
        first_peak_index = peaks[0]
        first_peak_value = correlation[first_peak_index]
        first_peak_time_sec = first_peak_index/fs_hz
        first_peak_sign = np.sign(np.real(first_peak_value))
      
        logger.info("first peak-> sample number: %s, value: %s, time: %s, sign: %s", first_peak_index, first_peak_value, first_peak_time_sec, first_peak_sign)
    else:
        logger.warning("No peaks found!")
        first_peak_index = 0
        first_peak_sign = 1
        
    #roll to the first peak
    proc_samples = np.roll(proc_samples, -1*first_peak_index)
    
    #find the sps and resample
    if len(peaks) > 1:
        # Compute differences between consecutive peak indexes
        peak_diffs = np.diff(peaks)
        avg_peak_diff = np.mean(peak_diffs)
        expected_peak_diff = 1270  # Expected samples between peaks
        
        logger.info("Found %d peaks at indexes: %s", len(peaks), peaks.tolist())
        logger.info("Peak differences: %s", peak_diffs.tolist())
        logger.info("Average peak difference: %.2f samples", avg_peak_diff)
        logger.info("Expected peak difference: %d samples", expected_peak_diff)
        # logger.info("Average peak difference: %.3f ms", avg_peak_diff/samplerate_hz*1000)
        
        # Resample signal to correct timing
        resample_ratio = expected_peak_diff / avg_peak_diff
        logger.info("Resampling ratio: %.6f", resample_ratio)
        
        proc_samples = signal.resample_poly(proc_samples, int(resample_ratio*1000), 1000)
        logger.info("Resampled signal length: %d samples", len(proc_samples))
        
        proc_samples = proc_samples[0:1270*(96+4)]
        logger.info("Truncated signal to %d samples", len(proc_samples))
        
        # Plot correlation and peaks
        if enable_plotting:
            plt.figure(figsize=(12, 6))
            plt.subplot(2, 1, 1)
            plt.plot(abs(correlation))
            plt.plot(peaks, abs(correlation)[peaks], 'rx', markersize=10, label='Detected Peaks')
            plt.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold={threshold}')
            plt.title('Correlation with Preamble Pattern')
            plt.xlabel('Sample Index')
            plt.ylabel('Correlation Magnitude')
            plt.legend()
            plt.grid(True)
            
            plt.subplot(2, 1, 2)
            plt.plot(range(len(peak_diffs)), peak_diffs, 'bo-')
            plt.axhline(y=avg_peak_diff, color='r', linestyle='--', label=f'Average={avg_peak_diff:.1f}')
            plt.title('Peak Spacing')
            plt.xlabel('Peak Pair Index')
            plt.ylabel('Sample Difference')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            
    # Create full preamble modulated with goldcode at 10 samples per symbol
    from IoTBSConst import GCs, preamble
    sps = 10
    tag_id = 0  # Using first goldcode for now
    goldcode = GCs[tag_id]
    payload_bits = np.array([0,1,0,0,1,0,1,0,0,1,1,0,1,1,1,1])
    
    # Create preamble sequence: each bit modulated by goldcode, then upsampled
    preamble_gc_sps = np.array([])
    nbits = 16
    for i in range(nbits):
        bit = preamble[i]
        # bit = payload_bits[i]
    # for bit in preamble:
        # Convert bit (0/1) to BPSK (-1/+1), multiply by goldcode, then upsample
        bit_bpsk = 2 * bit - 1  # Convert 0->-1, 1->+1
        modulated_chip_seq = goldcode * bit_bpsk
        upsampled_seq = np.repeat(modulated_chip_seq.astype(np.complex64), sps)
        preamble_gc_sps = np.concatenate([preamble_gc_sps, upsampled_seq])
            
    correlation = np.correlate(proc_samples, preamble_gc_sps, mode='valid')
    
    # if (enable_plotting):
    # plt.figure()
    # plt.plot(abs(correlation))
    # # plt.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold={threshold}')
    # plt.title('Correlation with Preamble Pattern')
    # plt.xlabel('Sample Index')
    # plt.ylabel('Correlation Magnitude')
    # plt.legend()
    # plt.grid(True)
    
    # plt.show()

    
    
    return proc_samples
    

def mm_time_recovery(samples, samps_per_symbol):
    
    samples_interpolated = signal.resample_poly(samples, 16, 1)
    mu = 0 # initial estimate of phase of sample
    out = np.zeros(len(samples) + 10, dtype=np.complex64)
    out_rail = np.zeros(len(samples) + 10, dtype=np.complex64) # stores values, each iteration we need the previous 2 values plus current value
    i_in = 0 # input samples index
    i_out = 2 # output index (let first two outputs be 0)
    while i_out < len(samples) and i_in+16 < len(samples):
        # out[i_out] = samples[i_in] # grab what we think is the "best" sample
        out[i_out] = samples_interpolated[i_in*16 + int(mu*16)]
        out_rail[i_out] = int(np.real(out[i_out]) > 0) + 1j*int(np.imag(out[i_out]) > 0)
        x = (out_rail[i_out] - out_rail[i_out-2]) * np.conj(out[i_out-1])
        y = (out[i_out] - out[i_out-2]) * np.conj(out_rail[i_out-1])
        mm_val = np.real(y - x)
        mu += samps_per_symbol + 0.3*mm_val
        i_in += int(np.floor(mu)) # round down to nearest int since we are using it as an index
        mu = mu - np.floor(mu) # remove the integer part of mu
        i_out += 1 # increment output index
    out = out[2:i_out] # remove the first two, and anything after i_out (that was never filled out)
    
    return(out)

# # course frequency correction handled in time_freq_code search
# def course_f_correct(samples, samp_rate):
#     samples_sqr = samples**2
#     psd_sq = np.fft.fftshift(np.abs(np.fft.fft(samples_sqr)))
#     f = np.linspace(-samp_rate/2.0, samp_rate/2.0, len(psd_sq))
#     max_freq = f[np.argmax(psd_sq)]
#     Ts = 1/samp_rate # calc sample period
#     # t = np.arange(0, Ts*len(samples), Ts) # create time vector
#     t = np.linspace(0,(len(samples)-1)/samp_rate, len(samples))
#     logger.info("test 11")
#     samples = samples * np.exp(-1j*2*np.pi*max_freq*t/2.0)
    
#     logger.info("coarse freq offset: %s", max_freq)
    
#     return(samples.astype(np.complex64) )

def costas_loop(samples, samp_rate):
    
    N = len(samples)
    phase = 0
    freq = 0
    # These next two params is what to adjust, to make the feedback loop faster or slower (which impacts stability)
    alpha = 0.132
    beta = 0.00932
    out = np.zeros(N, dtype=np.complex64)
    freq_log = []
    for i in range(N):
        out[i] = samples[i] * np.exp(-1j*phase) # adjust the input sample by the inverse of the estimated phase offset
        error = np.real(out[i]) * np.imag(out[i]) # This is the error formula for 2nd order Costas Loop (e.g. for BPSK)

        # Advance the loop (recalc phase and freq offset)
        freq += (beta * error)
        freq_log.append(freq * samp_rate / (2*np.pi)) # convert from angular velocity to Hz for logging
        phase += freq + (alpha * error)

        # Optional: Adjust phase so its always between 0 and 2pi, recall that phase wraps around every 2pi
        while phase >= 2*np.pi:
            phase -= 2*np.pi
        while phase < 0:
            phase += 2*np.pi
    
    # plt.figure()
    # plt.plot(freq_log)
    # plt.title("Frequency offset from costas loop")
    # plt.grid("on")
            
    return(out)

def demod_bpsk(samples):
            
    nbits = len(samples)
    bits = np.zeros(nbits)
    for i in range(nbits):
        bits[i] = int(np.real(samples[i]) > 0)
        
    return(bits.astype(int))   

def sync_word_search_correction(in_bits, sync_seq, payload_len):
    
    in_bits_bip = in_bits*2-1
    sync_seq_bip = sync_seq*2-1
    
    sync_corr = np.correlate(in_bits_bip, sync_seq_bip, mode='valid')
    sync_thresh = 0#len(sync_seq)-1
    peaks, _ = signal.find_peaks(abs(sync_corr), height=sync_thresh)
    
    # logger.info(f"peaks: {peaks}")
    
    # Get the first peak above threshold
    if len(peaks) > 0:
        first_peak_sample_num = peaks[np.argmax(abs(sync_corr[peaks]))]
        first_peak_value = sync_corr[first_peak_sample_num]
        first_peak_sign = np.sign(first_peak_value)
      
        logger.info("first sync peak-> sample number: %s, value: %s, sign: %s", first_peak_sample_num, first_peak_value, first_peak_sign)
    else:
        logger.warning("No peaks found!")
        first_peak_sample_num = 0
        first_peak_sign = 1
        
    out_all_bits = (in_bits_bip*first_peak_sign+1)/2
    out_payload_bits = out_all_bits[first_peak_sample_num+len(sync_seq):first_peak_sample_num+len(sync_seq)+payload_len]
    
    # plt.figure()
    # plt.plot(sync_corr)
    # plt.show()

    return out_all_bits, out_payload_bits

def demodulate_packet(input_samples, tag_params, radio_params, demod_settings, enable_plotting=False):
    logger.info("Processing %s samples generated by %s tags", len(input_samples), tag_params.ntags) 
    
    samplerate_hz = radio_params.get('samplerate_hz', {})
    target_sps = radio_params.get('target_sps', {})
    
    manual_freq_enable = demod_settings.get('manual_enable',{}).get('freq',{})
    manual_freq_val_hz = demod_settings.get('manual_value',{}).get('freq_hz',{})
    manual_resamp_enable = demod_settings.get('manual_enable',{}).get('resamp',{})
    manual_resamp_rate_val = demod_settings.get('manual_value',{}).get('resamp_rate',{})
    manual_sample_shift_enable = demod_settings.get('manual_enable',{}).get('start_index',{})
    manual_sample_shift_val = demod_settings.get('manual_value',{}).get('start_index',{})
    
    # logger.info(f"manual_freq_enable: {manual_freq_enable}")
    
    itagParams = tag_params.get_tag(0)
    nbits_sent = np.zeros(tag_params.ntags)
    nbits_error = np.zeros(tag_params.ntags)
    all_payloads = np.zeros((tag_params.ntags,16),dtype=np.uint8)
    
    max_corr_thresh_stage0 = 300
    
    # demod_info = parse_demod_settings(demod_settings)
    # print(demod_info)
    
    all_sigs_found = 0
    # start_ind = 0
    num_samps_stage0 = 127*80*10 #gc_len * sps * 10 to get ~10 goldcode symbols
    num_samps_stage1 = 1270*100 #127000*2
    # start_ind = 127000+124000
    if manual_sample_shift_enable:
        start_ind = manual_sample_shift_val
    else:
        start_ind = 254345-20000
    # num_samps = 1270*110
    
    
    while start_ind+num_samps_stage0 < len(input_samples) and all_sigs_found == 0:
        all_sigs_found = 1
        
        #pull subset of samples
        subset_samples = input_samples[start_ind:start_ind+num_samps_stage0]
    
        # Mix to center frequency
        center_freq_hz = -31e3
        time_array_sec = np.linspace(0,(len(subset_samples)-1)/samplerate_hz, len(subset_samples))
        mixed_samples = subset_samples*np.exp(-1j*2*np.pi*center_freq_hz*time_array_sec)
    
        logger.info("start_ind: %s, num_samps_stage0: %s, center_freq_hz: %s", start_ind, num_samps_stage0, center_freq_hz)
    
        # Low pass filter and automatic gain control
        filt_samples = lpf(mixed_samples, 2.5e3, samplerate_hz)
        agc_samples = agc(filt_samples, 1/np.sqrt(2))
    
        if enable_plotting:
            plot_time_psd_scat(subset_samples, samplerate_hz, "Raw Received Samples")
            plot_time_psd_scat(mixed_samples, samplerate_hz, "Mixed Samples")
            plot_time_psd_scat(agc_samples, samplerate_hz, "Filtered and AGCed Samples")
    
        #See if each tag's goldcode is present
        for itag in range(tag_params.ntags):
            
            logger.debug(f"Demodding tag {itag}") 
            itagParams = tag_params.get_tag(itag)
            gc_len = len(itagParams.goldcode)
            
            proc_samples = agc_samples 
            
            # Stage 1: Coarse frequency correction using Gold code correlation
            
            if manual_freq_enable == True:
                freq_search_array_hz = np.array([manual_freq_val_hz])
            else:
                max_input_freq_dev_hz = 1000
                max_output_freq_dev_hz = 100
                npts_freq_search = int(np.ceil(max_input_freq_dev_hz * 2 / max_output_freq_dev_hz) + 1)
                freq_search_array_hz = np.linspace(-1*max_input_freq_dev_hz,max_input_freq_dev_hz,npts_freq_search )
            
            resampled_gc = np.repeat(itagParams.goldcode.astype(np.complex64), target_sps)
            
            proc_samples, freq_correction_hz, max_corr = coarse_freq_correction_code_search(
                proc_samples, resampled_gc, samplerate_hz, 
                freq_search_array_hz, enable_plotting)
            
            if enable_plotting:
                plot_time_psd_scat(proc_samples, samplerate_hz, "After Coarse Freq Correction")
            
            #if a goldcode is present, pull more samples
            if max_corr > max_corr_thresh_stage0:
                logger.info(f"Signal found for tag {itag} at sample {start_ind}")
                
                proc_samples = input_samples[start_ind:start_ind+num_samps_stage1]
                
                # Mix to center frequency plus offset frequency
                mix_freq_hz = center_freq_hz - freq_correction_hz
                time_array_sec = np.linspace(0,(len(proc_samples)-1)/samplerate_hz, len(proc_samples))
                mixed_samples = proc_samples*np.exp(-1j*2*np.pi*mix_freq_hz*time_array_sec)
            
                logger.info("start_ind: %s, num_samps_stage0: %s, mix_freq_hz: %s", start_ind, num_samps_stage1, mix_freq_hz)
            
                # Low pass filter and automatic gain control
                filt_samples = lpf(mixed_samples, 25e3, samplerate_hz)
                proc_samples = agc(filt_samples, 1/np.sqrt(2))
                
                # #autocorrelate to see what it looks like
                
                # nsamps = len(proc_samples)
                # lags = np.arange(-nsamps+1, nsamps)
                # autocorr = np.correlate(proc_samples, proc_samples, mode="full")
                
                # # Find peaks above threshold
                # threshold = 5000
                # min_prominence = threshold
                # abs_autocorr = abs(autocorr)
                

                # # Find peaks in the absolute autocorrelation
                # peaks, properties = signal.find_peaks(abs_autocorr, height=threshold,prominence=min_prominence)

                # # Get peak values and corresponding lags
                # peak_values = abs_autocorr[peaks]
                # peak_lags = lags[peaks]

                # # Print peak information
                # logger.info(f"Found {len(peaks)} peaks above threshold {threshold}")
                # logger.info(f"peak_values: {peak_values}")
                # logger.info(f"peak_lags: {peak_lags}")
                
                # plt.figure()
                # plt.plot(lags, abs(autocorr))
                # plt.plot(lags,np.real(autocorr))
                # plt.plot(lags,np.imag(autocorr))
                # plt.grid("on")
                # plt.title("Autocorrelation after agc")
                # plt.legend(["abs","real","imag"])
                
                # # Plot peaks
                # plt.plot(peak_lags, peak_values, 'ro', markersize=8, label=f'Peaks > {threshold}')

                # # Add horizontal threshold line
                # plt.axhline(y=threshold, color='red', linestyle='--', alpha=0.7, label=f'Threshold = {threshold}')

                # # # Add annotations for peaks
                # # for i, (lag, value) in enumerate(zip(peak_lags, peak_values)):
                # #     plt.annotate(f'Peak {i+1}\n({lag}, {value:.0f})', 
                # # xy=(lag, value), 
                # # xytext=(10, 10), 
                # # textcoords='offset points',
                # # bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                # # arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

                
                
                
                #manually move to start index
                
                # #manually resample
                # manual_resamp_rate = 1.0075
                # proc_samples = signal.resample_poly(proc_samples, int(manual_resamp_rate*10000), 10000)
                
                #find correct resampling in multi-stage process
                
                # if manual_resamp_enable:
                #     npts_resamp_array = 1
                #     resamp_rate_array = np.array([manual_resamp_rate_val])
                # else:
                #     npts_resamp_array = 41
                #     resamp_rate_array = np.linspace(1.006, 1.007,npts_resamp_array)
                # peak_corr_resamp_rate = np.zeros(npts_resamp_array)
                
                # logger.info(f"resamp_rate_array: {resamp_rate_array}")
                
                # preamble_corr_seq = np.array([])
                # nbits = 16
                # for i in range(nbits):
                #     bit = itagParams.preamble_bits[i]
                #     bit_bpsk = 2 * bit - 1  # Convert 0->-1, 1->+1
                #     modulated_chip_seq = itagParams.goldcode * bit_bpsk
                #     upsampled_seq = np.repeat(modulated_chip_seq.astype(np.complex64), target_sps)
                #     preamble_corr_seq = np.concatenate([preamble_corr_seq, upsampled_seq])
                
                # for i,test_resamp_rate in enumerate(resamp_rate_array):
                #     resamp_samples = signal.resample_poly(proc_samples, int(test_resamp_rate*100000), 100000)
                
                #     # correlation = np.correlate(resamp_samples, resampled_gc, mode='valid')
                #     correlation = np.correlate(resamp_samples, preamble_corr_seq, mode='valid')

                #     # Get peak value of correlation magnitude
                #     peak_corr_resamp_rate[i] = np.max(np.abs(correlation))
                #     # logger.debug(f"cur_resamp_rate: {cur_resamp_rate}, Peak correlation value: {peak_value}, Peak index: {peak_index}")

                # logger.debug(f"max_corr_resamp_rate: {peak_corr_resamp_rate}")
                # max_peak_corr_resamp_rate = np.max(peak_corr_resamp_rate)
                # best_resamp_rate = resamp_rate_array[np.argmax(peak_corr_resamp_rate)]
                # logger.info(f"max_peak_corr_resamp_rate: {max_peak_corr_resamp_rate}, best_resamp_rate: {best_resamp_rate}")
                
                # #resample and find start index
                # proc_samples = signal.resample_poly(proc_samples, int(best_resamp_rate*10000), 10000)
                
                # # # logger.info
                
                # correlation = np.correlate(proc_samples, preamble_corr_seq, mode='valid')
                # # correlation = np.correlate(proc_samples, resampled_gc, mode='valid')
                
                
                # # Get peak value of correlation magnitude
                # preamble_peak_value = np.max(np.abs(correlation))
                # preamble_peak_index = np.argmax(np.abs(correlation))
                
                # if enable_plotting:
                #     plt.figure()
                #     plt.plot(np.abs(correlation))
                #     plt.plot(preamble_peak_index, preamble_peak_value, 'ro', markersize=8, label=f'Peak = {preamble_peak_value:.2f}')
                #     plt.title(f"Correlation with {nbits} bits at {best_resamp_rate} resamp rate- Peak Value: {preamble_peak_value:.2f}")
                #     plt.xlabel('Sample Index')
                #     plt.ylabel('Correlation Magnitude')
                #     plt.legend()
                #     plt.grid(True)
                    
                
                # # find peak within 80% of actual peak and that has no peak for 4 symbols before
                # height_threshold = .8
                # num_samps_to_check_prior = 127*10*2
                
                # peaks, _ = signal.find_peaks(np.abs(correlation), height=400)
                
                # if len(peaks) == 0:
                #     return None
                
                # # Get peak heights
                # peak_heights = np.abs(correlation[peaks])
                # max_peak_value = np.max(peak_heights)
                # threshold = height_threshold * max_peak_value
                
                # start_index = -1
                
                # # Check each peak
                
                # logger.debug(f"peaks[0:5]: {peaks[0:5]}")
                # for i, peak_idx in enumerate(peaks[0:5]):
                    
                #     peak_value = np.abs(correlation[peak_idx])
                    
                    
                #     # Check if peak is above threshold
                #     if peak_value >= threshold and peak_idx >= num_samps_to_check_prior:
                #         logger.debug(f"i, peak_idx, peak_value: {i}, {peak_idx}, {peak_value}")
                #         # Check if there are no peaks in the next m samples
                #         prev_m_samples_end = peak_idx - 1
                #         prev_m_samples_start = peak_idx - num_samps_to_check_prior
                        
                #         if prev_m_samples_start < prev_m_samples_end:
                #             # Find peaks in the next m samples
                #             remaining_peaks = peaks[peaks < peak_idx]  # All peaks before current
                #             peaks_in_prev_m = remaining_peaks[remaining_peaks > prev_m_samples_start]
                            
                #             logger.debug(f"peaks_in_prev_m: {peaks_in_prev_m}")
                            
                #             if len(peaks_in_prev_m) == 0:
                #                 start_index = peak_idx
                #         else:
                #             # If we're at the end of the array and no more samples to check
                #             start_index = -1
                    
                # logger.info(f"start_index: {start_index}")
                
                # # plt.show()
                # #Move to the start and truncate
                # if start_index != -1:
                #     proc_samples = proc_samples[start_index:start_index+127*(96+20)*10]
                    
                #     # logger.info(f"preamble_peak_value: {preamble_peak_value}, preamble_peak_index: {preamble_peak_index}")
                    
                #     #Move to the start and truncate
                #     proc_samples = proc_samples[preamble_peak_index:preamble_peak_index+127*(96+4)*10]
                #     # # proc_samples = proc_samples[0:127*(96+4)*10]
                    
                #     # if enable_plotting:
                #     #     plot_time_psd_scat(proc_samples, samplerate_hz, "Resampled and Rolled")
                    
                #     # despread
                #     # bits_to_calc = len(itagParams.actual_bits)
                #     bits_to_calc = len(itagParams.all_bits)+1
                #     logger.debug("Expected number of bits: %s", bits_to_calc)
                #     samps_per_bit = target_sps*len(itagParams.goldcode)
                #     despread_samples_repeat = np.zeros(samps_per_bit*bits_to_calc).astype(np.complex64)
                    
                #     logger.info(f"samps_per_bit: {samps_per_bit}")
                            
                #     for bit in range(bits_to_calc):
                #         # print(bit)
                #         despread_samples_repeat[bit*samps_per_bit:(bit+1)*samps_per_bit] = proc_samples[bit*samps_per_bit:(bit+1)*samps_per_bit]*resampled_gc#np.repeat(itagParams.goldcode.astype(np.complex64),target_sps)

                #     # if enable_plotting:
                #     #     plot_time_psd_scat(despread_samples_repeat, samplerate_hz, "Despread Samples w/ Repeat")
                    
                #     proc_samples = mm_time_recovery(despread_samples_repeat, target_sps)
                #     # if enable_plotting:
                #     #     plot_time_psd_scat(proc_samples, samplerate_hz, "MM Timing recovery")
                        
                #     #costas loop
                #     proc_samples = costas_loop(proc_samples, samplerate_hz)
                #     # if enable_plotting:
                #     #     plot_time_psd_scat(proc_samples, samplerate_hz, "Costas loop")
                    
                    
                #     #bpsk demod
                #     rx_bits_raw = demod_bpsk(proc_samples)
                #     # if enable_plotting:
                #     #     plot_time_psd_scat(proc_samples, samplerate_hz, "Raw demodded bits")
                #     # logger.info("Raw received bits: %s",rx_bits_raw.tolist())
                
                    
                #     # logger.debug("rx_bits_raw: %s", rx_bits_raw)
                    
                #     #average over cdma symbol - create array filled with NaNs
                #     nbits_data = len(itagParams.all_bits)
                #     proc_bits = np.full(nbits_data, np.nan)
                    
                #     # Bit decision logic: majority voting over gc_len chunks 
                #     for i in range(nbits_data):
                #         start_idx = i * gc_len
                #         end_idx = (i + 1) * gc_len-1
                #         if end_idx <= len(rx_bits_raw):
                #             chunk = rx_bits_raw[start_idx:end_idx]
                #             # Count 1s and 0s in the chunk
                #             ones_count = np.sum(chunk)
                #             zeros_count = len(chunk) - ones_count
                #             # Decide based on majority
                #             proc_bits[i] = 1 if ones_count > zeros_count else 0
                    
                #     valid_bits = proc_bits[~np.isnan(proc_bits)]
                            
                #     # logger.info("proc_bits:%s",proc_bits)
                #     corrected_proc_bits, rx_payload_bits = sync_word_search_correction(valid_bits, itagParams.sync_bits, len(itagParams.payload_bits))
   
                
                
                
        # #correlation to find if signal is present and, if so, what the offset freq is
        # # max_input_freq_dev_hz = 1000
        # # max_output_freq_dev_hz = 100
        # # npts_freq_search = int(np.ceil(max_input_freq_dev_hz *2/max_output_freq_dev_hz)+1)
        # resampled_gc = np.repeat(itagParams.goldcode.astype(np.complex64) , target_sps)
        # # proc_samples = time_freq_code_search(proc_samples, resampled_gc, target_sps, samplerate_hz, max_input_freq_dev_hz, npts_freq_search, enable_plotting)
        # proc_samples = time_code_resamp_search(proc_samples, resampled_gc, target_sps, samplerate_hz, enable_plotting)

        # # despread
        # # bits_to_calc = len(itagParams.actual_bits)
        # bits_to_calc = len(itagParams.all_bits)+4
        # logger.debug("Expected number of bits: %s", bits_to_calc)
        # samps_per_bit = target_sps*len(itagParams.goldcode)
        # despread_samples_repeat = np.zeros(samps_per_bit*bits_to_calc).astype(np.complex64)
                
        # for bit in range(bits_to_calc):
        #     despread_samples_repeat[bit*samps_per_bit:(bit+1)*samps_per_bit] = proc_samples[bit*samps_per_bit:(bit+1)*samps_per_bit]*np.repeat(itagParams.goldcode.astype(np.complex64),target_sps)

        # if enable_plotting:
        #     plot_time_psd_scat(despread_samples_repeat, samplerate_hz, "Despread Samples w/ Repeat")
        
        # despread_samples_interp = np.zeros(samps_per_bit*bits_to_calc).astype(np.complex64)
        
        # for bit in range(bits_to_calc):
        #     despread_samples_interp[bit*samps_per_bit:(bit+1)*samps_per_bit] = proc_samples[bit*samps_per_bit:(bit+1)*samps_per_bit]*signal.resample_poly(itagParams.goldcode.astype(np.complex64),target_sps,1)

        # if enable_plotting:
        #     plot_time_psd_scat(despread_samples_interp, samplerate_hz, "Despread Samples w/ Interp")

        # #mm time recovery
        # proc_samples = mm_time_recovery(despread_samples_repeat, target_sps)
        # if enable_plotting:
        #     plot_time_psd_scat(proc_samples, samplerate_hz, "MM Timing recovery")
      
        # # # coarse f correct- should not be needed since we do this in the time_freq_code_search
        # # proc_samples = course_f_correct(proc_samples, samplerate_hz)
        # # plt.figure()
        # # plt.plot(np.real(proc_samples))
        # # plt.title("After course f correct")
        # # plt.grid("on")
              
        # #costas loop
        # proc_samples = costas_loop(proc_samples, samplerate_hz)
        # if enable_plotting:
        #     plot_time_psd_scat(proc_samples, samplerate_hz, "Costas loop")
        # # plt.figure()
        # # plt.plot(proc_samples)
        # # plt.title("After Costas Loop")
        # # plt.grid("on")
        
        # #bpsk demod
        # rx_bits_raw = demod_bpsk(proc_samples)
        # if enable_plotting:
        #     plot_time_psd_scat(proc_samples, samplerate_hz, "Raw demodded bits")
        # # logger.info("Raw received bits: %s",rx_bits_raw.tolist())
        # # plt.figure()
        # # plt.plot(rx_bits_raw)
        # # plt.title("rx_bits_raw")
        # # plt.grid("on")
        
        # logger.debug("rx_bits_raw: %s", rx_bits_raw)
        
        # #average over cdma symbol - create array filled with NaNs
        # nbits_data = len(itagParams.all_bits)
        # proc_bits = np.full(nbits_data, np.nan)
        
        # # Bit decision logic: majority voting over gc_len chunks 
        # for i in range(nbits_data):
        #     start_idx = i * gc_len
        #     end_idx = (i + 1) * gc_len-1
        #     if end_idx <= len(rx_bits_raw):
        #         chunk = rx_bits_raw[start_idx:end_idx]
        #         # Count 1s and 0s in the chunk
        #         ones_count = np.sum(chunk)
        #         zeros_count = len(chunk) - ones_count
        #         # Decide based on majority
        #         proc_bits[i] = 1 if ones_count > zeros_count else 0
        
        # valid_bits = proc_bits[~np.isnan(proc_bits)]
                
        # # logger.info("proc_bits:%s",proc_bits)
        # corrected_proc_bits, rx_payload_bits = sync_word_search_correction(valid_bits, itagParams.sync_bits, len(itagParams.payload_bits))
   
        
        # # Stage 1: Coarse frequency correction using Gold code correlation
        # max_input_freq_dev_hz = 1000
        # max_output_freq_dev_hz = 100
        # npts_freq_search = int(np.ceil(max_input_freq_dev_hz * 2 / max_output_freq_dev_hz) + 1)
        # resampled_gc = np.repeat(itagParams.goldcode.astype(np.complex64), target_sps)
        
        # proc_samples, freq_correction = coarse_freq_correction(
        #     proc_samples, resampled_gc, samplerate_hz, 
        #     max_input_freq_dev_hz, npts_freq_search, enable_plotting)
        
        # if enable_plotting:
        #     plot_time_psd_scat(proc_samples, samplerate_hz, "After Coarse Freq Correction")
            
        # ## test stage: How to recover the start index
        
        # test_resamp_rate = np.array([1,1.0064])
        # # test_resamp_rate = np.array([0.999, 1, 1.001])
        # # test_resamp_rate = np.linspace(0.95,1.05,101)
        # # test_resamp_rate = np.linspace(1.006,1.007,11)
        
        # corr_seq = np.array([])
        # nbits = 64
        # for i in range(nbits):
        #     bit = itagParams.preamble_bits[i]
        #     bit_bpsk = 2 * bit - 1  # Convert 0->-1, 1->+1
        #     modulated_chip_seq = itagParams.goldcode * bit_bpsk
        #     upsampled_seq = np.repeat(modulated_chip_seq.astype(np.complex64), target_sps)
        #     corr_seq = np.concatenate([corr_seq, upsampled_seq])
        
        # for cur_resamp_rate in test_resamp_rate:
            
        #     #resample
        #     resamp_samples = signal.resample_poly(proc_samples, int(cur_resamp_rate*10000), 10000)
            
        #     #correlate with preamble
        #     correlation = np.correlate(resamp_samples, corr_seq, mode='valid')

        #     # Get peak value of correlation magnitude
        #     peak_value = np.max(np.abs(correlation))
        #     peak_index = np.argmax(np.abs(correlation))
        #     logger.debug(f"cur_resamp_rate: {cur_resamp_rate}, Peak correlation value: {peak_value}, Peak index: {peak_index}")
        #     # print(f"Peak correlation value: {peak_value}")
        #     # print(f"Peak index: {peak_index}")

        #     # Plot the peak point
        #     plt.figure()
        #     plt.plot(np.abs(correlation))
        #     plt.plot(peak_index, peak_value, 'ro', markersize=8, label=f'Peak = {peak_value:.2f}')
        #     plt.title(f"Correlation with {nbits} bits at {cur_resamp_rate} resamp rate- Peak Value: {peak_value:.2f}")
        #     plt.xlabel('Sample Index')
        #     plt.ylabel('Correlation Magnitude')
        #     plt.legend()
        #     plt.grid(True)
            
        
        # # Stage 2: Robust timing recovery (coarse + fine)
        # proc_samples, timing_sign = robust_timing_recovery(
        #     proc_samples, itagParams.goldcode, target_sps, 
        #     samplerate_hz, enable_plotting)
        
        # logger.info(f"timing sign: {timing_sign}")
        
        # if enable_plotting:
        #     plot_time_psd_scat(proc_samples, samplerate_hz, "After Timing Recovery")
        
        # # Stage 3: EARLY DESPREADING with multiple hypotheses
        # despread_samples, best_timing_offset = multi_hypothesis_despread(
        #     proc_samples, itagParams.goldcode, target_sps, num_hypotheses=8)
        # logger.info(f"best_timing_offset: {best_timing_offset}")
        
        # if enable_plotting:
        #     plot_time_psd_scat(despread_samples, samplerate_hz/target_sps/gc_len, 
        #                      "After Early Despreading")
            
        # # Stage 4: Fine timing recovery on despread signal
        # # Now working with symbol-rate samples after despreading
        # symbol_rate_sps = 4  # Samples per symbol after despreading
        # despread_samples_upsampled = signal.resample_poly(despread_samples, symbol_rate_sps, 1)
        
        # proc_samples = mm_time_recovery(despread_samples_upsampled, symbol_rate_sps)
        # if enable_plotting:
        #     plot_time_psd_scat(proc_samples, samplerate_hz/gc_len, "After Fine MM Timing Recovery")
        
        # # Stage 5: Enhanced Costas loop for fine frequency correction
        # proc_samples = enhanced_costas_loop(proc_samples, samplerate_hz/gc_len)
        # if enable_plotting:
        #     plot_time_psd_scat(proc_samples, samplerate_hz/gc_len, "After Enhanced Costas Loop")
        
        # # Stage 6: BPSK demodulation
        # rx_bits_raw = demod_bpsk(proc_samples)
        # logger.debug("rx_bits_raw length: %d", len(rx_bits_raw))
        
        # # Stage 7: Sync word search and bit correction
        # nbits_data = len(itagParams.all_bits)
        
        # # Apply timing sign correction if needed
        # if timing_sign < 0:
        #     rx_bits_raw = 1 - rx_bits_raw
        
        # corrected_proc_bits, rx_payload_bits = sync_word_search_correction(
        #     rx_bits_raw, itagParams.sync_bits, len(itagParams.payload_bits))
        
        # # common to both pipelines
        # logger.info("Transmitted actual bits:  %s", itagParams.all_bits)
        # logger.info("Corrected processed bits: %s", corrected_proc_bits.astype(int)) 
        # logger.info("Raw processed bits:       %s", rx_bits_raw[:len(itagParams.all_bits)].astype(int)) 
        
        # logger.info("Transmitted payload bits:  %s", itagParams.payload_bits)
        # logger.info("Received payload bits:     %s", rx_payload_bits.astype(int))
        
        # all_payloads[itag] = rx_payload_bits.astype(int)
        
        nbits_sent[itag] = len(itagParams.payload_bits)
        try:
            nbits_error[itag] = sum(abs(itagParams.payload_bits-rx_payload_bits.astype(int)))
        except:
            nbits_error[itag] =len(itagParams.payload_bits)
        
        BER = nbits_error[itag]/nbits_sent[itag]
        logger.debug(f"BER: {BER}")
        
    # logger.info(f"nbits_error: {nbits_error}")
    return all_payloads, nbits_error, nbits_sent