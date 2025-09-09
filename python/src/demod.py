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
    # transition_bw = data_rate*0.5
    b, a = signal.butter(5, cutoff_freq, 'low', fs=samp_rate)
    out_samples = signal.filtfilt(b, a, samples)
    
    return(out_samples)

def agc(samples,out_amplitude):
    
    mag = np.sum(np.abs(samples)**2)/len(samples)
    gain = out_amplitude/np.sqrt(mag)
    samples_out = samples*gain
    
    return(samples_out) 

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
        # logger.info("Average peak difference: %.3f ms", avg_peak_diff/radio_params.samplerate_hz*1000)
        
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


def time_code_resamp_search(input_samples, target_corr_samples, input_sps, fs_hz, max_freq_dev_hz, nfreq_pts : int, enable_plotting=False):
    
    proc_samples = input_samples
    
    #must get the frequency to within ~150 Hz for the costas loop to be able to correct it
    
    # freq_offsets_hz = np.linspace(-1*max_freq_dev_hz, max_freq_dev_hz, nfreq_pts)#[-100,-50,0,50,100]
    # logger.debug("Test frequency offsets: %s", freq_offsets_hz)
    # # freq_offsets_hz = [0]
    # time_delays_sec = np.linspace(0, (len(input_samples)-1)/fs_hz, len(input_samples))
    
    # # target_gc_stretch = np.repeat(target_gc,input_sps)
    
    # # logger.info(f"len(samples): {len(samples)}, len(target_gc_stretch): {len(target_gc_stretch)}, dif: {len(samples)-len(target_gc_stretch)}")
    # corr_len = len(input_samples)-len(target_corr_samples)+1
    # # logger.info(f"corr_len: {corr_len}")
    
    # # Create meshgrid for time delay and frequency offset
    # X, Y = np.meshgrid(time_delays_sec[0:corr_len], freq_offsets_hz)
    # Z = np.zeros((len(freq_offsets_hz), corr_len))
    
    # for j, freq_offset_hz in enumerate(freq_offsets_hz):
    #     # logger.info(f"Loop {j}, freq_offset_hz: {freq_offset_hz}")
    #     test_samples = input_samples * np.exp(+1j*2*np.pi*freq_offset_hz*time_delays_sec)
        
    #     correlation = np.correlate(test_samples, target_corr_samples, mode='valid')
    #     Z[j, :] = np.abs(correlation)
    
    #     if enable_plotting:
    #         plt.figure()
    #         plt.plot(abs(correlation))
    #         plt.grid("on")
    #         plt.title(f"Correlation with Repeated Goldcode, freq_offset_hz: {freq_offset_hz}")
    
    # if enable_plotting:
    #     fig_lab = plt.figure()
    #     ax = fig_lab.add_subplot(111, projection='3d')
    #     ax.plot_surface(X, Y, Z[:,:], cmap='viridis')
    #     ax.set_title(f'Correlation with Repeated Goldcode')

    #     # Set labels
    #     ax.set_xlabel('Time Delay (s)')
    #     ax.set_ylabel('Frequency Offset (Hz)')
    #     ax.set_zlabel('Signal Strength')
    
    # max_corr = np.max(Z)
    # max_corr_idx = np.argmax(Z)
    # max_corr_idx = np.unravel_index(max_corr_idx, Z.shape)
    # freq_adj_hz = freq_offsets_hz[max_corr_idx[0]]

    # logger.info(f"max_corr: {max_corr}")
    # logger.info(f"freq index of max correlation: {max_corr_idx[0]}, corresponding freq_adj_hz: {freq_adj_hz}")
    # # logger.info(f"time index of max correlation: {max_corr_idx[1]}")
    
    # logger.info("freq_offsets_hz: %s", freq_offsets_hz)
    # # logger.info("time_delays_sec[0:5]: %s", time_delays_sec[0:5])
    
    # output_samples = input_samples * np.exp(1j*2*np.pi*freq_adj_hz*time_delays_sec)
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
        # logger.info("Average peak difference: %.3f ms", avg_peak_diff/radio_params.samplerate_hz*1000)
        
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
    plt.figure()
    plt.plot(abs(correlation))
    # plt.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold={threshold}')
    plt.title('Correlation with Preamble Pattern')
    plt.xlabel('Sample Index')
    plt.ylabel('Correlation Magnitude')
    plt.legend()
    plt.grid(True)
    
    plt.show()

    
    
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

# course frequency correction handled in time_freq_code search
def course_f_correct(samples, samp_rate):
    samples_sqr = samples**2
    psd_sq = np.fft.fftshift(np.abs(np.fft.fft(samples_sqr)))
    f = np.linspace(-samp_rate/2.0, samp_rate/2.0, len(psd_sq))
    max_freq = f[np.argmax(psd_sq)]
    Ts = 1/samp_rate # calc sample period
    # t = np.arange(0, Ts*len(samples), Ts) # create time vector
    t = np.linspace(0,(len(samples)-1)/samp_rate, len(samples))
    logger.info("test 11")
    samples = samples * np.exp(-1j*2*np.pi*max_freq*t/2.0)
    
    logger.info("coarse freq offset: %s", max_freq)
    
    return(samples.astype(np.complex64) )

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

# def sync_word_sync(bits_in, s_word, nbits_out, samps_per_bit):
        
#     long_sync_word = np.repeat(s_word,samps_per_bit)
#     sync_word_corr = np.correlate(bits_in, long_sync_word, mode='valid')
    
#     plt.figure()
#     plt.plot(sync_word_corr)
    
#     offset = int(np.argmax(np.abs(sync_word_corr))) #the filter introduces a 6 bit delay
#     logger.info("offset: %s", offset)
#     logger.info("value of correlation at offset: %s", sync_word_corr[offset])
#     sign = np.sign(sync_word_corr[offset])
#     bits_out = bits_in[offset+len(s_word):]

#     bits_out = (sign*bits_out[:nbits_out]+1)/2

#     return(bits_out, sync_word_corr[offset])

def demodulate_packet(input_samples, tag_params, radio_params, enable_plotting=False):
    logger.info("Processing %s samples generated by %s tags", len(input_samples), tag_params.ntags) 
    
    
    start_ind = 127000+124000
    num_samps = 1270*100
    
    subset_samples = input_samples[start_ind:start_ind+num_samps]
    
    #mix to center at -50kHz
    offset_freq_hz = 505
    center_freq_hz = -50e3+offset_freq_hz
    # center_freq_hz = 0
    time_array_sec = np.linspace(0,(len(subset_samples)-1)/radio_params.samplerate_hz, len(subset_samples))
    mixed_samples = subset_samples*np.exp(-1j*2*np.pi*center_freq_hz*time_array_sec)
    
    logger.info("start_ind: %s, num_samps: %s, offset_freq_hz: %s", start_ind, num_samps, offset_freq_hz)
    # logger.info("time_array_sec[0:5]: %s", time_array_sec[0:5])

    
    #low pass filter
    filt_samples = lpf(mixed_samples, 25e3, radio_params.samplerate_hz)

        
    # if enable_plotting:
    #     plot_time_psd_scat(subset_samples, radio_params.samplerate_hz, "Raw Received Samples")
    #     plot_time_psd_scat(mixed_samples, radio_params.samplerate_hz, "Mixed Samples")
    #     plot_time_psd_scat(filt_samples, radio_params.samplerate_hz, "Filtered Samples")
    
    for itag in range(tag_params.ntags):
        logger.info("Demodding tag %s", itag)
        itagParams = tag_params.get_tag(itag)
        proc_samples = filt_samples
        
        #agc
        proc_samples = agc(proc_samples, 1/np.sqrt(2))
        # if enable_plotting:
        #     plot_time_psd_scat(proc_samples, radio_params.samplerate_hz, "Filtered & AGCed Samples")
        
        #correlation to find if signal is present and, if so, what the offset freq is
        max_input_freq_dev_hz = 1000
        max_output_freq_dev_hz = 100
        npts_freq_search = int(np.ceil(max_input_freq_dev_hz *2/max_output_freq_dev_hz)+1)
        resampled_gc = np.repeat(itagParams.goldcode.astype(np.complex64) , itagParams.sps)
        # proc_samples = time_freq_code_search(proc_samples, resampled_gc, itagParams.sps, radio_params.samplerate_hz, max_input_freq_dev_hz, npts_freq_search, enable_plotting)
        proc_samples = time_code_resamp_search(proc_samples, resampled_gc, itagParams.sps, radio_params.samplerate_hz, max_input_freq_dev_hz, npts_freq_search, enable_plotting)


        # despread
        # bits_to_calc = len(itagParams.actual_bits)
        bits_to_calc = len(itagParams.actual_bits)+4
        logger.debug("Expected number of bits: %s", bits_to_calc)
        samps_per_bit = itagParams.sps*len(itagParams.goldcode)
        despread_samples_repeat = np.zeros(samps_per_bit*bits_to_calc).astype(np.complex64)
                
        for bit in range(bits_to_calc):
            despread_samples_repeat[bit*samps_per_bit:(bit+1)*samps_per_bit] = proc_samples[bit*samps_per_bit:(bit+1)*samps_per_bit]*np.repeat(itagParams.goldcode.astype(np.complex64),itagParams.sps)

        if enable_plotting:
            plot_time_psd_scat(despread_samples_repeat, radio_params.samplerate_hz, "Despread Samples w/ Repeat")
        
        despread_samples_interp = np.zeros(samps_per_bit*bits_to_calc).astype(np.complex64)
        
        for bit in range(bits_to_calc):
            despread_samples_interp[bit*samps_per_bit:(bit+1)*samps_per_bit] = proc_samples[bit*samps_per_bit:(bit+1)*samps_per_bit]*signal.resample_poly(itagParams.goldcode.astype(np.complex64),itagParams.sps,1)

        if enable_plotting:
            plot_time_psd_scat(despread_samples_interp, radio_params.samplerate_hz, "Despread Samples w/ Interp")

        #mm time recovery
        proc_samples = mm_time_recovery(despread_samples_repeat, itagParams.sps)
        if enable_plotting:
            plot_time_psd_scat(proc_samples, radio_params.samplerate_hz, "MM Timing recovery")
      
        # # coarse f correct- should not be needed since we do this in the time_freq_code_search
        # proc_samples = course_f_correct(proc_samples, radio_params.samplerate_hz)
        # plt.figure()
        # plt.plot(np.real(proc_samples))
        # plt.title("After course f correct")
        # plt.grid("on")
              
        #costas loop
        proc_samples = costas_loop(proc_samples, radio_params.samplerate_hz)
        if enable_plotting:
            plot_time_psd_scat(proc_samples, radio_params.samplerate_hz, "Costas loop")
        # plt.figure()
        # plt.plot(proc_samples)
        # plt.title("After Costas Loop")
        # plt.grid("on")
        
        #bpsk demod
        rx_bits_raw = demod_bpsk(proc_samples)
        if enable_plotting:
            plot_time_psd_scat(proc_samples, radio_params.samplerate_hz, "Raw demodded bits")
        # logger.info("Raw received bits: %s",rx_bits_raw.tolist())
        # plt.figure()
        # plt.plot(rx_bits_raw)
        # plt.title("rx_bits_raw")
        # plt.grid("on")
        
        logger.debug("rx_bits_raw: %s", rx_bits_raw)
        
        #average over cdma symbol - create array filled with NaNs
        nbits_data = len(itagParams.actual_bits)
        proc_bits = np.full(nbits_data, np.nan)
        
        # Bit decision logic: majority voting over gc_len chunks 
        for i in range(nbits_data):
            start_idx = i * gc_len
            end_idx = (i + 1) * gc_len-1
            if end_idx <= len(rx_bits_raw):
                chunk = rx_bits_raw[start_idx:end_idx]
                # Count 1s and 0s in the chunk
                ones_count = np.sum(chunk)
                zeros_count = len(chunk) - ones_count
                # Decide based on majority
                proc_bits[i] = 1 if ones_count > zeros_count else 0
        
        valid_bits = proc_bits[~np.isnan(proc_bits)]
                
        # logger.info("proc_bits:%s",proc_bits)
        corrected_proc_bits, rx_payload_bits = sync_word_search_correction(valid_bits, sync_seq, bitsPerPacket)
        
        # rx_bits_payload, sync_word_corr = sync_word_sync(rx_bits_raw*2-1, tag_params.sync_bits*2-1, len(tag_params.payload_bits), samps_per_bit)
        # logger.info("Sync word correlation: %s", sync_word_corr)
   
        logger.info("Transmitted actual bits:  %s", itagParams.actual_bits)
        logger.info("Corrected processed bits: %s", corrected_proc_bits.astype(int)) 
        logger.info("Raw processed bits:       %s", valid_bits.astype(int)) 
        
        logger.info("Transmitted payload bits:  %s", itagParams.payload_bits)
        logger.info("Received payload bits:     %s", rx_payload_bits.astype(int))
        
        
        
        num_bits = len(itagParams.payload_bits)
        #in case the length of payload_bits is not the same as the itagParams 
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