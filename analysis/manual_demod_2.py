#Search for goldcodes and demodulate the bits if a goldcode is present

# pseudocode:
# - select a small number of samples (4 goldcodes worth, shifting by 2 goldcodes every time)
# - mix, filter, and AGC these samples
# - search for a high goldcode correlation at different frequency offsets for each goldcode
# - if there are peaks when previous samples did not have peaks, continue, othewise grab the next samples from the top
# - grab a larger number of samples (80+10 goldcodes)
# - mix, filter, AGC, coarse frequency adjust, and SPS adjust
# - find and move to signal start with preamble correlation
# - despread
# - mm timing, costas loop, and bpsk demod

#imports
import logging
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import scipy.signal as signal

#constants
GCs = np.array([
[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, 1, 1, -1, -1, -1, 1, 1, -1, 1, 1, 1, 1, 1, 1, -1, 1, 1, -1, -1, -1, -1, 1, 1, -1, -1, 1, -1, 1, 1, -1, -1, 1, -1, -1, 1, -1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1, 1, 1, -1, -1, -1, 1, -1, 1, -1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, -1, 1, 1, -1, 1, -1, -1, 1, 1, 1, 1, 1, -1, 1, -1, 1, -1, -1, 1, 1, -1, 1, -1, -1, 1, 1, 1, 1, 1, 1, -1, -1, 1, -1, 1, 1, -1],
[1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, 1, -1, -1, 1, -1, 1, 1, 1, -1, -1, -1, 1, 1, -1, -1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, -1, -1, -1, 1, 1, -1, 1, 1, 1, -1, 1, -1, -1, 1, -1, 1, 1, 1, -1, -1, -1, 1, -1, -1, -1, 1, 1, -1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1, -1, -1, 1, 1, -1, -1, -1, 1, -1, -1, -1, 1, 1, 1, -1, -1, 1, 1, 1, -1, 1, -1, 1, -1, -1, -1, 1, 1, 1, 1, 1, -1, -1, -1, -1, 1, -1, 1, -1, 1, 1, 1, -1, 1, -1, -1],
[1, 1, -1, -1, -1, -1, -1, 1, 1, -1, -1, -1, -1, 1, 1, 1, -1, -1, -1, -1, -1, 1, 1, -1, 1, -1, -1, 1, 1, -1, 1, -1, -1, -1, 1, -1, 1, 1, -1, 1, 1, -1, -1, -1, -1, 1, -1, -1, 1, -1, -1, -1, -1, 1, -1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, -1, 1, -1, 1, -1, -1, 1, -1, -1, 1, 1, 1, 1, -1, 1, -1, 1, -1, -1, -1, 1, 1, -1, -1, 1, -1, -1, 1, -1, -1, -1, 1, 1, 1, -1, -1, -1, 1, -1, -1, 1, 1, -1, 1, 1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, -1, 1, -1, 1],
[-1, 1, 1, -1, -1, -1, -1, 1, 1, 1, -1, -1, -1, 1, -1, 1, 1, -1, -1, 1, -1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, -1, 1, 1, -1, 1, 1, 1, -1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, -1, 1, -1, 1, -1, 1, -1, 1, 1, 1, -1, -1, -1, -1, -1, 1, -1, -1, -1, 1, -1, -1, -1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, -1, -1, -1, 1, -1, 1, 1, 1, -1, -1, 1, 1, -1, -1, 1, -1, -1, -1, 1, -1, 1, -1, -1, 1, 1, 1, -1, 1, -1, 1, 1, 1, 1, -1, 1],
[-1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, -1, -1, 1, -1, -1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, 1, 1, -1, -1, -1, 1, -1, 1, 1, -1, -1, -1, 1, -1, 1, 1, 1, -1, 1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, 1, 1, -1, -1, 1, 1, 1, 1, 1, -1, -1, -1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, 1, 1, -1, 1, 1, 1, 1, 1, 1, -1, -1, 1, -1, -1, -1, 1, -1, 1, -1, -1, -1, 1, 1, -1, 1, 1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1],
[-1, -1, -1, 1, 1, -1, -1, 1, 1, 1, 1, 1, -1, 1, -1, -1, -1, 1, 1, 1, 1, -1, 1, 1, 1, -1, -1, -1, 1, -1, -1, -1, 1, -1, -1, 1, 1, 1, -1, 1, -1, 1, 1, -1, -1, -1, 1, -1, -1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, 1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, -1, 1, 1, -1, 1, -1, 1, -1, 1, 1, 1, -1, 1, 1, -1, 1, 1, 1, -1, 1, -1, 1, 1, -1, -1, 1, -1, 1, -1, 1, -1, 1, 1, -1, 1, 1, 1, 1, 1, 1, -1, -1, -1, 1, -1, -1, 1, 1, 1, 1],
[-1, -1, -1, -1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, 1, -1, 1, -1, -1, 1, 1, -1, 1, 1, 1, 1, -1, 1, -1, 1, -1, 1, 1, -1, -1, -1, 1, 1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, 1, 1, -1, -1, -1, 1, -1, -1, 1, 1, -1, -1, -1, 1, -1, -1, 1, 1, -1, 1, -1, 1, -1, -1, -1, -1, -1, 1, -1, -1, 1, -1, 1, 1, 1, -1, 1, 1, 1, 1, -1, 1, 1, -1, 1, -1, -1, 1, -1, -1, 1, 1, -1, -1, -1],
[1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, -1, 1, -1, -1, 1, 1, 1, 1, -1, 1, 1, 1, -1, 1, -1, -1, -1, -1, 1, -1, 1, 1, 1, -1, -1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1, -1, 1, 1, -1, -1, -1, 1, -1, 1, 1, -1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, -1, 1, -1, -1, -1, -1, 1, -1, -1, 1, -1, 1, -1, -1, -1, -1, 1, -1, 1, 1, -1, -1, 1, 1, 1, -1, -1, -1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, 1, 1],
[-1, 1, -1, -1, -1, -1, 1, -1, 1, 1, 1, 1, 1, -1, 1, -1, -1, -1, -1, 1, -1, 1, -1, -1, -1, -1, 1, -1, 1, -1, 1, -1, 1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1, 1, 1, 1, 1, -1, 1, -1, 1, -1, -1, -1, -1, 1, 1, -1, 1, 1, 1, -1, 1, -1, 1, -1, -1, 1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1, 1, -1, 1, 1, 1, -1, 1, -1, -1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1, 1, 1, -1, 1, -1, 1, -1, 1, 1, -1, 1, 1, 1, 1, -1, 1, -1, -1, -1, -1, 1, -1, 1, 1, -1, -1, -1, 1, 1, -1],
[1, -1, 1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, 1, -1, -1, -1, 1, 1, 1, -1, -1, 1, 1, 1, 1, -1, -1, 1, -1, 1, -1, -1, 1, 1, -1, 1, 1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1, 1, 1, 1, -1, 1, 1, -1, 1, 1, -1, -1, -1, -1, -1, -1, 1, 1, -1, 1, 1, 1, -1, -1, 1, -1, -1, -1, -1, 1, -1, 1, 1, 1, 1, 1, -1, 1, 1, -1, 1, -1, 1, 1, -1, 1, 1, -1, 1, -1, 1, 1, 1, 1, -1, 1, 1, 1, -1, -1],
])
preamble = np.array([1,1,1,1,1,1,1,1,1,0,1,0,1,0,1,0,1,1,1,1,1,1,1,1,1,0,1,0,1,0,1,0,1,1,1,1,1,1,1,1,1,0,1,0,1,0,1,0,1,1,1,1,1,1,1,1,1,0,1,0,1,0,1,0])

sent_payload =np.array([0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1])

def plt_time_fft(samples, sample_rate_hz, title_prefix="", peak_threshold = 0):
    """
    Compute FFT, plot spectrum, and detect peaks
    
    Args:
        samples: Input signal samples
        sample_rate_hz: Sample rate in Hz
        title_prefix: Optional prefix for plot title
    """
    # compute fft
    fft = np.fft.fftshift(np.fft.fft(samples))
    fft_db = 10*np.log10(np.abs(fft)**2)
    freqs_hz = np.fft.fftshift(np.fft.fftfreq(len(samples), 1/sample_rate_hz))
    
    # find peaks
    peaks, _ = signal.find_peaks(fft_db, height=peak_threshold, prominence=5, distance=10)
    
    # get peak frequencies and amplitudes
    peak_freqs_hz = freqs_hz[peaks]
    peak_amplitudes_db = fft_db[peaks]
    
    # sort peaks by amplitude (highest first)
    sort_indices = np.argsort(peak_amplitudes_db)[::-1]
    peak_freqs_hz = peak_freqs_hz[sort_indices]
    peak_amplitudes_db = peak_amplitudes_db[sort_indices]
    
    # log results
    logger.info(f"\nFFT Analysis - {title_prefix}:")
    logger.info(f"Found {len(peaks)} peaks above {peak_threshold:.2f} dB threshold")
    
    for i, (freq, amp) in enumerate(zip(peak_freqs_hz[:10], peak_amplitudes_db[:10])):  # show top 10
        logger.info(f"Peak {i+1}: {freq/1e3:+8.1f} kHz, {amp:6.1f} dB")
    
    # plot
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(np.real(samples), label='real')
    plt.plot(np.imag(samples), label='imag')

    plt.xlabel("sample number")
    plt.ylabel("magnitude")
    plt.title(f"{title_prefix}Samples")
    plt.grid(True)
    
    plt.subplot(2,1,2)
    plt.plot(freqs_hz/1e3, fft_db, label='FFT Magnitude')
    plt.axhline(y=peak_threshold, color='orange', linestyle=':', label=f'Peak Threshold ({peak_threshold:.1f} dB)')
    
    # # mark detected peaks
    # if len(peaks) > 0:
    #     plt.plot(peak_freqs_hz/1e3, peak_amplitudes_db, 'ro', markersize=8, label=f'Peaks ({len(peaks)} found)')
    
    plt.ylabel("amplitude (db)")
    plt.xlabel("frequency (kHz)")
    plt.legend()
    plt.grid(True)
    plt.title(f"{title_prefix}FFT Analysis with Peak Detection")
    
    return fft, fft_db, freqs_hz, peak_freqs_hz, peak_amplitudes_db


def dc_block(x: np.ndarray) -> np.ndarray:
    return x - np.mean(x)

def agc_simple(x: np.ndarray, target_rms: float = 1.0, eps: float = 1e-12) -> np.ndarray:
    rms = np.sqrt(np.mean(np.abs(x)**2) + eps)
    gain = target_rms / max(rms, eps)
    return x * gain

def lpf_fir(x: np.ndarray, cutoff_hz: float, fs: float, numtaps: int = 129) -> np.ndarray:
    # Safe LPF; keep cutoff >= chiprate pre-despread if you use it at all.
    if cutoff_hz >= fs/2:
        return x
    taps = signal.firwin(numtaps, cutoff_hz, fs=fs)
    return signal.lfilter(taps, [1.0], x)

def lpf_iir(samples: np.ndarray, cutoff_hz: float, fs_hz: float, order: int = 5):
    #filter
    b, a = signal.butter(order, cutoff_hz, 'low', fs=fs_hz)
    out_samples = signal.filtfilt(b, a, samples)
    return(out_samples)

def mix_filt_DCblock_AGC(in_samples, mix_freq_hz, samplerate_hz, cutoff_freq_hz):
    
    mid_samples = in_samples
    
    #dc block
    mid_samples = dc_block(mid_samples)
    
    #freq shift
    time_array_sec = np.linspace(0,(len(mid_samples)-1)/samplerate_hz,len(mid_samples))
    mid_samples = mid_samples*np.exp(-1j*2*np.pi*mix_freq_hz*time_array_sec)
    
    #filter
    # mid_samples = lpf_fir(mid_samples, cutoff_freq_hz, samplerate_hz, numtaps = 129*2)
    mid_samples = lpf_iir(mid_samples, cutoff_freq_hz, samplerate_hz)

    
    #AGC
    mid_samples = agc_simple(mid_samples)
    
    return mid_samples

def gc_search(samples, goldcode, samplerate_hz, sps, max_freq_dev_hz, freq_step_hz, peak_thresh):
    #generate goldcode at sps
    resampled_gc = np.repeat(goldcode , sps)
    
    #generate frequency and time array
    freq_offsets_hz = np.linspace(-1*max_freq_dev_hz, max_freq_dev_hz, int(2*max_freq_dev_hz/freq_step_hz+1))
    time_array_sec = np.linspace(0, (len(samples)-1)/samplerate_hz, len(samples))
    
    #correlation housekeeping
    corr_len = len(samples)-len(resampled_gc)+1
    X, Y = np.meshgrid(time_array_sec[0:corr_len], freq_offsets_hz)
    Z = np.zeros((len(freq_offsets_hz), corr_len))
    
    for j, freq_offset_hz in enumerate(freq_offsets_hz):
        test_samples = samples * np.exp(-1j*2*np.pi*freq_offset_hz*time_array_sec)
        
        correlation = np.correlate(test_samples, resampled_gc, mode='valid')
        Z[j, :] = np.abs(correlation)
        
    # fig_lab = plt.figure()
    # ax = fig_lab.add_subplot(111, projection='3d')
    # ax.plot_surface(X, Y, Z[:,:], cmap='viridis')
    # ax.set_title(f'Correlation with Repeated Goldcode')

    # # Set labels
    # ax.set_xlabel('Time Delay (s)')
    # ax.set_ylabel('Frequency Offset (Hz)')
    # ax.set_zlabel('Signal Strength')
    
    max_corr = np.max(Z)
    max_corr_idx = np.argmax(Z)
    max_corr_idx = np.unravel_index(max_corr_idx, Z.shape)
    freq_adj_hz = freq_offsets_hz[max_corr_idx[0]]


    gc_detected = max_corr > peak_thresh
    logger.info(f"max_corr: {max_corr}, gc_detected: {gc_detected}")
    # logger.info(f"freq index of max correlation: {max_corr_idx[0]}, corresponding freq_adj_hz: {freq_adj_hz}")
    
    
    return gc_detected, freq_adj_hz
    # threshold = peak_thresh_ratio * max_corr
    # indices = np.argwhere(Z > threshold)

    # peaks = []
    # for (i, j) in indices:
    #     freq = freq_offsets_hz[i]
    #     time = time_array_sec[j]
    #     val = Z[i, j]
    #     peaks.append((freq, time, val))
    # # logger.info(f"peaks: {peaks}")
    
    # return np.sum(Z > threshold), freq_adj_hz
    
def find_sps(samples, goldcode, target_sps, peak_threshold):
    upsampled_gc = np.repeat(goldcode, target_sps)
    
    corr_mag = np.abs(np.correlate(samples, upsampled_gc, mode='valid'))
    
    peaks, properties = signal.find_peaks(corr_mag, height= peak_threshold, distance=9000)
    
    logger.info(f'peak: {peaks}')
    
    lags = np.arange(len(corr_mag))
    peak_lags = lags[peaks]
    peak_values = corr_mag[peaks]
    
    first_peak_index = peak_lags[0]
    
    actual_gc_sps_mean = 0
    actual_gc_sps_sd = 100
    num_peaks = len(peaks)
    
    # calculate and log differences between consecutive peaks
    if len(peaks) > 1:
        peak_diffs = np.diff(peaks)
        logger.info(f"\nPeak index differences:")
        for i, diff in enumerate(peak_diffs):
            logger.info(f"Peak {i+1} to Peak {i+2}: {diff} samples")
        
        if len(peak_diffs) > 0:
            actual_gc_sps_mean = np.mean(peak_diffs)
            actual_gc_sps_sd = np.std(peak_diffs)
            
            logger.info(f"Mean difference: {actual_gc_sps_mean:.1f} samples")
            logger.info(f"Std deviation: {actual_gc_sps_sd:.1f} samples")            
            
      # plot
    plt.figure(figsize=(12, 6))
    plt.plot(lags, corr_mag, 'k-', alpha=0.8, label='Magnitude')
    plt.axhline(y=peak_threshold, color='orange', linestyle=':', label=f'Peak Threshold ({peak_threshold:.4f})')
    plt.plot(peak_lags, peak_values, 'ro', markersize=8, label=f'Peaks ({len(peaks)} found)')
    plt.ylabel("Magnitude")
    plt.xlabel("Lags")
    plt.legend()
    plt.grid(True)
    plt.title(f"Time Domain - Magnitude with Peaks")
    
    plt.tight_layout()
    
    return actual_gc_sps_mean, actual_gc_sps_sd, num_peaks, first_peak_index
    
def preamble_search(proc_samples, samplerate_hz, goldcode, target_sps, preamble):
    upsampled_gc = np.repeat(goldcode, target_sps)
    
    preamble_gc_at_sps = np.array([])
    for i in range(len(preamble)):
        bit = preamble[i]
        bit_bpsk = 2 * bit - 1  # Convert 0->-1, 1->+1
        modulated_chip_seq = upsampled_gc * bit_bpsk
        preamble_gc_at_sps = np.concatenate([preamble_gc_at_sps, modulated_chip_seq])
        
    pre_corr = signal.correlate(proc_samples, preamble_gc_at_sps, mode="valid")
    pre_corr_mag = np.abs(pre_corr)
    
    max_preamble_corr_val= max(pre_corr_mag)
    max_preamble_corr_ind = np.argmax(pre_corr_mag)
    max_preamble_corr_sign = np.sign(pre_corr[max_preamble_corr_ind])
    logger.info(f"max_preamble_corr_val: {max_preamble_corr_val}, max_preamble_corr_ind: {max_preamble_corr_ind}, max_preamble_corr_sign: {max_preamble_corr_sign} ")
    
    max_preamble_corr_val_start= max(pre_corr_mag[0:10160*2])
    max_preamble_corr_ind_start = np.argmax(pre_corr_mag[0:10160*2])
    max_preamble_corr_sign_start = np.sign(pre_corr[max_preamble_corr_ind_start])
    logger.info(f"max_preamble_corr_val: {max_preamble_corr_val_start}, max_preamble_corr_ind: {max_preamble_corr_ind_start}, max_preamble_corr_sign: {max_preamble_corr_sign_start} ")
    
    
    plt.figure(figsize=(12, 6))
    plt.plot(pre_corr_mag)
    
    plt.ylabel("Magnitude")
    plt.xlabel("Lags")
    plt.grid(True)
    plt.title(f"Preamble correlation")
    
    
    # return max_preamble_corr_ind, max_preamble_corr_sign
    return max_preamble_corr_ind_start, max_preamble_corr_sign

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

def bip_to_bin_bits(in_bip):
    return (in_bip+1)/2

def bin_to_bip_bits(in_bin):
    return in_bin*2-1

def sign_correction(in_bits, preamble):
    logger.info(f"in_bits: {in_bits}")
    logger.info(f"preamble: {preamble}")
    mult_pre_packet= bin_to_bip_bits(in_bits[56:len(preamble)])*bin_to_bip_bits(preamble[56:len(preamble)])
    pol = np.mean(mult_pre_packet)
    logger.info(f"pol: {pol},  diff_pre_packet: { mult_pre_packet}")
    
    fixed_bits = bip_to_bin_bits(np.sign(pol)*bin_to_bip_bits(in_bits))
    logger.info(f"fixed_bits: {fixed_bits}")
    
    return fixed_bits

    

def demod_cdma_packet(samples, samplerate_hz, cutoff_freq_hz, offset_freq_hz, goldcode, target_sps, preamble):

    #mix/filter/dc block/AGC
    logger.info(f"mixing to {offset_freq_hz} Hz with cutoff frequency {cutoff_freq_hz} Hz")    
    proc_samples = mix_filt_DCblock_AGC(samples, offset_freq_hz, samplerate_hz, cutoff_freq_hz*1.5)
    plt_time_fft(proc_samples, samplerate_hz, title_prefix="long samples functional processing: ")
    
    samples_per_gc_symbol= target_sps*len(goldcode)
    logger.info(f"samples_per_gc_symbol: {samples_per_gc_symbol}")
    
    #compute SPS and resample
    peak_threshold = 2000
    actual_gc_sps, actual_gc_sps_sd, num_peaks, _ = find_sps(proc_samples, goldcode, target_sps, peak_threshold)
    
    if num_peaks != 80:
        logger.warning(f"incorrect number of peaks in packet: {num_peaks}. Should be 80")
        return 0, 0
    if actual_gc_sps_sd > 10:
        logger.warning(f"high standard deviation in samples per symbol: {actual_gc_sps_sd}")
        return 0, 0
        
    logger.info(f"actual_gc_sps: {actual_gc_sps}")
    resamp_rate = target_sps*len(GCs[0])/actual_gc_sps
    logger.info(f"Resamp_rate: {resamp_rate}")
    proc_samples = signal.resample_poly(proc_samples, int(resamp_rate*10000), 10000)
    
    # #find signal start with preamble corr and grab those samples
    # preamble_start_index, preamble_sign = preamble_search(proc_samples, samplerate_hz, goldcode, target_sps, preamble)
    
    # packet_start_ind = int(preamble_start_index)
    # packet_stop_ind = int(preamble_start_index+80*samples_per_gc_symbol)


    
    #find start index from single goldcode correlation. Run the same function again
    actual_gc_sps, actual_gc_sps_sd, num_peaks, first_gc_index = find_sps(proc_samples, goldcode, target_sps, peak_threshold)
    
    if num_peaks != 80:
        logger.warning(f"incorrect number of peaks in packet: {num_peaks}. Should be 80")
        return 0, 0
    difference_gc_sps = actual_gc_sps-target_sps*len(goldcode)
    logger.info(f"difference_gc_sps: {difference_gc_sps}")
    if np.abs(difference_gc_sps) > 10:
        logger.warning(f"Resampling did not align the sample rates, actual_gc_sps: {actual_gc_sps}; target_sps: {target_sps}")
        return 0, 0
    
    packet_start_ind = int(first_gc_index)
    packet_stop_ind = int(first_gc_index +80*samples_per_gc_symbol)

    proc_samples = proc_samples[packet_start_ind: packet_stop_ind]
    
    #despread
    gc_at_target_sps = np.repeat(goldcode, target_sps)
    bits_to_calc = 80
    logger.debug("Expected number of bits: %s", bits_to_calc)
    samps_per_bit = int(target_sps*127)
    despread_samples = np.zeros(int(samps_per_bit*bits_to_calc)).astype(np.complex64)

    logger.info(f"samps_per_bit: {samps_per_bit}")
            
    for bit in range(bits_to_calc):
        # print(bit)
        despread_samples[bit*samps_per_bit:(bit+1)*samps_per_bit] = proc_samples[bit*samps_per_bit:(bit+1)*samps_per_bit]*gc_at_target_sps

    proc_samples = despread_samples
    plt_time_fft(proc_samples, samplerate_hz, title_prefix = "Despread Samples: ")


    #mm timing, costas loop, and bpsk demod
    
    proc_samples = mm_time_recovery(proc_samples, target_sps)
    # plt_time_fft(proc_samples, samplerate_hz, title_prefix = "MM Timing Recovery: ")

    #costas loop
    # proc_samples = costas_loop(proc_samples, samplerate_hz/127)
    proc_samples = costas_loop(proc_samples, samplerate_hz)
    # plt_time_fft(proc_samples, samplerate_hz, title_prefix = "Costas Loop: ")

    #bpsk demod
    rx_bits_raw = demod_bpsk(proc_samples)
    logger.info(f"rx_bits_raw: {rx_bits_raw}")

    gc_len = 127
    nbits_data = 80
    proc_bits = np.full(nbits_data, np.nan)

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
    
    logger.info(f"proc_bits pre correction: {proc_bits}")
    proc_bits = sign_correction(proc_bits, preamble)

    logger.info(f"proc_bits post correction: {proc_bits}")
    payload_bits = proc_bits[64:80]

    logger.info(f"sent payload bits: {sent_payload}")
    logger.info(f"received payloadbits: {payload_bits}")
    n_errors = np.sum(np.abs(sent_payload-payload_bits))
    logger.info(f"# errors: {n_errors}")
    
    
    
    


    # return demod_success, demod_bits
    return 1, payload_bits

# enable logging
logger = logging.getLogger("analysis") 
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(levelname)s: %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

samplerate_hz = 1000000
chiprate_hz = 12500
gc_len = len(GCs[0])
datarate_hz = chiprate_hz/gc_len
target_sps = samplerate_hz /chiprate_hz

num_gcs = len(GCs)

#Read file
# fname = "../python/src/cloud_samples/usrp_n210_20250919_130428_915MHz_1.000Msps_50.0dB_4000000samps.npy" #L053R8 tag (better clock + buffer), 50khz carrier, 80us per bit, proper packet 

#L053R8 tag (better clock + buffer), 50khz carrier, 80us per bit, proper packet 
fname = "../python/src/cloud_samples/usrp_n210_20250924_134913_915MHz_1.000Msps_50.0dB_18000000samps.npy"

try:
    samples = np.load(fname)
    logger.info(f"Loaded {len(samples)} samples from {fname}")
except FileNotFoundError:
    logger.error(f"File not found: {fname}")
    raise
except Exception as e:
    logger.error(f"Error loading samples from {fname}: {e}")
    

num_symbols_window_first_loop = 4
num_symbols_jump_first_loop = 2

num_samps_window_first_loop = int(num_symbols_window_first_loop*gc_len*target_sps)
num_samps_jump_first_loop = int(num_symbols_jump_first_loop*gc_len*target_sps)
num_loops = int(np.ceil(len(samples)/num_samps_jump_first_loop))
# num_loops = 80

logger.info(f"num_samps_first_loop: {num_samps_window_first_loop}")
logger.info(f"number of loops: {num_loops}")

recent_detected_gc = np.ones([4, num_gcs])
gc_detected = np.zeros([num_loops, num_gcs])
next_start_ind_gc = np.zeros(num_gcs)

# for start_loop_n in range(num_loops):
for start_loop_n in range(30, num_loops):
# for start_loop_n in range(70, num_loops):
    #grab specific samples
    start_samp = int(start_loop_n*num_samps_window_first_loop)
    end_samp = int(start_samp + num_samps_window_first_loop)
    logger.info(f"{start_loop_n}-> {start_samp}: {end_samp}")
    
    proc_samples = samples[start_samp:end_samp]
    
    #mix, filter, and AGC
    # offset_freq_hz = -124.5e3
    offset_freq_hz = -123.3e3
    proc_samples = mix_filt_DCblock_AGC(proc_samples, offset_freq_hz, samplerate_hz, chiprate_hz*1.5)

    # plt_time_fft(proc_samples, samplerate_hz)
    
    #test for presence of each goldcode
    for gc_n in range(1):#num_gcs):
        if start_samp >= next_start_ind_gc[gc_n]:
            max_freq_dev_hz = 1000
            freq_step_hz = 50
            peak_thresh = 3000
            peak_det, freq_adj_hz = gc_search(proc_samples, GCs[gc_n], samplerate_hz, target_sps,  max_freq_dev_hz, freq_step_hz, peak_thresh)

            
            
                # gc_detected[start_loop_n, gc_n] = 1
                # break
            gc_detected[start_loop_n, gc_n] = peak_det
            recent_detected_gc[0:3, gc_n] = recent_detected_gc[1:4, gc_n]
            recent_detected_gc[3,gc_n] = peak_det
            
            
            signal_to_demod = sum(recent_detected_gc[:,gc_n] == [0,0,1,1]) == 4
            
            logger.info(f"recent_detected_gc: {recent_detected_gc[:,gc_n]}, signal_to_demod: {signal_to_demod}")
            
            if signal_to_demod:
            # if start_loop_n == 34:
                logger.info("demodding")
                
                #select indexes
                long_samples_start_ind = int(start_samp- 2*num_samps_window_first_loop)
                # long_samples_end_ind = long_samples_start_ind + 90*num_samps_window_first_loop
                long_samples_end_ind = long_samples_start_ind + 24*num_samps_window_first_loop
                long_samples = samples[long_samples_start_ind:long_samples_end_ind]
                coarse_offset_freq = offset_freq_hz+freq_adj_hz
                # coarse_offset_freq = offset_freq_hz+250
                
                #send to demod
                demod_success, demod_bits = demod_cdma_packet(long_samples, samplerate_hz, chiprate_hz*1.5, coarse_offset_freq, GCs[gc_n], target_sps, preamble)
        
                if demod_success:
                    next_start_ind_gc[gc_n] = long_samples_start_ind+36*num_samps_window_first_loop
                # plt.show()
# logger.info("gc detected:\n%s", np.array2string(gc_detected, threshold=np.inf, max_line_width=np.inf))

    
    # plt.show()