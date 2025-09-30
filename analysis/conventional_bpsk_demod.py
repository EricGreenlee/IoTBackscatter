#Search for goldcodes and demodulate the bits if a goldcode is present

#imports
import logging
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import scipy.signal as signal
from fractions import Fraction

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
    
    # for i, (freq, amp) in enumerate(zip(peak_freqs_hz[:10], peak_amplitudes_db[:10])):  # show top 10
    #     logger.info(f"Peak {i+1}: {freq/1e3:+8.1f} kHz, {amp:6.1f} dB")
    
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
    
    # return fft, fft_db, freqs_hz, peak_freqs_hz, peak_amplitudes_db

def mix(in_samples, mix_freq_hz, samplerate_hz):
    time_array_sec = np.linspace(0,(len(in_samples)-1)/samplerate_hz,len(in_samples))
    out_samples = in_samples*np.exp(-1j*2*np.pi*mix_freq_hz*time_array_sec)
    
    return out_samples

def lpf_iir(samples: np.ndarray, cutoff_hz: float, fs_hz: float, order: int = 5):
    #filter
    b, a = signal.butter(order, cutoff_hz, 'low', fs=fs_hz)
    out_samples = signal.filtfilt(b, a, samples)
    return(out_samples)

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


    gc_detected = max_corr > peak_thresh
    logger.info(f"max_corr: {max_corr}, gc_detected: {gc_detected}")
    # logger.info(f"freq index of max correlation: {max_corr_idx[0]}, corresponding freq_adj_hz: {freq_adj_hz}")
    
    
    return gc_detected, freq_adj_hz

def normalized_gc_search(samples, goldcode, samplerate_hz, sps, max_freq_dev_hz, freq_step_hz, peak_thresh):
    #generate goldcode at sps
    resampled_gc = np.repeat(goldcode , sps)
    gc_sps_len = len(resampled_gc)
    gc_energy = np.vdot(resampled_gc, resampled_gc).real
    
    #generate frequency and time array
    freq_offsets_hz = np.linspace(-max_freq_dev_hz, max_freq_dev_hz, int(2*max_freq_dev_hz/freq_step_hz+1))
    time_array_sec = np.linspace(0, (len(samples)-1)/samplerate_hz, len(samples))
    
    #correlation housekeeping
    corr_len = len(samples)-gc_sps_len+1
    X, Y = np.meshgrid(time_array_sec[0:corr_len], freq_offsets_hz)
    Z = np.zeros((len(freq_offsets_hz), corr_len))
    
    for j, freq_offset_hz in enumerate(freq_offsets_hz):
        test_samples = samples * np.exp(-1j*2*np.pi*freq_offset_hz*time_array_sec)
        
        numer = signal.correlate(test_samples, resampled_gc, mode='valid', method= "fft")
        denom = np.sqrt(
            signal.correlate(np.abs(test_samples)**2, np.ones(gc_sps_len ), mode="valid") * gc_energy
        )
        gamma = np.abs(numer)**2 / (denom**2 + 1e-20)
        Z[j, :] = gamma
        
    fig_lab = plt.figure()
    ax = fig_lab.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z[:,:], cmap='viridis')
    ax.set_title(f'Normalized Correlation with Repeated Goldcode')

    # Set labels
    ax.set_xlabel('Time Delay (s)')
    ax.set_ylabel('Frequency Offset (Hz)')
    ax.set_zlabel('Signal Strength')
    
    max_corr = np.max(Z)
    max_corr_idx = np.argmax(Z)
    max_corr_idx = np.unravel_index(max_corr_idx, Z.shape)
    freq_adj_hz = freq_offsets_hz[max_corr_idx[0]]


    gc_detected = max_corr > peak_thresh
    logger.info(f"max_corr: {max_corr}, gc_detected: {gc_detected}")
    # logger.info(f"freq index of max correlation: {max_corr_idx[0]}, corresponding freq_adj_hz: {freq_adj_hz}")
    
    
    return gc_detected, freq_adj_hz

def get_start_ind_sps(samples, goldcode, target_sps, peak_threshold = 0.05):
    
    #generate goldcode at sps
    resampled_gc = np.repeat(goldcode, target_sps)
    gc_sps_len = len(resampled_gc)
    gc_energy = np.vdot(resampled_gc, resampled_gc).real
        
    numer = signal.correlate(samples, resampled_gc, mode='valid', method= "fft")
    denom = np.sqrt(
        signal.correlate(np.abs(samples)**2, np.ones(gc_sps_len ), mode="valid") * gc_energy
    )
    corr_mag = np.abs(numer)**2 / (denom**2 + 1e-20)
        # Z[j, :] = gamma
        
    peaks, properties = signal.find_peaks(corr_mag, height= peak_threshold, distance=int(gc_sps_len*.5))
    
    logger.info(f'peak: {peaks}')
    
    lags = np.arange(len(corr_mag))
    peak_lags = lags[peaks]
    peak_values = corr_mag[peaks]
    
    first_peak_index = 0
    
    actual_gc_sps_mean = 0
    actual_gc_sps_sd = 100
    num_peaks = len(peaks)
    
    # calculate and log differences between consecutive peaks
    if len(peaks) > 1:
        first_peak_index = peak_lags[0]
        peak_diffs = np.diff(peaks)
        logger.info(f"\nPeak index differences:")
        # for i, diff in enumerate(peak_diffs):
        #     logger.info(f"Peak {i+1} to Peak {i+2}: {diff} samples")
        
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
    # logger.info(f"freq index of max correlation: {max_corr_idx[0]}, corresponding freq_adj_hz: {freq_adj_hz}")
    
    
    
    return first_peak_index, actual_gc_sps_mean, num_peaks

# def despread_and_integrate(samples, goldcode, sps_chip):
#     """
#     Despread samples with a Gold code and integrate over one symbol (127 chips).
#     Returns one complex symbol per Gold code period.
#     """
#     gc_os = np.repeat(goldcode.astype(np.complex64), sps_chip)
#     n_sym_chips = len(gc_os)             # samples per symbol
#     n_syms = len(samples) // n_sym_chips
#     samples = samples[:n_syms*n_sym_chips]

#     gc_stream = np.tile(gc_os, n_syms)
#     despread = samples * gc_stream
#     sym_matrix = despread.reshape(n_syms, n_sym_chips)
#     sym_samples = sym_matrix.sum(axis=1)  # coherent integrate

#     return sym_samples


# def correct_with_preamble(sym_samples, preamble_bits):
#     """
#     Estimate constant phase and residual CFO from the preamble.
#     Rotate the entire sequence accordingly.
#     """
#     # Map 0/1 → ±1 reference symbols
#     preamble_ref = 2*preamble_bits.astype(np.float32) - 1
#     N = len(preamble_ref)

#     # Phase error sequence
#     phase_err = np.unwrap(np.angle(sym_samples[:N] * preamble_ref))
#     k = np.arange(N)
#     coeffs = np.polyfit(k, phase_err, 1)  # slope + intercept
#     slope, offset = coeffs[0], coeffs[1]

#     n = np.arange(len(sym_samples))
#     correction = np.exp(-1j*(slope*n + offset))
#     sym_corrected = sym_samples * correction

#     return sym_corrected

# def costas_loop_bpsk(samples, fs, loop_bw_hz=300.0, zeta=0.707):
#     """Second-order Costas loop for BPSK."""
#     N = len(samples)
#     out = np.zeros(N, dtype=np.complex64)
#     phase = 0.0
#     freq = 0.0

#     # loop filter coeffs
#     wn = 2*np.pi*loop_bw_hz * (4*zeta / (zeta + 1/(4*zeta)))
#     alpha = (2*zeta*wn) / fs
#     beta  = (wn**2) / fs

#     for i in range(N):
#         x = samples[i] * np.exp(-1j*phase)
#         out[i] = x
#         error = np.real(x) * np.imag(x)   # BPSK error detector
#         freq += beta * error
#         phase += freq + alpha * error
#         if phase >= 2*np.pi: phase -= 2*np.pi
#         elif phase < 0: phase += 2*np.pi
#     return out

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

def dc_block_iir(x: np.ndarray, r: float = 0.995) -> np.ndarray:
    """
    Simple complex DC blocker: y[n] = x[n] - x[n-1] + r*y[n-1]
    r close to 1.0 → lower cutoff. Works well to remove DC/LO leakage pre-correlation.
    """
    y = np.zeros_like(x, dtype=np.complex64)
    prev_x = 0+0j
    for i, xi in enumerate(x.astype(np.complex64)):
        y[i] = (xi - prev_x) + (r * (y[i-1] if i else 0))
        prev_x = xi
    return y

def rms_agc(x: np.ndarray, target_rms: float = 1.0, attack: float = 0.01, decay: float = 0.001) -> np.ndarray:
    """
    Feed-forward RMS AGC with separate attack/decay to avoid pumping. Returns complex64.
    """
    env = 0.0
    y = np.empty_like(x, dtype=np.complex64)
    for i, xi in enumerate(x):
        mag2 = xi.real*xi.real + xi.imag*xi.imag
        a = attack if mag2 > env else decay
        env = (1 - a)*env + a*mag2
        gain = target_rms / np.sqrt(env + 1e-12)
        y[i] = xi * gain
    return y

def lpf_fir(x: np.ndarray, fs_hz: float, cutoff_hz: float, taps: int = 129) -> np.ndarray:
    """Linear-phase FIR low-pass used before any decimation to avoid aliasing/timing distortion."""
    h = signal.firwin(taps, cutoff_hz, fs=fs_hz)
    return signal.lfilter(h, 1.0, x)

def costas_loop_bpsk(x: np.ndarray, fs_hz: float, loop_bw_hz: float = 100.0, zeta: float = 0.707) -> np.ndarray:
    """
    2nd-order Costas loop for BPSK. loop_bw_hz ~ 0.5–1% of chiprate is a good start.
    """
    out = np.zeros_like(x, dtype=np.complex64)
    phase = 0.0
    freq  = 0.0

    # Standard discrete-time 2nd-order loop (alpha/beta from BL, zeta)
    BL = loop_bw_hz
    Wn = 2*np.pi*BL
    # Simple bilinear-style approximation for alpha/beta at sample rate fs
    # (These values are robust for our purposes)
    Kp = 1.0  # BPSK detector gain
    z  = zeta
    denom = 1 + 2*z*(Wn/fs_hz) + (Wn/fs_hz)**2
    alpha = (4*Kp*2*z*(Wn/fs_hz)) / denom
    beta  = (4*Kp*(Wn/fs_hz)**2)  / denom

    for i, s in enumerate(x):
        v = s * np.exp(-1j*phase)
        out[i] = v
        err = np.real(v) * np.imag(v)  # BPSK Costas error
        freq  += beta  * err
        phase += freq + alpha * err
        # Wrap for numerical hygiene
        if phase >= np.pi: phase -= 2*np.pi
        elif phase < -np.pi: phase += 2*np.pi

    return out

def gardner_2sps(x: np.ndarray, gain: float = 0.02) -> np.ndarray:
    """
    Gardner timing at exactly 2 sps. Emits ~1 sample/chip, centered.
    """
    x = np.asarray(x, dtype=np.complex64)
    k = 0
    out = []
    mu = 0.0
    N = len(x)
    while k + 2 < N:
        x0, xm, x1 = x[k], x[k+1], x[k+2]
        e = np.real((x0 - x1) * np.conj(xm))  # Gardner TED
        mu += gain * e
        if mu >= 0.5:
            k += 3  # slip a sample when we drift too far
            mu -= 1.0
        else:
            k += 2
        out.append(xm)  # mid-sample is symbol-center at 2 sps
    return np.asarray(out, dtype=np.complex64)

def integrate_and_dump(x: np.ndarray, sps: int = 1) -> np.ndarray:
    """Sum over each symbol period. For 1 chip/symbol and 1 sps, this is identity."""
    if sps == 1:
        return x
    N = (len(x)//sps)*sps
    return x[:N].reshape(-1, sps).sum(axis=1)

def hard_bits(x: np.ndarray) -> np.ndarray:
    """Real > 0 → 1 else 0."""
    return (np.real(x) > 0).astype(np.int8)

def desired_resample_up_down(actual_gc_sps: float, target_sps_per_chip: float, gc_len: int) -> tuple[int,int]:
    """
    Compute rational up/down to convert from actual samples-per-GC (measured)
    to desired samples-per-chip (target) while preserving chiprate reference.
    """
    # We know actual samples per GC (actual_gc_sps); we want (target_sps_per_chip * gc_len) samples per GC.
    ratio = (target_sps_per_chip * gc_len) / actual_gc_sps  # new/old
    frac = Fraction(ratio).limit_denominator(10000)
    return frac.numerator, frac.denominator

def bip_to_bin_bits(in_bip):
    return (in_bip+1)/2

def bin_to_bip_bits(in_bin):
    return in_bin*2-1

def estimate_snr(sig):
    power = np.mean(np.abs(sig)**2)
    noise = np.var(sig - np.mean(sig))
    return 10*np.log10(power/noise)


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
    
target_input_sps = 80
gc_n = 0

start_index = 1219200 #+ 10*gc_len*target_input_sps
stop_index = start_index + 100*gc_len*target_input_sps
# stop_index = start_index + 10*gc_len*target_input_sps


# proc_samples = samples[start_index:stop_index]

# --- simplified raw-chip demod path (no despreading) ---

proc = samples[start_index:stop_index]
logger.info(f"Initial slice: {len(proc)} samples")

# 3) DC block (light) BEFORE correlation so DC won’t bias the normalized metric
proc = dc_block_iir(proc, r=0.995)

# 1) Mix to rough center (bring near baseband)
offset_freq_hz = -123.3e3
proc = mix(proc, offset_freq_hz, samplerate_hz)
logger.info(f"Mixed to {offset_freq_hz:+.1f} Hz")

# 2) Narrow LPF ~1.2–1.5 × chiprate (keep spectrum, reject close interferers)
proc = lpf_iir(proc, cutoff_hz=chiprate_hz*1.5, fs_hz=samplerate_hz, order=5)

# # 3) DC block (light) BEFORE correlation so DC won’t bias the normalized metric
# proc = dc_block_iir(proc, r=0.995)

# 4) Normalized GC search on a window → fine Δf
test_win = proc[10*gc_len*target_input_sps : 20*gc_len*target_input_sps]
max_freq_dev_hz = 1000
freq_step_hz     = 100
peak_thresh      = 0.05
gc_found, fine_cfo_hz = normalized_gc_search(
    test_win, GCs[gc_n], samplerate_hz, target_sps,
    max_freq_dev_hz, freq_step_hz, peak_thresh
)
logger.info(f"GC found? {gc_found} | fine CFO: {fine_cfo_hz:+.1f} Hz")

# 5) Apply fine CFO correction on the full region
proc = mix(proc, fine_cfo_hz, samplerate_hz)
logger.info(f"Applied fine CFO mix {fine_cfo_hz:+.1f} Hz")

# print("Pre-resample SNR:", estimate_snr(proc))

# 6) Use GC correlation to find start index and the actual samples-per-GC spacing
start_ind, actual_input_gc_sps, num_peaks = get_start_ind_sps(proc, GCs[gc_n], target_sps)
logger.info(f"Start index: {start_ind} | Measured samples/GC: {actual_input_gc_sps:.2f} | peaks: {num_peaks}")

# Trim to a packet region beginning at the best boundary (grab ~80 GCs like before)
# Keep enough samples to survive resampling margins
chips_per_packet = 80 * gc_len
buffer_nsamps = 80*200

want_input_samps = int(np.ceil(chips_per_packet * (actual_input_gc_sps / gc_len)))+buffer_nsamps

checkpoint_samps = proc

for samp_offset in [0]:#[-2,-1,0,1,2]:
    logger.info(f"samp_offset: {samp_offset}")
    
    proc = checkpoint_samps[start_ind +samp_offset: start_ind+ samp_offset + want_input_samps]
    logger.info(f"Trimmed to packet region: {len(proc)} samples")

    # plt_time_fft(proc, sample_rate_hz=samplerate_hz, title_prefix="Post trimming: ", peak_threshold= 10)

    # plt.figure()
    # plt.plot(np.real(proc), np.imag(proc), '.')
    # plt.title("constellation after trimming")
    # plt.grid(True)


    # 7) Resample to target 2 samples/chip (better for Gardner/Costas)
    target_sps_per_chip = 2.0
    # target_sps_per_chip = 8.0
    up, down = desired_resample_up_down(actual_input_gc_sps, target_sps_per_chip, gc_len)
    proc = signal.resample_poly(proc, up, down)
    fs_after = samplerate_hz * (up / down)
    logger.info(f"Resampled with up={up}, down={down} → fs={fs_after:.1f} Hz; ~{fs_after/chiprate_hz:.2f} sps/chip")
    
    # fs_after = samplerate_hz/40
    # proc = signal.decimate(proc, 40, ftype="fir", zero_phase=True)

    # plt_time_fft(proc, sample_rate_hz=fs_after, title_prefix="Post resampling: ", peak_threshold= 10)
    # print("Post-resample SNR:", estimate_snr(proc))


    # 8) (Optional) light FIR to clean any resampling images (cut around 0.45*chiprate)
    proc = lpf_fir(proc, fs_after, cutoff_hz=0.45*chiprate_hz)

    # 9) AGC now that BW and rate are low (stable loops)
    proc = rms_agc(proc, target_rms=1.0, attack=0.02, decay=0.002)

    # 10) Carrier recovery (Costas) BEFORE timing; BW ≈ 0.5–1% of chiprate
    proc = costas_loop_bpsk(proc, fs_after, loop_bw_hz=100.0, zeta=0.707)

    plt.figure()
    plt.plot(np.real(proc), np.imag(proc), '.')
    plt.title(f"constellation after costas loop, sample offset: {samp_offset}")
    plt.grid(True)

    # 11) Timing recovery (Gardner at exactly 2 sps) → ~1 sample per chip, centered
    proc = gardner_2sps(proc, gain=0.02)

    # 12) Matched filter / integrate-and-dump (1 chip/symbol here) → hard decisions
    chips_cx = integrate_and_dump(proc, sps=1)
    raw_chip_bits = hard_bits(chips_cx)
    logger.info(f"Recovered {len(raw_chip_bits)} raw chip bits at ≈{chiprate_hz:.1f} chips/s")

    # 13) Optional: GC-only correlation for alignment sanity (no despreading)
    gc_pm = GCs[gc_n].astype(np.float32)  # ±1
    c = signal.correlate(2*raw_chip_bits-1, gc_pm, mode='valid', method='fft')

    plt_time_fft(np.real(c), sample_rate_hz=chiprate_hz, title_prefix="bit correlation with goldcode : ", peak_threshold= 10)

    peaks, _ = signal.find_peaks(np.abs(c), height = gc_len/2)
    # (fft_db, height=peak_threshold, prominence=5, distance=10)

    logger.info(f"peaks: {peaks}")

    # packet_start_ind = peaks[0]

    # gc_boundary = np.argmax(np.abs(c))
    gc_boundary_start = 32
    gc_boundary_start = peaks[0]
    logger.info(f"Likely GC boundary in raw chips at index {gc_boundary_start}")
    # logger.info(f"Raw chip bits @boundary: {bin_to_bip_bits(raw_chip_bits[gc_boundary:gc_boundary+gc_len])}")
    # logger.info(f"Goldcode: {GCs[0]}")

    tot_errors = 0
    for i in range(80):
        gc_boundary = gc_boundary_start+i*gc_len
        tx_rx_dif = GCs[0]+bin_to_bip_bits(raw_chip_bits[gc_boundary:gc_boundary+gc_len])
        num_errors = min(np.sum(np.abs(tx_rx_dif)/2),127-np.sum(np.abs(tx_rx_dif)/2))
        # logger.info(f"difference: {tx_rx_dif}")
        logger.info(f"gc symbol: {i}, num_errors: {num_errors}")
        tot_errors = tot_errors+num_errors
        
    logger.info(f"total_errors: {tot_errors}")
    logger.info(f"BER: {tot_errors/(80*gc_len)}")

    # --- Chip-level BER across the whole spread BPSK packet ---

    # Build transmitted chip sequence
    tx_bits = np.concatenate([preamble, sent_payload])
    tx_chips = []
    gc = GCs[gc_n].astype(int)  # ±1 sequence

    for b in tx_bits:
        symbol_chips = gc * (1 if b == 1 else -1)  # BPSK spreading
        tx_chips.append(symbol_chips)

    tx_chips = np.concatenate(tx_chips)  # length = 80*127 = 10160

    # Align RX to boundary before comparison
    gc_boundary = gc_boundary_start
    rx_chips = raw_chip_bits[gc_boundary : gc_boundary + len(tx_chips)]

    # Make sure lengths match
    min_len = min(len(rx_chips), len(tx_chips))
    rx_chips = rx_chips[:min_len]
    tx_chips = tx_chips[:min_len]

    # Compute BER
    chip_errors = np.sum(rx_chips != bip_to_bin_bits(tx_chips))
    chip_ber = min(chip_errors,min_len-chip_errors) / min_len

    # plt.figure()
    # plt.plot(rx_chips, label= "rx_chips")
    # plt.plot(bip_to_bin_bits(-1*tx_chips)+2, label= "inverted tx_chips")
    # plt.plot(bip_to_bin_bits(tx_chips)-2, label= "tx_chips")
    # plt.grid(True)
    # plt.legend()

    logger.info(f"--- Chip-level BER ---")
    logger.info(f"Compared {min_len} chips")
    logger.info(f"Errors: {chip_errors}")
    logger.info(f"BER: {chip_ber:.3e}")


#plot with easy closing
def on_key(event):
    if event.key == 'q':
        plt.close('all')
        logger.info("All plot windows closed")

# Connect the key event handler to all figures
for fig_num in plt.get_fignums():
    fig = plt.figure(fig_num)
    fig.canvas.mpl_connect('key_press_event', on_key)

logger.info("Press 'q' in any plot window to close all plots")
plt.show()


