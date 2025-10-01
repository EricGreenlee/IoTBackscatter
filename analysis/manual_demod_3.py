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
    # logger.info(f"freq index of max correlation: {max_corr_idx[0]}, corresponding freq_adj_hz: {freq_adj_hz}")
    
    
    
    return first_peak_index, actual_gc_sps_mean, num_peaks

def despread_and_integrate(samples, goldcode, sps_chip):
    """
    Despread samples with a Gold code and integrate over one symbol (127 chips).
    Returns one complex symbol per Gold code period.
    """
    gc_os = np.repeat(goldcode.astype(np.complex64), sps_chip)
    n_sym_chips = len(gc_os)             # samples per symbol
    n_syms = len(samples) // n_sym_chips
    samples = samples[:n_syms*n_sym_chips]

    gc_stream = np.tile(gc_os, n_syms)
    despread = samples * gc_stream
    sym_matrix = despread.reshape(n_syms, n_sym_chips)
    sym_samples = sym_matrix.sum(axis=1)  # coherent integrate

    return sym_samples


def correct_with_preamble(sym_samples, preamble_bits):
    """
    Estimate constant phase and residual CFO from the preamble.
    Rotate the entire sequence accordingly.
    """
    # Map 0/1 → ±1 reference symbols
    preamble_ref = 2*preamble_bits.astype(np.float32) - 1
    N = len(preamble_ref)

    # Phase error sequence
    phase_err = np.unwrap(np.angle(sym_samples[:N] * preamble_ref))
    k = np.arange(N)
    coeffs = np.polyfit(k, phase_err, 1)  # slope + intercept
    slope, offset = coeffs[0], coeffs[1]

    n = np.arange(len(sym_samples))
    correction = np.exp(-1j*(slope*n + offset))
    sym_corrected = sym_samples * correction

    return sym_corrected

def costas_loop_bpsk(samples, fs, loop_bw_hz=300.0, zeta=0.707):
    """Second-order Costas loop for BPSK."""
    N = len(samples)
    out = np.zeros(N, dtype=np.complex64)
    phase = 0.0
    freq = 0.0

    # loop filter coeffs
    wn = 2*np.pi*loop_bw_hz * (4*zeta / (zeta + 1/(4*zeta)))
    alpha = (2*zeta*wn) / fs
    beta  = (wn**2) / fs

    for i in range(N):
        x = samples[i] * np.exp(-1j*phase)
        out[i] = x
        error = np.real(x) * np.imag(x)   # BPSK error detector
        freq += beta * error
        phase += freq + alpha * error
        if phase >= 2*np.pi: phase -= 2*np.pi
        elif phase < 0: phase += 2*np.pi
    return out

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


proc_samples = samples[start_index:stop_index]

#mix and filter, no AGC or DC blocking
offset_freq_hz = -123.3e3
proc_samples = mix(proc_samples, offset_freq_hz, samplerate_hz)
logger.info(f"mixed to {offset_freq_hz} Hz")

# plt_time_fft(proc_samples, sample_rate_hz=samplerate_hz, title_prefix="Post mixing: ", peak_threshold= 10)

proc_samples = lpf_iir(proc_samples, chiprate_hz*1.5, samplerate_hz)
# plt_time_fft(proc_samples, sample_rate_hz=samplerate_hz, title_prefix="Post filtering: ", peak_threshold= 10)

#skip DC block and IQ fix for now

# search for goldcode
test_samples = proc_samples[10*gc_len*target_input_sps:20*gc_len*target_input_sps]


max_freq_dev_hz = 1000
freq_step_hz = 100
peak_thresh = .05
peak_det, freq_adj_hz = normalized_gc_search(test_samples, GCs[gc_n], samplerate_hz, target_sps,  max_freq_dev_hz, freq_step_hz, peak_thresh)

#mix to intermediate freq
proc_samples = mix(proc_samples, freq_adj_hz, samplerate_hz)
logger.info(f"mixed to {freq_adj_hz} Hz")


#time and sps align
start_ind, actual_input_gc_sps, num_peaks = get_start_ind_sps(proc_samples, GCs[gc_n], target_sps)

proc_samples = proc_samples[start_ind:start_ind+80*gc_len*target_input_sps]


resamp_rate = target_sps*len(GCs[0])/actual_input_gc_sps
logger.info(f"resamp_rate: {resamp_rate}")

proc_samples = signal.resample_poly(proc_samples, int(resamp_rate*10000), 10000)

#test it again to make sure resampling and rolling worked correctly
# start_ind, actual_input_gc_sps, num_peaks = get_start_ind_sps(proc_samples, GCs[gc_n], target_sps)
plt_time_fft(proc_samples, sample_rate_hz=samplerate_hz, title_prefix="Post resampling: ", peak_threshold= 10)


# #Downsample to a smaller number of sps
# target_output_sps_chip = 16
# Fs_target = chiprate_hz * target_output_sps_chip

# p, q = Fraction(Fs_target / samplerate_hz).limit_denominator(1000).as_integer_ratio()
# proc_samples = signal.resample_poly(proc_samples, p, q)
# samplerate_hz = Fs_target

# logger.info(f"Downsampled to {target_output_sps_chip} samples/chip → {Fs_target:.1f} Hz effective Fs")
# plt_time_fft(proc_samples, sample_rate_hz=samplerate_hz, title_prefix="Post downsampling: ", peak_threshold= 10)


#costas loop for better frequency alignment
proc_samples = costas_loop_bpsk(proc_samples, fs=samplerate_hz,loop_bw_hz=300.0)

#despread
# Step 3: despread + integrate
target_output_sps_chip = 80
sym_samples = despread_and_integrate(proc_samples, GCs[gc_n], target_output_sps_chip)
plt_time_fft(sym_samples, sample_rate_hz=samplerate_hz, title_prefix="Post despreading: ", peak_threshold= 1000)


#costas loop
Fs_sym = chiprate_hz / len(GCs[0])   # ~197 Hz
sym_tracked = costas_loop_bpsk(sym_samples, fs=Fs_sym, loop_bw_hz=8.0)

# Step 4: phase/CFO correction using preamble
sym_corrected = correct_with_preamble(sym_samples, preamble)

plt.figure()
plt.scatter(np.real(sym_corrected), np.imag(sym_corrected), c='b', alpha=0.6)
plt.title("Constellation after preamble correction")
plt.xlabel("In-phase")
plt.ylabel("Quadrature")
plt.grid(True)


# Step 5: decision
rx_bits = (np.real(sym_corrected) > 0).astype(int)

# Step 6: BER check
logger.info(f"Received bits: {rx_bits[:len(sent_payload)]}")
logger.info(f"Sent payload:  {sent_payload}")
errors = np.sum(rx_bits[:len(sent_payload)] != sent_payload)
logger.info(f"BER: {errors}/{len(sent_payload)} = {errors/len(sent_payload):.2%}")



plt.show()


