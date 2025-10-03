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

def lpf_fir(x: np.ndarray, fs_hz: float, cutoff_hz: float, taps: int = 129) -> np.ndarray:
    """Linear-phase FIR low-pass used before any decimation to avoid aliasing/timing distortion."""
    h = signal.firwin(taps, cutoff_hz, fs=fs_hz)
    return signal.lfilter(h, 1.0, x)

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

    phase_array = []
    freq_array = []
    for i, s in enumerate(x):
        v = s * np.exp(-1j*phase)
        out[i] = v
        err = np.real(v) * np.imag(v)  # BPSK Costas error
        freq  += beta  * err
        phase += freq + alpha * err
        # Wrap for numerical hygiene
        if phase >= np.pi: phase -= 2*np.pi
        elif phase < -np.pi: phase += 2*np.pi
        phase_array.append(phase)
        freq_array.append(freq)
        
    plt.figure()
    plt.plot(freq_array)
    plt.title("Phase estimate over time")
    plt.grid(True)

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
    corr_val = numer/np.abs(denom+ 1e-20)
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
    plt.subplot(2,1,1)
    plt.plot(lags, corr_mag, 'k-', alpha=0.8, label='Magnitude')
    plt.axhline(y=peak_threshold, color='orange', linestyle=':', label=f'Peak Threshold ({peak_threshold:.4f})')
    plt.plot(peak_lags, peak_values, 'ro', markersize=8, label=f'Peaks ({len(peaks)} found)')
    plt.ylabel("Magnitude")
    plt.xlabel("Lags")
    plt.legend()
    plt.grid(True)
    plt.title(f"Time Domain - Magnitude with Peaks")
    
    plt.subplot(2,1,2)
    plt.plot(lags, np.real(corr_val), '-', alpha=0.8, label='Real Values')
    plt.plot(lags, np.imag(corr_val), '-', alpha=0.8, label='Imag Values')
    plt.legend()
    plt.grid(True)
    plt.title(f"Time Domain - Real and image with Peaks")
    plt.ylabel("Amplitude")
    plt.xlabel("Lags")
    plt.tight_layout()
    # logger.info(f"freq index of max correlation: {max_corr_idx[0]}, corresponding freq_adj_hz: {freq_adj_hz}")
    
    
    
    return first_peak_index, actual_gc_sps_mean, num_peaks


    
# fname = "../python/src/cloud_samples/usrp_n210_20251002_141525_915MHz_0.200Msps_38.0dB_1000000samps.npy"
fname = "../python/src/cloud_samples/usrp_n210_20251002_154324_915MHz_2.000Msps_38.0dB_10000000samps.npy"

try:
    samples = np.load(fname)
    logger.info(f"Loaded {len(samples)} samples from {fname}")
except FileNotFoundError:
    logger.error(f"File not found: {fname}")
    raise
except Exception as e:
    logger.error(f"Error loading samples from {fname}: {e}")
    
gc_n = 0

samplerate_hz = 2000000
chiprate_hz = 25000
gc_len = len(GCs[0])
datarate_hz = chiprate_hz/gc_len
# target_sps = samplerate_hz /chiprate_hz
ideal_input_sps = int(samplerate_hz /chiprate_hz)

num_gcs = len(GCs)


start_index = int(0+ 1*80*gc_len*ideal_input_sps)
stop_index = int(start_index + 120*gc_len*ideal_input_sps)

proc_samples = samples[start_index: stop_index]

plt_time_fft(proc_samples, sample_rate_hz=samplerate_hz, title_prefix="Initial input")

#mix to -65.41
mix_freq_hz = -65010
proc_samples = mix(proc_samples, mix_freq_hz, samplerate_hz)

plt_time_fft(proc_samples, sample_rate_hz=samplerate_hz, title_prefix="After mixing")

#filter
proc_samples = lpf_fir(proc_samples, fs_hz = samplerate_hz, cutoff_hz = 0.55*chiprate_hz, taps = 129)

plt_time_fft(proc_samples, sample_rate_hz=samplerate_hz, title_prefix="After filtering")

test_win = proc_samples[10*gc_len*ideal_input_sps: 20*gc_len*ideal_input_sps]
max_freq_dev_hz = 1000
freq_step_hz     = 100
peak_thresh      = 0.05
gc_found, fine_cfo_hz = normalized_gc_search(
    test_win, GCs[gc_n], samplerate_hz, ideal_input_sps,
    max_freq_dev_hz, freq_step_hz, peak_thresh
)
logger.info(f"GC found? {gc_found} | fine CFO: {fine_cfo_hz:+.1f} Hz")
proc_samples = mix(proc_samples, fine_cfo_hz, samplerate_hz)

test_win = proc_samples[10*gc_len*ideal_input_sps : 20*gc_len*ideal_input_sps]
max_freq_dev_hz = 1000
freq_step_hz     = 100
peak_thresh      = 0.05
gc_found, fine_cfo_hz = normalized_gc_search(
    test_win, GCs[gc_n], samplerate_hz, ideal_input_sps,
    max_freq_dev_hz, freq_step_hz, peak_thresh
)
logger.info(f"Second pass: GC found? {gc_found} | fine CFO: {fine_cfo_hz:+.1f} Hz")

first_peak_ind, actual_inpt_sps, num_peaks = get_start_ind_sps(proc_samples, GCs[0], ideal_input_sps, peak_threshold=0.1)

logger.info(f"first_peak_ind, actual_inpt_sps, num_peaks: {first_peak_ind}, {actual_inpt_sps}, {num_peaks}")


#AGC 
proc_samples = rms_agc(proc_samples, target_rms=1.0, attack=0.02, decay=0.002)

plt_time_fft(proc_samples, sample_rate_hz=samplerate_hz, title_prefix="After AGC")


# 4) Carrier recovery (BPSK Costas) with BL ≈ 0.5–1% Rc
proc_samples = costas_loop_bpsk(proc_samples, samplerate_hz, loop_bw_hz=0.008*chiprate_hz, zeta=0.707)

plt_time_fft(proc_samples, sample_rate_hz=samplerate_hz, title_prefix="After costas loop")

plt.figure()
plt.plot(np.real(proc_samples), np.imag(proc_samples), '.')
plt.title("constellation after costas loop")
plt.grid(True)


# 5) Timing recovery: go to exactly 2 sps/chip then Gardner
up, down = Fraction((2.0*chiprate_hz)/samplerate_hz).limit_denominator(1000).numerator, \
           Fraction((2.0*chiprate_hz)/samplerate_hz).limit_denominator(1000).denominator
proc_samples= signal.resample_poly(proc_samples, up, down)
samplerate_hz = int(2.0*chiprate_hz)
proc_samples = gardner_2sps(proc_samples, gain=0.02)  

plt_time_fft(proc_samples, sample_rate_hz=samplerate_hz, title_prefix="After Gardner")
plt.figure()
plt.plot(np.real(proc_samples), np.imag(proc_samples), '.')
plt.title("constellation after Gardner")
plt.grid(True)


#Matched filter
chips_cx = integrate_and_dump(proc_samples, sps=1)    # real valued ~ per chip
chip_bits = (np.real(chips_cx) > 0).astype(np.int8)

plt_time_fft(proc_samples, sample_rate_hz=samplerate_hz, title_prefix="Chip bits")

 


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
