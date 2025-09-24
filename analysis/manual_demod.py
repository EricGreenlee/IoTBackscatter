import logging
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import find_peaks

# Goldcodes
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

#
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
    peaks, _ = find_peaks(fft_db, height=peak_threshold, prominence=5, distance=10)
    
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

def compute_autocorr(samples, plot= False, threshold=None):
    autocorrelation = signal.correlate(samples, samples, mode="full")
    autocorr_mag = np.abs(autocorrelation)

    if threshold is None:
        threshold = np.mean(autocorr_mag) + 5 * np.std(autocorr_mag)
        
    logger.info(f"Peak threshold: {threshold}")
        
    peaks, properties = find_peaks(autocorr_mag, height= threshold)
    
    lags = np.arange(len(proc_samples)*2-1)-len(proc_samples)+1
    peak_lags = lags[peaks]
    peak_values = autocorr_mag[peaks]
    
    mean_peak_dif = 0
    sd_peak_dif = 0
    
    # calculate and log differences between consecutive peaks
    if len(peaks) > 1:
        peak_diffs = np.diff(peaks)
        logger.info(f"\nPeak index differences:")
        for i, diff in enumerate(peak_diffs):
            logger.info(f"Peak {i+1} to Peak {i+2}: {diff} samples")
        
        if len(peak_diffs) > 0:
            logger.info(f"Mean difference: {np.mean(peak_diffs):.1f} samples")
            logger.info(f"Std deviation: {np.std(peak_diffs):.1f} samples")
            mean_peak_dif = np.mean(peak_diffs)
            sd_peak_dif = np.std(peak_diffs)


    
    
    # plot I and Q components if complex
    if plot:
        # plot
        plt.figure(figsize=(12, 6))
        # plt.subplot(2, 1, 1)
        # # plt.plot(time_ms, np.real(samples), 'b-', alpha=0.7, label='I (Real)')
        # # plt.plot(time_ms, np.imag(samples), 'r-', alpha=0.7, label='Q (Imag)')
        # plt.plot(lags, np.real(samples), 'b-', alpha=0.7, label='I (Real)')
        # plt.plot(lags, np.imag(samples), 'r-', alpha=0.7, label='Q (Imag)')
        # plt.plot(peak_lags, np.real(samples[peaks]), 'go', markersize=6, label=f'Peaks ({len(peaks)} found)')
        # plt.ylabel("Amplitude")
        # plt.legend()
        # plt.grid(True)
        # plt.title(f"Autocorrelation Time Domain - I/Q Components")
        
        # # plot magnitude with peaks
        # plt.subplot(2, 1, 2)
        plt.plot(lags, autocorr_mag, 'k-', alpha=0.8, label='Magnitude')
        plt.axhline(y=threshold, color='orange', linestyle=':', label=f'Peak Threshold ({threshold:.4f})')
        plt.plot(peak_lags, peak_values, 'ro', markersize=8, label=f'Peaks ({len(peaks)} found)')
        plt.ylabel("Magnitude")
        plt.xlabel("Lags")
        plt.legend()
        plt.grid(True)
        plt.title(f"Autocorrelation Time Domain - Magnitude with Peaks")

    return 0, mean_peak_dif, sd_peak_dif

def plt_time_peaks(samples, sample_rate_hz, title_prefix="", peak_height=None, prominence=0.1, distance=10):
    """
    Find and plot peaks in time domain signal
    
    Args:
        samples: Input signal samples
        sample_rate_hz: Sample rate in Hz
        title_prefix: Optional prefix for plot title
        peak_height: Minimum peak height (default: auto-calculated)
        prominence: Required prominence of peaks
        distance: Minimum distance between peaks in samples
    """
    # compute magnitude for peak detection
    magnitude = np.abs(samples)
    
    # auto-calculate peak height if not provided
    if peak_height is None:
        peak_height = np.mean(magnitude) + 2 * np.std(magnitude)
    
    # find peaks
    peaks, properties = find_peaks(magnitude, height=peak_height, prominence=prominence, distance=distance)
    
    # create time array
    # time_ms = np.arange(len(samples)) / sample_rate_hz * 1000
    # peak_times_ms = time_ms[peaks]
    # peak_values = magnitude[peaks]
    
    lags = np.arange(len(samples))
    peak_lags = lags[peaks]
    peak_values = magnitude[peaks]
    
    # log results
    logger.info(f"\nTime Domain Peak Analysis - {title_prefix}:")
    logger.info(f"Peak height threshold: {peak_height:.4f}")
    logger.info(f"Found {len(peaks)} peaks")
    
    for i, (time, value) in enumerate(zip(peak_lags, peak_values)):  # show first 10
        logger.info(f"Peak {i+1}: {time} lag, magnitude: {value:.4f}")
    
    # calculate and log differences between consecutive peaks
    if len(peaks) > 1:
        peak_diffs = np.diff(peaks)
        logger.info(f"\nPeak index differences:")
        for i, diff in enumerate(peak_diffs):
            logger.info(f"Peak {i+1} to Peak {i+2}: {diff} samples")
        
        if len(peak_diffs) > 0:
            logger.info(f"Mean difference: {np.mean(peak_diffs):.1f} samples")
            logger.info(f"Std deviation: {np.std(peak_diffs):.1f} samples")
    
    # plot
    plt.figure(figsize=(12, 6))
    
    # plot I and Q components if complex
    if np.iscomplexobj(samples):
        plt.subplot(2, 1, 1)
        # plt.plot(time_ms, np.real(samples), 'b-', alpha=0.7, label='I (Real)')
        # plt.plot(time_ms, np.imag(samples), 'r-', alpha=0.7, label='Q (Imag)')
        plt.plot(lags, np.real(samples), 'b-', alpha=0.7, label='I (Real)')
        plt.plot(lags, np.imag(samples), 'r-', alpha=0.7, label='Q (Imag)')
        plt.plot(peak_lags, np.real(samples[peaks]), 'go', markersize=6, label=f'Peaks ({len(peaks)} found)')
        plt.ylabel("Amplitude")
        plt.legend()
        plt.grid(True)
        plt.title(f"{title_prefix}Time Domain - I/Q Components")
        
        # plot magnitude with peaks
        plt.subplot(2, 1, 2)
        plt.plot(lags, magnitude, 'k-', alpha=0.8, label='Magnitude')
        plt.axhline(y=peak_height, color='orange', linestyle=':', label=f'Peak Threshold ({peak_height:.4f})')
        plt.plot(peak_lags, peak_values, 'ro', markersize=8, label=f'Peaks ({len(peaks)} found)')
        plt.ylabel("Magnitude")
        plt.xlabel("Lags")
        plt.legend()
        plt.grid(True)
        plt.title(f"{title_prefix}Time Domain - Magnitude with Peaks")
    else:
        # real signal - single plot
        plt.plot(lags, samples, 'b-', alpha=0.8, label='Signal')
        plt.axhline(y=peak_height, color='orange', linestyle=':', label=f'Peak Threshold ({peak_height:.4f})')
        plt.plot(peak_lags, samples[peaks], 'ro', markersize=8, label=f'Peaks ({len(peaks)} found)')
        plt.ylabel("Amplitude")
        plt.xlabel("Lags")
        plt.legend()
        plt.grid(True)
        plt.title(f"{title_prefix}Time Domain with Peaks")
    
    plt.tight_layout()
    
    return peaks, peak_lags, peak_values, properties 

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

#Read file
fname = "../python/src/cloud_samples/usrp_n210_20250919_130428_915MHz_1.000Msps_50.0dB_4000000samps.npy" #L053R8 tag (better clock + buffer), 50khz carrier, 80us per bit, proper packet 
samplerate_hz = 1000000
datarate_hz = 25000

try:
    samples = np.load(fname)
    logger.info(f"Loaded {len(samples)} samples from {fname}")
except FileNotFoundError:
    logger.error(f"File not found: {fname}")
    raise
except Exception as e:
    logger.error(f"Error loading samples from {fname}: {e}")
    raise# compute and show metadata
nsamps = len(samples)
dur_ms = nsamps/samplerate_hz*1000

# logger.info(f"\nSample Analysis:")
logger.info(f"Number of samples: {nsamps}")
logger.info(f"Sample rate: {samplerate_hz/1e6:.2f} MS/s")
logger.info(f"Duration: {dur_ms:.2f} ms")

#Plot time and freq display
# plt_time_fft(samples, samplerate_hz, title_prefix="Raw Input: ")

#manually select samples
# proc_samples = samples[1300000:2200000]
proc_samples = samples[1400000:1500000]

#freq shift
time_array_sec = np.linspace(0,(len(proc_samples)-1)/samplerate_hz,len(proc_samples))
mix_freq_hz = -124.5e3
proc_samples = proc_samples*np.exp(-1j*2*np.pi*mix_freq_hz*time_array_sec)

#filter and agc
proc_samples = lpf(proc_samples, datarate_hz/2, samplerate_hz)
proc_samples = agc(proc_samples, 1/np.sqrt(2))

# plt_time_fft(proc_samples, samplerate_hz, title_prefix="Mixed, filtered, and AGCd: ")

#autocorrelate to find resampling rate and resample
peak_autocorr_val, autocorr_mean_sample_spacing, autocorr_sd_sample_spacing = compute_autocorr(proc_samples, plot= True)#, threshold = 1000)
target_sps = 80
resamp_rate = target_sps*len(GCs[0])/autocorr_mean_sample_spacing
logger.info(f"Resamp_rate: {resamp_rate}")
proc_samples = signal.resample_poly(proc_samples, int(resamp_rate*10000), 10000)
new_time_array_sec = np.linspace(0,(len(proc_samples)-1)/samplerate_hz,len(proc_samples))
peak_autocorr_val, autocorr_mean_sample_spacing, autocorr_sd_sample_spacing = compute_autocorr(proc_samples, plot= True)


#coarse frequency correction
gc_at_target_sps = np.repeat(GCs[0].astype(np.complex64), target_sps)
# freq_array_hz = np.linspace(-1000, 1000, 11)
# freq_array_hz = np.linspace(200, 400, 5)
# freq_array_hz = [250]

# for freq_hz in freq_array_hz :
    
#     mixed_samples = proc_samples*np.exp(-1j*2*np.pi*freq_hz*new_time_array_sec)

#     gc_corr = signal.correlate(mixed_samples, gc_at_target_sps, mode="full")

#     plt_time_peaks(gc_corr, samplerate_hz, f"correlation with goldcode at {freq_hz} Hz", peak_height=2000, distance = 1000)

coarse_freq_offset_hz = 250

#cross correlate with preamble to find packet start
proc_samples = samples[1300000:2200000]


#freq shift
time_array_sec = np.linspace(0,(len(proc_samples)-1)/samplerate_hz,len(proc_samples))
# mix_freq_hz = -124.5e3
proc_samples = proc_samples*np.exp(-1j*2*np.pi*mix_freq_hz*time_array_sec)

#filter and agc
proc_samples = lpf(proc_samples, datarate_hz/2, samplerate_hz)
proc_samples = agc(proc_samples, 1/np.sqrt(2))

plt_time_fft(proc_samples, samplerate_hz, title_prefix="Mixed, filtered, and AGCd: ")

proc_samples = signal.resample_poly(proc_samples, int(resamp_rate*10000), 10000)
new_time_array_sec = np.linspace(0,(len(proc_samples)-1)/samplerate_hz,len(proc_samples))


gc_at_target_sps = np.repeat(GCs[0].astype(np.complex64), target_sps)
proc_samples = proc_samples*np.exp(-1j*2*np.pi*coarse_freq_offset_hz*new_time_array_sec)

gc_corr = signal.correlate(proc_samples, gc_at_target_sps, mode="valid")

plt_time_peaks(gc_corr, samplerate_hz, f"correlation with goldcode", peak_height=2000, distance = 1000)

sim_pre = np.array([])
for i in range(64):
    bit = preamble[i]
    bit_bpsk = 2 * bit - 1  # Convert 0->-1, 1->+1
    modulated_chip_seq = GCs[0] * bit_bpsk
    upsampled_seq = np.repeat(modulated_chip_seq.astype(np.complex64), target_sps)
    sim_pre = np.concatenate([sim_pre, upsampled_seq])
         
pre_corr = signal.correlate(proc_samples, sim_pre, mode="valid")

plt_time_peaks(pre_corr, samplerate_hz, f"correlation with preamble", peak_height=2000, distance = 1000)

max_preamble_corr_val= max(np.abs(pre_corr))
max_preamble_corr_ind = np.argmax(np.abs(pre_corr))
logger.info(f"max_preamble_corr_val: {max_preamble_corr_val}, max_preamble_corr_ind: {max_preamble_corr_ind} ")

test_samples = proc_samples[max_preamble_corr_ind-240-10160: max_preamble_corr_ind-240+79*10160]
proc_samples = proc_samples[max_preamble_corr_ind-10160: max_preamble_corr_ind+79*10160]

# plt_time_fft(test_samples, samplerate_hz, title_prefix = "Unaligned Signal: ")
# plt_time_fft(proc_samples, samplerate_hz, title_prefix = "Aligned Signal: ")


#despread and downsample

bits_to_calc = 80
logger.debug("Expected number of bits: %s", bits_to_calc)
samps_per_bit = target_sps*127
despread_samples = np.zeros(samps_per_bit*bits_to_calc).astype(np.complex64)
wrong_despread_samples = np.zeros(samps_per_bit*bits_to_calc).astype(np.complex64)

logger.info(f"samps_per_bit: {samps_per_bit}")
        
for bit in range(bits_to_calc):
    # print(bit)
    despread_samples[bit*samps_per_bit:(bit+1)*samps_per_bit] = proc_samples[bit*samps_per_bit:(bit+1)*samps_per_bit]*gc_at_target_sps#np.repeat(itagParams.goldcode.astype(np.complex64),target_sps)
    wrong_despread_samples[bit*samps_per_bit:(bit+1)*samps_per_bit] = test_samples[bit*samps_per_bit:(bit+1)*samps_per_bit]*gc_at_target_sps

plt_time_fft(despread_samples, samplerate_hz, title_prefix = "Despread Samples: ")
proc_samples=despread_samples
# plt_time_fft(wrong_despread_samples, samplerate_hz, title_prefix = "Wrong Despread Samples: ")

# proc_samples = lpf(despread_samples, datarate_hz/127*2, samplerate_hz)
# proc_samples = signal.resample_poly(proc_samples, 1, 127)

# plt_time_fft(proc_samples, samplerate_hz, title_prefix = "Downsampled Samples: ")


#mm timing recovery
proc_samples = mm_time_recovery(proc_samples, target_sps)
plt_time_fft(proc_samples, samplerate_hz, title_prefix = "MM Timing Recovery: ")

#costas loop
# proc_samples = costas_loop(proc_samples, samplerate_hz/127)
proc_samples = costas_loop(proc_samples, samplerate_hz)
plt_time_fft(proc_samples, samplerate_hz, title_prefix = "Costas Loop: ")

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

logger.info(f"proc_bits: {proc_bits}")
payload_bits = proc_bits[64:80]

logger.info(f"sent payload bits: {sent_payload}")
logger.info(f"received payloadbits: {payload_bits}")
n_errors = np.sum(np.abs(sent_payload-payload_bits))
logger.info(f"# errors: {n_errors}")
# logger.info

plt.show()