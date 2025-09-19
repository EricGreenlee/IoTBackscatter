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


# enable logging
logger = logging.getLogger("analysis") 
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(levelname)s: %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)



def plt_fft(samples, sample_rate_hz, title_prefix="", peak_threshold = 0):
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
    plt.plot(freqs_hz/1e3, fft_db, label='FFT Magnitude')
    plt.axhline(y=peak_threshold, color='orange', linestyle=':', label=f'Peak Threshold ({peak_threshold:.1f} dB)')
    
    # mark detected peaks
    if len(peaks) > 0:
        plt.plot(peak_freqs_hz/1e3, peak_amplitudes_db, 'ro', markersize=8, label=f'Peaks ({len(peaks)} found)')
    
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
    
    lags = np.arange(len(samples))-len(samples)/2+.5
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

# configurations
# sample_rate_hz = 25e4
# sample_rate_hz = 100e4
sample_rate_hz = 1000000


# import samples from specified file
# fname = "local_samples/usrp_n210_20250911_113432_915MHz_1.000Msps_50.0dB_100000samps.npy" #no tag transmission
# fname = "local_samples/usrp_n210_20250911_121153_915MHz_1.000Msps_50.0dB_100000samps.npy" #carrier wave from tag
# fname = "local_samples/usrp_n210_20250911_122909_915MHz_1.000Msps_50.0dB_100000samps.npy" #alternating 0's and 1's from tag
# fname = "local_samples/usrp_n210_20250911_124317_915MHz_1.000Msps_50.0dB_100000samps.npy" #repeated goldcode
# fname = "local_samples/usrp_n210_20250912_181047_915MHz_0.250Msps_50.0dB_100000samps.npy"
# fname = "local_samples/usrp_n210_20250912_182742_915MHz_0.248Msps_50.0dB_100000samps.npy" 
# fname = "local_samples/usrp_n210_20250912_183113_915MHz_0.252Msps_50.0dB_100000samps.npy"#sampled to aim for 10 SPS exactly
# fname = "local_samples/usrp_n210_20250912_183807_915MHz_1.003Msps_50.0dB_100000samps.npy"
# fname = "local_samples/usrp_n210_20250912_184742_915MHz_1.000Msps_50.0dB_200000samps.npy"
# fname = "local_samples/usrp_n210_20250915_175847_915MHz_1.000Msps_50.0dB_200000samps.npy" #uses new arduino code that may be more frequency stable, at 75khz
# fname = "local_samples/usrp_n210_20250915_183405_915MHz_1.000Msps_50.0dB_200000samps.npy" #stable freq with modulated signal
# fname = "local_samples/usrp_n210_20250915_184724_915MHz_1.000Msps_50.0dB_1000000samps.npy" #stable freq with modulated signal and sufficient samples
# fname = "local_samples/usrp_n210_20250916_133433_915MHz_1.000Msps_50.0dB_1000000samps.npy" #repeated gc with arduino stable freq
# fname = "local_samples/usrp_n210_20250918_111249_915MHz_1.000Msps_50.0dB_1000000samps.npy" #L053R8 tag, 50khz carrier, 80us per bit, repeated GC
# fname = "local_samples/usrp_n210_20250918_111854_915MHz_1.000Msps_50.0dB_1000000samps.npy" #copy of above, to verify: L053R8 tag, 50khz carrier, 80us per bit, repeated GC
# fname = "local_samples/usrp_n210_20250918_112106_915MHz_1.000Msps_50.0dB_1000000samps.npy" #copy of above
# fname = "local_samples/usrp_n210_20250918_134221_915MHz_1.000Msps_50.0dB_1000000samps.npy" #L053R8 tag, 50khz carrier, 80us per bit, preamble and payload
# fname = "local_samples/usrp_n210_20250918_135918_915MHz_1.000Msps_50.0dB_2000000samps.npy" #L053R8 tag, 50khz carrier, 80us per bit, preamble and payload
# fname = "local_samples/usrp_n210_20250918_141826_915MHz_1.000Msps_50.0dB_2000000samps.npy" #L053R8 tag, 50khz carrier, 80us per bit, preamble and payload all 1's with a break between
# fname = "local_samples/usrp_n210_20250919_125228_915MHz_1.000Msps_50.0dB_4000000samps.npy" #L053R8 tag (better clock + buffer), 50khz carrier, 80us per bit, packet with all 1's
fname = "local_samples/usrp_n210_20250919_130428_915MHz_1.000Msps_50.0dB_4000000samps.npy" #L053R8 tag (better clock + buffer), 50khz carrier, 80us per bit, proper packet 

try:
    samples = np.load(fname)
    logger.info(f"Loaded {len(samples)} samples from {fname}")
except FileNotFoundError:
    logger.error(f"File not found: {fname}")
    raise
except Exception as e:
    logger.error(f"Error loading samples from {fname}: {e}")
    raise

# compute and show metadata
nsamps = len(samples)
dur_ms = nsamps/sample_rate_hz*1000

logger.info(f"\nSample Analysis:")
logger.info(f"Number of samples: {nsamps}")
logger.info(f"Sample rate: {sample_rate_hz/1e6:.2f} MS/s")
logger.info(f"Duration: {dur_ms:.2f} ms")

# Power statistics
power = np.abs(samples) ** 2
avg_power = np.mean(power)
peak_power = np.max(power)

logger.info(f"Average power: {10*np.log10(avg_power):.2f} dB")
logger.info(f"Peak power: {10*np.log10(peak_power):.2f} dB")
logger.info(f"Dynamic range: {10*np.log10(peak_power/avg_power):.2f} dB")

peak_threshold = 0

# analyze original samples
fft, fft_db, freqs_hz, peak_freqs_hz, peak_amplitudes_db = plt_fft(samples, sample_rate_hz, "Original - ")

# zoom plot around a specific frequency range
# zoom_center_khz = -122  # center frequency for zoom (kHz)
# zoom_bandwidth_khz = 100  # bandwidth for zoom (kHz)

# plt.figure()
# zoom_mask = (freqs_hz/1e3 >= zoom_center_khz - zoom_bandwidth_khz/2) & (freqs_hz/1e3 <= zoom_center_khz + zoom_bandwidth_khz/2)
# plt.plot(freqs_hz[zoom_mask]/1e3, fft_db[zoom_mask], label='FFT Magnitude (Zoomed)')

# # show peaks in zoom range
# zoom_peak_mask = (peak_freqs_hz/1e3 >= zoom_center_khz - zoom_bandwidth_khz/2) & (peak_freqs_hz/1e3 <= zoom_center_khz + zoom_bandwidth_khz/2)
# if np.any(zoom_peak_mask):
#     zoom_peak_freqs = peak_freqs_hz[zoom_peak_mask]
#     zoom_peak_amps = peak_amplitudes_db[zoom_peak_mask]
#     plt.plot(zoom_peak_freqs/1e3, zoom_peak_amps, 'ro', markersize=8, label=f'Peaks in range ({np.sum(zoom_peak_mask)} found)')



# plt.axhline(y=peak_threshold, color='orange', linestyle=':', label=f'Peak Threshold ({peak_threshold:.1f} dB)')
# plt.ylabel("amplitude (db)")
# plt.xlabel("frequency (kHz)")
# plt.legend()
# plt.grid(True)
# plt.title(f"Zoomed FFT: {zoom_center_khz} ± {zoom_bandwidth_khz/2} kHz")
# plt.xlim(zoom_center_khz - zoom_bandwidth_khz/2, zoom_center_khz + zoom_bandwidth_khz/2)

#mix, filter, and AGC
proc_samples = samples[1300000:2200000]
# proc_samples = samples
time_array_sec = np.linspace(0,(len(proc_samples)-1)/sample_rate_hz,len(proc_samples))
mix_freq_hz = -124.5e3
# mix_freq_hz = -50e3
data_rate_hz = 12500
# data_rate_hz = 25000



proc_samples = proc_samples*np.exp(-1j*2*np.pi*mix_freq_hz*time_array_sec)

# plt_fft(proc_samples, sample_rate_hz, "Mixed - ")

proc_samples = lpf(proc_samples, data_rate_hz, sample_rate_hz)

# plt_fft(proc_samples, sample_rate_hz, "Filtered- ")

proc_samples = agc(proc_samples, 1/np.sqrt(2))

# plt.figure()
# plt.plot(proc_samples[0:2000])
# plt.grid(True)
# plt.title("time domain of agc'd samples")

plt_fft(proc_samples, sample_rate_hz, "AGC- ",peak_threshold = 80)

plt.figure()
plt.plot( proc_samples)
plt.grid(True)
plt.title("filtered and AGCd signal")

# analyze time domain peaks on processed samples
# plt_time_peaks(proc_samples[0:10000], sample_rate_hz, "Processed - ")  # use first 10k samples for visibility

#autocorrelation
# np_corr = np.correlate(proc_samples, proc_samples, mode="full")

# plt.figure()
# plt.plot(np_corr)
# plt.title("correlation from numpy")
# plt.grid(True)
# plt_time_peaks(np_corr, sample_rate_hz, "correlation from numpy")


scipy_corr = signal.correlate(proc_samples, proc_samples, mode="full")
# plt.figure()
# plt.plot(scipy_corr)
# plt.title("correlation from scipy")
# plt.grid(True)

plt_time_peaks(scipy_corr, sample_rate_hz, "autocorrelation from scipy", peak_height=20000)

# np_corr_rev = np.correlate(proc_samples, proc_samples[::-1], mode="full")

# # plt.figure()
# # plt.plot(np_corr)
# # plt.title("correlation from numpy")
# # plt.grid(True)
# plt_time_peaks(np_corr_rev, sample_rate_hz, "correlation from numpy reversed")


# scipy_corr_rev = signal.correlate(proc_samples, proc_samples[::-1], mode="full")
# # plt.figure()
# # plt.plot(scipy_corr)
# # plt.title("correlation from scipy")
# # plt.grid(True)

# plt_time_peaks(scipy_corr_rev, sample_rate_hz, "correlation from scipy reversed")

# # correlate with simulated goldcode
# sim_gc_sps = {}
# gc_corr = {}

input_sps = 80
target_sps = 80
# target_sps = 40
# actual_sps = 39.7
manual_resamp_rate = 10160/10114#/(input_sps/target_sps)
# manual_resamp_rate = 5080/5045#/(input_sps/target_sps)
proc_samples = signal.resample_poly(proc_samples, int(manual_resamp_rate*10000), 10000)
new_time_array_sec = np.linspace(0,(len(proc_samples)-1)/sample_rate_hz,len(proc_samples))

sim_gc_sps = np.repeat(GCs[0].astype(np.complex64), target_sps)

# freq_array_hz = np.linspace(-1000, 1000, 11)
# # freq_array_hz = np.linspace(100, 300, 5)
# # freq_array_hz = [0]

# for freq_hz in freq_array_hz :
    
#     mixed_samples = proc_samples*np.exp(-1j*2*np.pi*freq_hz*new_time_array_sec)

#     gc_corr = signal.correlate(mixed_samples, sim_gc_sps, mode="full")

#     plt_time_peaks(gc_corr, sample_rate_hz, f"correlation with goldcode at {freq_hz} Hz", peak_height=2000, distance = 1000)

# freq_hz = -800
freq_hz = 250
proc_samples = proc_samples*np.exp(-1j*2*np.pi*freq_hz*new_time_array_sec)

# #correlate with repeated goldcodes
# num_gcs_rep_array = [1,4,16,64]

# for num_gcs_rep in num_gcs_rep_array:
    
#     sim_gc_rep = np.tile(sim_gc_sps, num_gcs_rep)
    
#     gc_corr = signal.correlate(proc_samples, sim_gc_rep, mode="full")

#     plt_time_peaks(gc_corr, sample_rate_hz, f"correlation with {num_gcs_rep} goldcodes ", peak_height=1000*num_gcs_rep, distance = 1000)


#correlate with  preamble
# num_bits_pre_array = [1,4,16,64]
num_bits_pre_array = [1,64]

for num_bits_pre in num_bits_pre_array:
    
    sim_pre = np.array([])
    for i in range(num_bits_pre):
        bit = preamble[i]
        bit_bpsk = 2 * bit - 1  # Convert 0->-1, 1->+1
        modulated_chip_seq = GCs[0] * bit_bpsk
        upsampled_seq = np.repeat(modulated_chip_seq.astype(np.complex64), target_sps)
        sim_pre = np.concatenate([sim_pre, upsampled_seq])
                
    
    gc_corr = signal.correlate(proc_samples, sim_pre, mode="full")

    print(f"num_bits_pre: {num_bits_pre}")
    plt_time_peaks(gc_corr, sample_rate_hz, f"correlation with {num_bits_pre} preamble bits ", peak_height=2000*np.sqrt(num_bits_pre), distance = 1000)



plt.show()
