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

def mix_filt_DCblock_AGC(in_samples, mix_freq_hz, samplerate_hz, cutoff_freq_hz):
    #freq shift
    time_array_sec = np.linspace(0,(len(in_samples)-1)/samplerate_hz,len(proc_samples))
    mid_samples = proc_samples*np.exp(-1j*2*np.pi*mix_freq_hz*time_array_sec)
    
    #filter
    mid_samples = lpf_fir(mid_samples, cutoff_freq_hz, samplerate_hz)
    
    #dc block
    mid_samples = dc_block(mid_samples)
    
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
fname = "../python/src/cloud_samples/usrp_n210_20250919_130428_915MHz_1.000Msps_50.0dB_4000000samps.npy" #L053R8 tag (better clock + buffer), 50khz carrier, 80us per bit, proper packet 


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
# num_loops = int(np.ceil(len(samples)/num_samps_jump_first_loop))
num_loops = 80

logger.info(f"num_samps_first_loop: {num_samps_window_first_loop}")
logger.info(f"number of loops: {num_loops}")

recent_detected_gc = np.ones(num_gcs)
gc_detected = np.zeros([num_loops, num_gcs])

for start_loop_n in range(num_loops):
    #grab specific samples
    start_samp = int(start_loop_n*num_samps_window_first_loop)
    end_samp = int(start_samp + num_samps_window_first_loop)
    logger.info(f"{start_loop_n}-> {start_samp}: {end_samp}")
    
    proc_samples = samples[start_samp:end_samp]
    
    #mix, filter, and AGC
    offset_freq_hz = -124.5e3
    proc_samples = mix_filt_DCblock_AGC(proc_samples, offset_freq_hz, samplerate_hz, chiprate_hz*1.5)

    # plt_time_fft(proc_samples, samplerate_hz)
    
    #test for presence of each goldcode
    for gc_n in range(1):#num_gcs):
        max_freq_dev_hz = 1000
        freq_step_hz = 200
        peak_thresh = 3000
        peak_det, freq_adj_hz = gc_search(proc_samples, GCs[gc_n], samplerate_hz, target_sps,  max_freq_dev_hz, freq_step_hz, peak_thresh)

        logger.info(f"recent_detected_gc: {recent_detected_gc}")

        if peak_det and recent_detected_gc[gc_n] == 0:
            logger.info("demodding")
            # gc_detected[start_loop_n, gc_n] = 1
            # break
        gc_detected[start_loop_n, gc_n] = peak_det
        recent_detected_gc[gc_n] = peak_det
    
logger.info("gc detected:\n%s", np.array2string(gc_detected, threshold=np.inf, max_line_width=np.inf))

    
    # plt.show()