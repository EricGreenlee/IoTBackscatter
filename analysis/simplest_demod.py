
#imports
import logging
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import scipy.signal as signal
from fractions import Fraction

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
    
def mix(in_samples, mix_freq_hz, samplerate_hz):
    time_array_sec = np.linspace(0,(len(in_samples)-1)/samplerate_hz,len(in_samples))
    out_samples = in_samples*np.exp(-1j*2*np.pi*mix_freq_hz*time_array_sec)
    
    return out_samples

def lpf_fir(x, fs, cutoff, transition, taps=None):
    # Hamming; length ~ 4*fs/transition is a decent rule if taps not given.
    if taps is None:
        taps = int(np.ceil(4*fs/transition)) | 1  # make it odd
    h = signal.firwin(taps, cutoff, fs=fs)
    return signal.lfilter(h, 1.0, x), h

def decimate_fir(x, fs, q):
    # Use resample_poly as an efficient FIR downsampler with built-in anti-alias filter
    y = signal.resample_poly(x, up=1, down=q, window=('kaiser', 8.6))
    return y, fs/q

#read signal
fname = "local_samples/usrp_n210_20251003_143438_915MHz_1.000Msps_38.0dB_1000000samps.npy"

try:
    samples = np.load(fname)
    logger.info(f"Loaded {len(samples)} samples from {fname}")
except FileNotFoundError:
    logger.error(f"File not found: {fname}")
    raise
except Exception as e:
    logger.error(f"Error loading samples from {fname}: {e}")
    
samplerate_hz = 1e6

PRE_BITS = 64
PAYLOAD_BITS = 16
PREAMBLE_TEMPLATE_BITS = np.ones(PRE_BITS, dtype=np.int8)  # all '1's for DBPSK
    
plt_time_fft(samples, sample_rate_hz = samplerate_hz, title_prefix= "Raw input: ")
    
    
#mix to center frequency
manual_mix_freq_hz = -62.45e3
proc_samples = mix(samples, samplerate_hz=samplerate_hz, mix_freq_hz=manual_mix_freq_hz)

plt_time_fft(proc_samples, sample_rate_hz = samplerate_hz, title_prefix= f"Mixed samples to {manual_mix_freq_hz}: ")

#lpf and decimation
# 2A) Coarse LPF (10 kHz), then decimate by 50 -> 20 kS/s
x1, _ = lpf_fir(proc_samples, samplerate_hz, cutoff=10_000, transition=5_000, taps=257)
x2, fs2 = decimate_fir(x1, samplerate_hz, q=20)

plt_time_fft(x2, sample_rate_hz = fs2, title_prefix= f"First filtered samples: ")


# 2B) Narrow LPF (250 Hz) for SNR measurement
# x3, _ = lpf_fir(x2, fs2, cutoff=250, transition=200, taps=401)

# plt_time_fft(x3, sample_rate_hz = fs2, title_prefix= f"Second filtered samples: ")

# #Compute SNR
# # 3A) Time-domain SNR using a simple noise reference
# # Make a high-pass version at 5000 Hz to estimate noise
# b_hp = signal.firwin(401, 600, fs=fs2, pass_zero=False)
# x_hp = signal.lfilter(b_hp, 1.0, x2)  # use pre-narrow-LPF stream to avoid bias

# # Choose a stable window (e.g., middle 0.5 s)
# N = len(x3)
# i0, i1 = int(0.25*N), int(0.75*N)
# Psig = np.mean(np.abs(x3[i0:i1])**2)
# Pnoi = np.mean(np.abs(x_hp[i0:i1])**2) + 1e-20
# snr_db_time = 10*np.log10(Psig/Pnoi)
# print(f"SNR (time-domain): {snr_db_time:.1f} dB")

# # 3B) Spectral SNR around DC vs off-bands (Welch)
# f, Pxx = signal.welch(x2[i0:i1], fs2, nperseg=4096, return_onesided=False, scaling='spectrum')
# # shift to center DC
# ordr = np.argsort(f); f, Pxx = f[ordr], Pxx[ordr]
# # integrate ±300 Hz
# in_sig = (f >= -300) & (f <= 300)
# in_n1  = (f >= 2000) & (f <= 2600)
# in_n2  = (f >= -2600) & (f <= -2000)
# Psig_band = np.trapz(Pxx[in_sig], f[in_sig])
# Pnoi_band = 0.5*(np.trapz(Pxx[in_n1], f[in_n1]) + np.trapz(Pxx[in_n2], f[in_n2])) + 1e-20
# snr_db_spec = 10*np.log10(Psig_band/Pnoi_band)
# print(f"SNR (spectral):   {snr_db_spec:.1f} dB")

def rms_agc(x, target_rms=1.0, attack=0.02, decay=0.002):
    env = 0.0
    y = np.empty_like(x, dtype=np.complex64)
    for i, xi in enumerate(x.astype(np.complex64)):
        mag2 = (xi.real*xi.real + xi.imag*xi.imag)
        a = attack if mag2 > env else decay
        env = (1 - a)*env + a*mag2
        y[i] = xi * (target_rms / np.sqrt(env + 1e-12))
    return y

x3 = rms_agc(x2, target_rms=1.0)

# === 3) Matched filter at 4 sps = boxcar of 4 ===
# We'll produce 4 "phase streams" (offsets 0..3), each integrate-and-dump by 4.
def i_and_d_by4_at_phase(x, phase):
    xph = x[phase:]
    Nsym = (len(xph) // 4)
    use = xph[:Nsym*4]
    sym = use.reshape(-1, 4).sum(axis=1)   # complex, matched filter
    return sym

# === 4) Differential detector (DBPSK) ===
# bit 1 ↔ π flip → Re{d[n]} < 0 ; bit 0 ↔ Re{d[n]} > 0
def dpsk_diff_detect(sym):
    d = sym[1:] * np.conj(sym[:-1])     # complex
    bits = (np.real(d) < 0).astype(np.int8)
    return bits, d

# === 5) Sliding preamble search over 4 phases; pick best ===
# Map bits -> {+1,-1} so we can correlate with a {+1/-1} template.
def bits_to_pm1(bits):  # 0→-1, 1→+1
    a = np.asarray(bits).astype(np.int8).reshape(-1)  # force flat numeric array
    return (2*a - 1).astype(np.int8)

def dbpsk_to_bpsk(d_seq, init_symbol=1):
    """Convert differential bits (0=no flip, 1=flip) to BPSK ±1 symbols"""
    bpsk = []
    sym = init_symbol
    for b in d_seq:
        if b == 1:
            sym = -sym   # flip
        # if b == 0: sym stays the same
        bpsk.append(sym)
    return bpsk

dbpsk_preamble_tag = np.array([1,0,1,0,0,0,0,1,1,1,1,0,1,1,0,0,
              0,1,1,1,1,0,0,1,1,0,1,1,0,0,0,1,
              1,1,0,0,1,1,1,1,0,0,1,0,1,0,0,1,
              1,0,1,1,1,0,0,1,0,1,1,0,1,0,0,1], dtype=np.int8)

bpsk_preamble_tag = dbpsk_to_bpsk(dbpsk_preamble_tag, init_symbol=1)
print(f"Preamble bits undifferentiated: {bpsk_preamble_tag}")
template_pm1 = dbpsk_preamble_tag*2-1
# bin_bpsk_preamble_tag = bpsk_preamble_tag*2-1
# print(bin_bpsk_preamble_tag)

dbpsk_payload_tag = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=np.int8);
bpsk_payload = dbpsk_to_bpsk(dbpsk_preamble_tag )
print(f"Payload bits undifferentiated: {bpsk_payload}")

# plt.show()

# template_pm1 = bits_to_pm1(PREAMBLE_TEMPLATE_BITS)  # all -1's for all-1 preamble


best = None
for ph in range(4):
    sym = i_and_d_by4_at_phase(x3, ph)
    bits, d = dpsk_diff_detect(sym)  # 'bits' length = len(sym)-1

    # Sliding correlation with template (valid positions only)
    s = bits_to_pm1(bits).astype(np.int16)
    corr = signal.correlate(s, template_pm1, mode='valid')
    k = np.argmax(np.abs(corr))
    peak = corr[k]
    metric = np.abs(peak)

    logger.info(f"Phase {ph}: preamble corr peak={peak}, |peak|={metric}, @sym={k}")

    if (best is None) or (metric > best['metric']):
        best = dict(phase=ph, sym=sym, bits=bits, corr=corr, k=k, metric=metric)

# === 6) Extract payload at best phase/position ===
ph   = best['phase']
sym  = best['sym']
bits = best['bits']
k    = best['k']         # preamble starts at bits index k
logger.info(f"Best phase={ph}, preamble start sym index={k}")

# The differential 'bits' index corresponds to transitions between sym[n-1]->sym[n].
# Preamble uses 64 bits → spans bits[k : k+64]
pre_rx = bits[k : k + PRE_BITS]
if len(pre_rx) < PRE_BITS:
    logger.warning("Not enough symbols for preamble—truncate")
    pre_rx = np.pad(pre_rx, (0, PRE_BITS - len(pre_rx)), constant_values=0)

# Payload follows immediately after preamble
pay_start = k + PRE_BITS
pay_stop  = pay_start + PAYLOAD_BITS
payload_rx = bits[pay_start : pay_stop]
if len(payload_rx) < PAYLOAD_BITS:
    logger.warning("Not enough symbols for full payload—truncate")
    payload_rx = np.pad(payload_rx, (0, PAYLOAD_BITS - len(payload_rx)), constant_values=0)

logger.info(f"Preamble bits: {pre_rx}")
logger.info(f"Payload bits (len={len(payload_rx)}): {payload_rx}")

# === Optional: quick plots to verify ===
# Constellation pre- and post-diff around the detected preamble
import matplotlib.pyplot as plt
win0 = slice(max(0, k-40), min(len(sym), k+PRE_BITS+40))
plt.figure()
plt.plot(np.real(sym[win0]), np.imag(sym[win0]), '.', alpha=0.5)
plt.title(f"Constellation near preamble (phase={ph})")
plt.xlabel("I"); plt.ylabel("Q"); plt.grid(True)

plt.figure()
d_best = sym[1:] * np.conj(sym[:-1])
plt.plot(np.real(d_best[win0]), np.imag(d_best[win0]), '.', alpha=0.5)
plt.title("Differential constellation d[n]=y[n]*conj(y[n-1])")
plt.xlabel("Re"); plt.ylabel("Im"); plt.grid(True)

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