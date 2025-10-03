import numpy as np
from scipy import signal
from fractions import Fraction
import logging

logger = logging.getLogger("dbpsk")
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

# ---------- Parameters you know ----------
Rs = 12_500.0          # symbol rate (4 cycles @50 kHz => 80us/sym)
Fs_in = 1_000_000.0    # receiver sample rate
Fs_sym = 4*Rs          # 4 sps target => 50 kS/s
LPF1_cut = 10_000.0    # Hz (0.8*Rs ~ 10 kHz)
LPF1_tr  = 5_000.0     # transition width (Hz)
mix_hz   = -62_450.0   # your coarse offset to center the tag
PRE_BITS = 64
PAY_BITS = 16

# Your 64-bit DIFFERENTIAL preamble (0/1: 1=flip, 0=no flip)
# (Replace with your actual differential pattern)
dbpsk_preamble = np.array([
    1,0,1,0,0,0,0,1, 1,1,1,0, 1,1,0,0,
    0,1,1,1,1,0,0,1, 1,0,1,1, 0,0,0,1,
    1,1,0,0,1,1,1,1, 0,0,1,0, 1,0,0,1,
    1,0,1,1,1,0,0,1, 0,1,1,0, 1,0,0,1
], dtype=np.int8)
dbpsk_payload_tag = np.array([
    1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0
    ], dtype=np.int8)


# ---------- Helpers ----------
def mix_to_baseband(x: np.ndarray, fs: float, freq_hz: float) -> np.ndarray:
    n = np.arange(x.size, dtype=np.float64)
    return x * np.exp(-1j*2*np.pi*freq_hz*n/fs)

def lpf_fir(x: np.ndarray, fs: float, cutoff: float, transition: float, taps: int | None = None) -> np.ndarray:
    # Hamming; length ~ 4*fs/transition if not specified
    if taps is None:
        taps = int(np.ceil(4*fs/transition)) | 1
    h = signal.firwin(taps, cutoff, fs=fs)
    return signal.lfilter(h, 1.0, x)

def resample_to_rate(x: np.ndarray, fs_in: float, fs_out: float) -> tuple[np.ndarray, float]:
    frac = Fraction(fs_out, fs_in).limit_denominator(1000)
    up, down = frac.numerator, frac.denominator
    y = signal.resample_poly(x, up=up, down=down, window=('kaiser', 8.6))
    return y, fs_out

def rms_agc(x: np.ndarray, target_rms: float=1.0, attack: float=0.02, decay: float=0.002) -> np.ndarray:
    env = 0.0
    y = np.empty_like(x, dtype=np.complex64)
    for i, xi in enumerate(x):
        m2 = xi.real*xi.real + xi.imag*xi.imag
        a = attack if m2 > env else decay
        env = (1-a)*env + a*m2
        g = target_rms/np.sqrt(env + 1e-12)
        y[i] = xi * g
    return y

def i_and_d_byN_at_phase(x: np.ndarray, N: int, ph: int) -> np.ndarray:
    # integrate-and-dump groups of N starting at offset ph
    start = ph
    M = (x.size - start)//N
    block = x[start:start + M*N].reshape(M, N)
    return block.sum(axis=1)

def dpsk_diff_detect(sym: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # d[n] = y[n]*conj(y[n-1]); bit=1 if Re(d)<0 else 0
    d = sym[1:] * np.conj(sym[:-1])
    bits = (np.real(d) < 0).astype(np.int8)
    return bits, d

def bits_to_pm1(b):                 # 0 -> -1, 1 -> +1 (for correlation)
    a = np.asarray(b, dtype=np.int8).ravel()
    return (2*a - 1).astype(np.int8)

# ---------- Pipeline ----------
def dbpsk_demod(samples: np.ndarray):
    # 1) coarse mix
    x = mix_to_baseband(samples, Fs_in, mix_hz)
    # 2) LPF & decimate to 4 sps
    x = lpf_fir(x, Fs_in, cutoff=LPF1_cut, transition=LPF1_tr, taps=257)
    x, fs2 = resample_to_rate(x, Fs_in, Fs_sym)  # fs2 ~ 4*Rs
    # 3) optional AGC
    x = rms_agc(x, target_rms=1.0)

    # 4) try 4 symbol phases; correlate differential preamble
    tpl = bits_to_pm1(dbpsk_preamble).astype(np.int16)
    best = None
    for ph in range(4):
        sym = i_and_d_byN_at_phase(x, N=4, ph=ph).astype(np.complex64)
        bits, d = dpsk_diff_detect(sym)  # 0/1 differential bits
        s = bits_to_pm1(bits).astype(np.int16)
        corr = signal.correlate(s, tpl, mode='valid')
        k = int(np.argmax(np.abs(corr)))
        peak = int(corr[k]); metric = abs(peak)
        logger.info(f"phase {ph}: preamble corr peak={peak:+d}, |peak|={metric:d}, @sym={k}")
        if (best is None) or (metric > best['metric']):
            best = dict(phase=ph, sym=sym, bits=bits, k=k, peak=peak, metric=metric)

    ph, sym, bits, k = best['phase'], best['sym'], best['bits'], best['k']
    logger.info(f"BEST: phase={ph}, preamble start @sym={k}, |peak|={best['metric']}")

    # 5) extract preamble & payload (differential bits)
    pre_rx = bits[k:k+PRE_BITS]
    if pre_rx.size < PRE_BITS:
        pre_rx = np.pad(pre_rx, (0, PRE_BITS-pre_rx.size), constant_values=0)
    pay_rx = bits[k+PRE_BITS : k+PRE_BITS+PAY_BITS]
    if pay_rx.size < PAY_BITS:
        pay_rx = np.pad(pay_rx, (0, PAY_BITS-pay_rx.size), constant_values=0)

    # Hamming match for sanity
    hd = np.count_nonzero(pre_rx[:PRE_BITS] != dbpsk_preamble[:PRE_BITS])
    match = 1.0 - hd/PRE_BITS
    logger.info(f"Preamble Hamming match: {match*100:.1f}%")
    logger.info(f"Payload bits ({PAY_BITS}): {pay_rx.tolist()}")

    return dict(phase=ph, preamble_bits=pre_rx, payload_bits=pay_rx, start=k)

# ---------- Usage ----------
fname = "local_samples/usrp_n210_20251003_143438_915MHz_1.000Msps_38.0dB_1000000samps.npy"

samples = np.load(fname)  # complex64 baseband IQ
out = dbpsk_demod(samples)
print(out["payload_bits"])
