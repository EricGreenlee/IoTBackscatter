import logging
import numpy as np
import matplotlib.pyplot as plt

# enable logging
logger = logging.getLogger("analysis") 
logger.setLevel(logging.DEBUG)

# configurations
sample_rate_hz = 1e6


# import samples from specified file
fname = "local_samples/usrp_n210_20250911_113432_915MHz_1.000Msps_50.0dB_100000samps.npy" #no transmission

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

# compute fft
fft = np.fft.fftshift(np.fft.fft(samples))
fft_db = 10*np.log10(np.abs(fft)**2)
freqs_hz = np.fft.fftshift(np.fft.fftfreq(len(samples), 1/sample_rate_hz))

# plots
plt.figure()
plt.plot(freqs_hz/1e3,np.abs(fft))
# plt.plot(freqs_hz/1e3,fft_db)
plt.xlabel("frequency (kHz)")
plt.ylabel("amplitude (db)")
plt.grid("on")

plt.show()
