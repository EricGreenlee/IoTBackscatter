import matplotlib.pyplot as plt
import numpy as np


def plot_freq_time(samples, time, title):
    plt.figure(figsize=(12, 8))
    
    # subplot 1 - time domain
    plt.subplot(2, 1, 1)
    plt.plot(time, np.real(samples), 'b-', label='Real')
    plt.plot(time, np.imag(samples), 'r-', label='Imag')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title(f'{title} - Time Domain')
    plt.legend()
    plt.grid(True)
    
    # subplot 2 - freq domain
    plt.subplot(2, 1, 2)
    fft_samples = np.fft.fft(samples)
    freqs = np.fft.fftfreq(len(samples), 1/samp_rate_hz)
    plt.plot(freqs, np.abs(fft_samples))
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.title(f'{title} - Frequency Domain')
    plt.grid(True)
    plt.tight_layout()

print("Debugging complex exponential mixing...")

samp_rate_hz = 100
dur_sec = 5
og_freq_hz = 2
mix_freq_hz = 10  # frequency to mix up to

# define an original signal 
t_array = np.linspace(0, dur_sec, int(samp_rate_hz * dur_sec), endpoint=False)
og_sig = np.exp(1j * 2 * np.pi * og_freq_hz * t_array)  # complex exponential at 2 Hz
plot_freq_time(og_sig, t_array, "Original 2Hz signal")

# multiply it by a complex exponential to move to a higher frequency
mix_up = np.exp(1j * 2 * np.pi * mix_freq_hz * t_array)
mixed_up_sig = og_sig * mix_up
plot_freq_time(mixed_up_sig, t_array, "Mixed up to 12Hz (2+10)")

# multiply it by the negative complex exponential to bring it back to the original frequency
mix_down = np.exp(-1j * 2 * np.pi * mix_freq_hz * t_array)
recovered_sig = mixed_up_sig * mix_down
plot_freq_time(recovered_sig, t_array, "Recovered signal (should be 2Hz)")

# compare original vs recovered
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(t_array[:50], np.real(og_sig[:50]), 'b-', label='Original Real', linewidth=2)
plt.plot(t_array[:50], np.real(recovered_sig[:50]), 'r--', label='Recovered Real', linewidth=2)
plt.legend()
plt.title('Original vs Recovered Signal - Real Part')
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(t_array[:50], np.imag(og_sig[:50]), 'b-', label='Original Imag', linewidth=2)
plt.plot(t_array[:50], np.imag(recovered_sig[:50]), 'r--', label='Recovered Imag', linewidth=2)
plt.legend()
plt.title('Original vs Recovered Signal - Imaginary Part')
plt.grid(True)
plt.tight_layout()

# check if they're equal
print(f"Signals are equal: {np.allclose(og_sig, recovered_sig)}")
print(f"Max difference: {np.max(np.abs(og_sig - recovered_sig))}")

plt.show()