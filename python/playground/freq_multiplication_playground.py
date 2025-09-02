import matplotlib.pyplot as plt
import numpy as np

freq_hz = 5
duration_s = 1
nsamps = 101
timearray_s = np.linspace(0,duration_s,nsamps)
in_samples = np.exp(1j*2*np.pi*freq_hz*timearray_s)

plt.figure()
plt.plot(timearray_s,np.real(in_samples))
plt.plot(timearray_s, np.imag(in_samples))
plt.legend(["real","imag"])
plt.grid("on")
plt.title("original")

out_samples = in_samples*np.exp(-1j*2*np.pi*freq_hz*timearray_s)

plt.figure()
plt.plot(timearray_s,np.real(out_samples))
plt.plot(timearray_s, np.imag(out_samples))
plt.legend(["real","imag"])
plt.grid("on")
plt.title("demultiplied")

cos_samples = in_samples*np.real(np.exp(-1j*2*np.pi*freq_hz*timearray_s))

plt.figure()
plt.plot(timearray_s,np.real(cos_samples))
plt.plot(timearray_s, np.imag(cos_samples))
plt.legend(["real","imag"])
plt.grid("on")
plt.title("demultiplied cos")

sin_samples = in_samples*np.imag(np.exp(-1j*2*np.pi*freq_hz*timearray_s))

plt.figure()
plt.plot(timearray_s,np.real(sin_samples))
plt.plot(timearray_s, np.imag(sin_samples))
plt.legend(["real","imag"])
plt.grid("on")
plt.title("demultiplied sin")

plt.show()