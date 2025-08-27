import matplotlib.pyplot as plt
import numpy as np
from main import logger
from scipy import signal


def plot_time_psd_scat(samples, samp_rate, title):

    plt.figure()
    
    plt.subplot(3, 1, 1)
    plt.plot(np.real(samples),'.-')
    plt.plot(np.imag(samples),'.-')
    plt.title(title)
    plt.legend(['real', 'imag'])
    plt.grid('on')

    plt.subplot(3,1,2)
    plt.psd(samples, NFFT=1024, Fs=samp_rate)
    plt.title('PSD of '+title)

    plt.subplot(3, 1, 3)
    plt.scatter(np.real(samples), np.imag(samples))
    plt.title(title+' constellation')
    plt.grid('on')
    plt.axis('equal')
def lpf(samples, data_rate, samp_rate):

    #filter
    cutoff_freq = data_rate*1.5
    # transition_bw = data_rate*0.5
    b, a = signal.butter(5, cutoff_freq, 'low', fs=samp_rate)
    out_samples = signal.filtfilt(b, a, samples)
    
    return(out_samples)

def agc(samples,out_amplitude):
    
    mag = np.sum(np.abs(samples)**2)/len(samples)
    
    gain = out_amplitude/np.sqrt(mag)
    
    samples = samples*gain
    
    # mag_out = np.sum(np.abs(samples)**2)/len(samples)
    # print("original mag: ", mag)
    # print("gain: ",gain)
    # print("mag_out: ",mag_out)
    
    return(samples) 

def demodulate_packet(input_samples, tag_params, radio_params):
    logger.info("Processing %s samples from tag %s", len(input_samples), tag_params.tag_id) 
    plot_time_psd_scat(input_samples, radio_params.samplerate_hz, "Raw Received Samples")
    
    #low pass filter
    proc_samples = lpf(input_samples, 25e3, radio_params.samplerate_hz)
    plot_time_psd_scat(proc_samples, radio_params.samplerate_hz, "Filtered Samples")
    
    #agc
    proc_samples = agc(proc_samples, 1/np.sqrt(2))
    plot_time_psd_scat(proc_samples, radio_params.samplerate_hz, "Filtered & AGCed Samples")

    
    #despread
    
    #mm time recovery
    
    #coarse f correct
    
    #costas loop
    
    #bpsk demod
    
    #average over cdma symbol
    
    logger.info("Output bits: ")
    
    logger.info("BER: ")
    
    plt.show()