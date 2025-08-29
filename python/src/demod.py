import matplotlib.pyplot as plt
import numpy as np
from IoTBSConst import gc_len
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
    samples_out = samples*gain
    
    return(samples_out) 

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

def course_f_correct(samples, samp_rate):
    samples_sqr = samples**2
    psd_sq = np.fft.fftshift(np.abs(np.fft.fft(samples_sqr)))
    f = np.linspace(-samp_rate/2.0, samp_rate/2.0, len(psd_sq))
    max_freq = f[np.argmax(psd_sq)]
    Ts = 1/samp_rate # calc sample period
    # t = np.arange(0, Ts*len(samples), Ts) # create time vector
    t = np.linspace(0,(len(samples)-1)/samp_rate, len(samples))
    samples = samples * np.exp(-1j*2*np.pi*max_freq*t/2.0)
    
    logger.debug("coarse freq offset: %s", max_freq)
    
    return(samples)

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

def sync_word_sync(bits_in, s_word, nbits_out, samps_per_bit):
        
    long_sync_word = np.repeat(s_word,samps_per_bit)
    sync_word_corr = np.correlate(bits_in, long_sync_word, mode='valid')
    
    plt.figure()
    plt.plot(sync_word_corr)
    
    offset = int(np.argmax(np.abs(sync_word_corr))) #the filter introduces a 6 bit delay
    logger.info("offset: %s", offset)
    logger.info("value of correlation at offset: %s", sync_word_corr[offset])
    sign = np.sign(sync_word_corr[offset])
    bits_out = bits_in[offset+len(s_word):]

    bits_out = (sign*bits_out[:nbits_out]+1)/2

    return(bits_out, sync_word_corr[offset])

def demodulate_packet(input_samples, tag_params, radio_params):
    logger.info("Processing %s samples generated by %s tags", len(input_samples), tag_params.ntags) 
    # plot_time_psd_scat(input_samples, radio_params.samplerate_hz, "Raw Received Samples")
    
    for itag in range(tag_params.ntags):
        itagParams = tag_params.get_tag(itag)
        proc_samples = input_samples
        
        #low pass filter
        proc_samples = lpf(proc_samples, 25e3/2, radio_params.samplerate_hz)
        # plot_time_psd_scat(proc_samples, radio_params.samplerate_hz, "Filtered Samples")
        
        #agc
        proc_samples = agc(proc_samples, 1/np.sqrt(2))
        # plot_time_psd_scat(proc_samples, radio_params.samplerate_hz, "Filtered & AGCed Samples")

        #correlation to find if signal is present and, if so, what the offset freq is
        

        #despread
        bits_to_calc = len(itagParams.all_bits)
        logger.debug("Expected number of bits: %s", bits_to_calc)
        samps_per_bit = itagParams.sps*len(itagParams.goldcode)
        despread_samples_repeat = np.zeros(samps_per_bit*bits_to_calc).astype(np.complex64)
        
        for bit in range(bits_to_calc):
            despread_samples_repeat[bit*samps_per_bit:(bit+1)*samps_per_bit] = proc_samples[bit*samps_per_bit:(bit+1)*samps_per_bit]*np.repeat(itagParams.goldcode.astype(np.complex64),itagParams.sps)

        # plot_time_psd_scat(despread_samples_repeat, radio_params.samplerate_hz, "Despread Samples w/ Repeat")
        
        despread_samples_interp = np.zeros(samps_per_bit*bits_to_calc).astype(np.complex64)
        
        for bit in range(bits_to_calc):
            despread_samples_interp[bit*samps_per_bit:(bit+1)*samps_per_bit] = proc_samples[bit*samps_per_bit:(bit+1)*samps_per_bit]*signal.resample_poly(itagParams.goldcode.astype(np.complex64),itagParams.sps,1)

        # plot_time_psd_scat(despread_samples_interp, radio_params.samplerate_hz, "Despread Samples w/ Interp")


        #mm time recovery
        proc_samples = mm_time_recovery(despread_samples_repeat, itagParams.sps)
        
        #coarse f correct
        proc_samples = course_f_correct(proc_samples, radio_params.samplerate_hz)
        # plt.figure()
        # plt.plot(proc_samples)
        # plt.title("After course f correct")
        # plt.grid("on")
        
        #costas loop
        proc_samples = costas_loop(proc_samples, radio_params.samplerate_hz)
        # plt.figure()
        # plt.plot(proc_samples)
        # plt.title("After Costas Loop")
        # plt.grid("on")
        
        #bpsk demod
        rx_bits_raw = demod_bpsk(proc_samples)
        # logger.info("Raw received bits: %s",rx_bits_raw.tolist())
        # plt.figure()
        # plt.plot(rx_bits_raw)
        # plt.title("rx_bits_raw")
        # plt.grid("on")
        
        logger.debug("rx_bits_raw: %s", rx_bits_raw)
        
        #average over cdma symbol - create array filled with NaNs
        nbits_data = len(itagParams.actual_bits)
        proc_bits = np.full(nbits_data, np.nan)
        
        # Bit decision logic: majority voting over gc_len chunks 
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
                
        # logger.info("proc_bits:%s",proc_bits)
        
        # rx_bits_payload, sync_word_corr = sync_word_sync(rx_bits_raw*2-1, tag_params.sync_bits*2-1, len(tag_params.payload_bits), samps_per_bit)
        
        logger.debug("Processed bits: %s",proc_bits) 
        logger.debug("Transmitted actual bits: %s", itagParams.actual_bits)
        # logger.info("Sync word correlation: %s", sync_word_corr)
        
        num_bits = len(itagParams.actual_bits)
        num_errors = sum(abs(proc_bits-itagParams.actual_bits))
        BER = num_errors/num_bits
        
        logger.debug("BER (tag%s): %s", itag, BER)
        
        # Store results for this tag
        if itag == 0:
            results = {}
        results[itag] = {
            'num_errors': num_errors,
            'num_bits': num_bits,
            'ber': BER
        }
    
    plt.show()
    return results