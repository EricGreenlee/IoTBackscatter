import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")

from scipy import signal
# from rtlsdr import RtlSdr

#to do:
# - determine gold code
# - tune it for greater distances

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

def print_bits(bits,bits_per_line):
    for i in range(0, len(bits), bits_per_line):
        print(bits[i:i + bits_per_line])
        
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

def freq_cdma_search(samples, goldcode, fs, max_freq_dev, nfreq_pts, resamp_rate):

    freq_offsets = np.linspace(-max_freq_dev, max_freq_dev, nfreq_pts)
    # print("Freq bin spacing: ", freq_offsets[1]-freq_offsets[0])

    samples_resamp = signal.resample_poly(samples, resamp_rate, 1)
    t_resamp = np.linspace(0, (len(samples_resamp)-1)/fs, len(samples_resamp))

    cdma_code = signal.resample_poly(goldcode.astype(np.complex64),sps*resamp_rate,1)

    correlation = np.correlate(samples_resamp, cdma_code, mode='valid')
    time_delays = np.linspace(0, (len(correlation)-1)/fs, len(correlation))


    # Create meshgrid for time delay and frequency offset
    X, Y = np.meshgrid(time_delays, freq_offsets)
    Z = np.zeros((len(freq_offsets), len(time_delays)))

    for j, freq_offset in enumerate(freq_offsets):
        
        test_signal = samples_resamp * np.exp(-1j*2*np.pi*freq_offset*t_resamp)

        correlation = np.correlate(test_signal, cdma_code, mode='valid')

        Z[j, :] = np.abs(correlation)

    # fig_lab = plt.figure()
    # ax = fig_lab.add_subplot(111, projection='3d')
    # ax.plot_surface(X, Y, Z[:,:], cmap='viridis')
    # ax.set_title(f'Correlation with gold code')

    # # Set labels
    # ax.set_xlabel('Time Delay (s)')
    # ax.set_ylabel('Frequency Offset (Hz)')
    # ax.set_zlabel('Signal Strength')

    max_corr = np.max(Z)
    max_corr_idx = np.argmax(Z)
    max_corr_idx = np.unravel_index(max_corr_idx, Z.shape)

    # print("freq index of max correlation: ", max_corr_idx[0])
    # print("time index of max correlation: ", max_corr_idx[1])

    cand_freq_offset = freq_offsets[max_corr_idx[0]]
    cand_time_delay = time_delays[max_corr_idx[1]]
    cand_sample_delay = int(cand_time_delay*fs)

    # print("max correlation: ", max_corr)
    # print("freq_offset: ",cand_freq_offset)
    # print("cand_time_delay: ",cand_time_delay)
    # print("cand_sample_delay: ",cand_sample_delay)
    
    return(max_corr, cand_freq_offset, cand_sample_delay)

def plot_freq_cdma_search(samples, goldcode, fs, max_freq_dev, nfreq_pts, resamp_rate):

    freq_offsets = np.linspace(-max_freq_dev, max_freq_dev, nfreq_pts)
    # print("Freq bin spacing: ", freq_offsets[1]-freq_offsets[0])

    samples_resamp = signal.resample_poly(samples, resamp_rate, 1)
    t_resamp = np.linspace(0, (len(samples_resamp)-1)/fs, len(samples_resamp))

    cdma_code = signal.resample_poly(goldcode.astype(np.complex64),sps*resamp_rate,1)

    correlation = np.correlate(samples_resamp, cdma_code, mode='valid')
    time_delays = np.linspace(0, (len(correlation)-1)/fs, len(correlation))


    # Create meshgrid for time delay and frequency offset
    X, Y = np.meshgrid(time_delays, freq_offsets)
    Z = np.zeros((len(freq_offsets), len(time_delays)))

    for j, freq_offset in enumerate(freq_offsets):
        
        test_signal = samples_resamp * np.exp(-1j*2*np.pi*freq_offset*t_resamp)

        correlation = np.correlate(test_signal, cdma_code, mode='valid')

        Z[j, :] = np.abs(correlation)

    fig_lab = plt.figure()
    ax = fig_lab.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z[:,:], cmap='viridis')
    # ax.set_title(f'Correlation with gold code')

    # Set labels
    ax.set_xlabel('Time Delay (s)')
    ax.set_ylabel('Frequency Offset (Hz)')
    ax.set_zlabel('Signal Strength')

    max_corr = np.max(Z)
    max_corr_idx = np.argmax(Z)
    max_corr_idx = np.unravel_index(max_corr_idx, Z.shape)

    print("freq index of max correlation: ", max_corr_idx[0])
    print("time index of max correlation: ", max_corr_idx[1])

    cand_freq_offset = freq_offsets[max_corr_idx[0]]
    cand_time_delay = time_delays[max_corr_idx[1]]
    cand_sample_delay = int(cand_time_delay*fs)

    print("max correlation: ", max_corr)
    print("freq_offset: ",cand_freq_offset)
    print("cand_time_delay: ",cand_time_delay)
    print("cand_sample_delay: ",cand_sample_delay)
    
    return(max_corr, cand_freq_offset, cand_sample_delay)

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
    
    print("coarse freq offset: ", max_freq)
    
    return(samples)

def costas_loop(samples, samp_rate):
    
    N = len(samples)
    phase = 0
    freq = 0
    # These next two params is what to adjust, to make the feedback loop faster or slower (which impacts stability)
    # alpha = 0.132
    # beta = 0.00932
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
    
    plt.figure()
    plt.plot(freq_log)
    plt.title("Frequency offset from costas loop")
    plt.grid("on")
            
    return(out)

def demod_bpsk(samples):
            
    nbits = len(samples)
    bits = np.zeros(nbits)
    for i in range(nbits):
        bits[i] = int(np.real(samples[i]) > 0)
        
    return(bits)       

def sync_word_sync(bits_in, s_word, nbits_out):
        
    sync_word_corr = np.correlate(bits_in, s_word, mode='valid')
    
    plt.figure()
    plt.plot(sync_word_corr)
    
    offset = int(np.argmax(np.abs(sync_word_corr))) #the filter introduces a 6 bit delay
    print("offset: ", offset)
    print("value of correlation at offset: ", sync_word_corr[offset])
    sign = np.sign(sync_word_corr[offset])
    bits_out = bits_in[offset+len(s_word):]

    bits_out = (sign*bits_out[:nbits_out]+1)/2

    return(bits_out, sync_word_corr[offset])

def calc_sps(samples):
    auto_corr = np.abs(np.correlate(samples,samples,"full"))
    threshold = 0.25 * np.max(auto_corr)
    peaks, _ = signal.find_peaks(auto_corr, height=threshold)
    peak_values = auto_corr[peaks]

    plt.figure()
    plt.plot(auto_corr)
    plt.plot(peaks, peak_values, 'x')
    plt.title("Auto correlation")


    real_sps = np.mean(np.diff(peaks))
    print("actual samples per symbol:", real_sps)
    
    return(real_sps)

def demod_packet(samples, goldcode, fs, datarate, sps, sync_word, freq_off, expected_nbits):
    
    nbits_preamble = 16

    test_samples = lpf(samples, datarate, fs)
    test_samples = agc(test_samples, 1/np.sqrt(2))

    plot_time_psd_scat(test_samples,fs, "filter and agc")

    #frequency correction
    t = np.linspace(0,(len(test_samples)-1)/fs, len(test_samples))
    test_samples = test_samples *np.exp(-2j*np.pi*freq_off*t*2)

    #find the start indexes of every sample
    # goldcode = GC7_2_bip
    long_gc = signal.resample_poly(goldcode.astype(np.complex64),sps,1)
    gc_corr = np.abs(np.correlate(test_samples,long_gc,"full"))

    threshold = 0.6 * np.max(gc_corr)
    # gc_peak_inds, _ = signal.find_peaks(gc_corr, height=threshold)
    gc_peak_inds, _ = signal.find_peaks(gc_corr, height=threshold, distance=gc_len*sps/2)
    gc_peak_values = gc_corr[gc_peak_inds]
    
    #find the distance between peaks
    inter_peak_dist = np.diff(gc_peak_inds)

    plt.figure()
    plt.plot(gc_corr)
    plt.plot(gc_peak_inds, gc_peak_values, 'x')
    plt.title("Gold code correlation")

    print("gc_peaks indexes: ",gc_peak_inds)
    print("number of peaks: ",len(gc_peak_inds))
    print("difference in peaks: ",inter_peak_dist)
    print("mean diff: ",np.mean(np.diff(gc_peak_inds)))
    
    if np.any(inter_peak_dist > gc_len*sps*2):
        print("peaks too far apart, exiting")
        return(0,0)
    if len(gc_peak_inds) < 10:
        print("not enough peaks, exiting")
        return(0,0)
    
    
    bits_to_calc = np.min([expected_nbits+len(sync_word)+nbits_preamble, len(gc_peak_inds)-2])
    print("bits to calculate: ", bits_to_calc)


    despread_samples = np.zeros(sps*gc_len*bits_to_calc).astype(np.complex64)
    for bit in range(bits_to_calc):
        # print("bit: ",bit+1)
        try:
            despread_samples[bit*sps*gc_len:(bit+1)*sps*gc_len] = test_samples[gc_peak_inds[bit+1]:gc_peak_inds[bit+1]+sps*gc_len]*long_gc
        except:
            print("error in despread samples")
            print("bit: ",bit+1)
        
    plot_time_psd_scat(despread_samples,fs, "Despread samples")

    proc_samples = mm_time_recovery(despread_samples, sps)#*gc_len)
    plot_time_psd_scat(proc_samples, fs, "MM time recovery")

    proc_samples = course_f_correct(proc_samples, fs)#/len(GC7_2_bip))
    plot_time_psd_scat(proc_samples, fs, "coarse freq correction")

    proc_samples = costas_loop(proc_samples, fs)#/len(GC7_2_bip))
    plot_time_psd_scat(proc_samples, fs, "costas loop")

    bits_raw = demod_bpsk(proc_samples)

    # print("Output bits: ", bits_raw[0:100])
    # print_bits(bits_raw,127)

    bit_outs = np.zeros(bits_to_calc)
    for i in range(bits_to_calc):
        temp_sum = np.sum(bits_raw[i*gc_len:(i+1)*gc_len])
        if temp_sum > gc_len/2:
            bit_outs[i] = 1
        else:
            bit_outs[i] = 0
        # bit_outs[i] = np.sum(bits_raw[i*gc_len:(i+1)*gc_len])

    print_bits(bit_outs, 16)

    bits_out_payload, sync_word_corr = sync_word_sync(bit_outs*2-1, sync_word*2-1, expected_nbits)#100-32)
    
    return(bits_out_payload, sync_word_corr)

def demod_packet_plot(samples, goldcode, fs, datarate, sps, sync_word, freq_off, expected_nbits):
    
    nbits_preamble = 16
    
    plt.psd(samples, NFFT=1024, Fs=fs)

    test_samples = lpf(samples, datarate, fs)
    test_samples = agc(test_samples, 1/np.sqrt(2))

    # plot_time_psd_scat(test_samples,fs, "filter and agc")
    plt.figure()
    plt.psd(test_samples, NFFT=1024, Fs=fs)

    #frequency correction
    t = np.linspace(0,(len(test_samples)-1)/fs, len(test_samples))
    test_samples = test_samples *np.exp(-2j*np.pi*freq_off*t*2)

    #find the start indexes of every sample
    # goldcode = GC7_2_bip
    long_gc = signal.resample_poly(goldcode.astype(np.complex64),sps,1)
    gc_corr = np.abs(np.correlate(test_samples,long_gc,"full"))

    threshold = 0.6 * np.max(gc_corr)
    # gc_peak_inds, _ = signal.find_peaks(gc_corr, height=threshold)
    gc_peak_inds, _ = signal.find_peaks(gc_corr, height=threshold, distance=gc_len*sps/2)
    gc_peak_values = gc_corr[gc_peak_inds]
    
    #find the distance between peaks
    inter_peak_dist = np.diff(gc_peak_inds)

    plt.figure()
    plt.plot(gc_corr)
    plt.plot(gc_peak_inds, gc_peak_values, 'x')
    plt.title("Gold code correlation")

    print("gc_peaks indexes: ",gc_peak_inds)
    print("number of peaks: ",len(gc_peak_inds))
    print("difference in peaks: ",inter_peak_dist)
    print("mean diff: ",np.mean(np.diff(gc_peak_inds)))
    
    if np.any(inter_peak_dist > gc_len*sps*2):
        print("peaks too far apart, exiting")
        return(0,0)
    if len(gc_peak_inds) < 10:
        print("not enough peaks, exiting")
        return(0,0)
    
    
    bits_to_calc = np.min([expected_nbits+len(sync_word)+nbits_preamble, len(gc_peak_inds)-2])
    print("bits to calculate: ", bits_to_calc)


    despread_samples = np.zeros(sps*gc_len*bits_to_calc).astype(np.complex64)
    for bit in range(bits_to_calc):
        # print("bit: ",bit+1)
        try:
            despread_samples[bit*sps*gc_len:(bit+1)*sps*gc_len] = test_samples[gc_peak_inds[bit+1]:gc_peak_inds[bit+1]+sps*gc_len]*long_gc
        except:
            print("error in despread samples")
            print("bit: ",bit+1)
        
    # plot_time_psd_scat(despread_samples,fs, "Despread samples")
    plt.figure()
    plt.psd(despread_samples, NFFT=1024, Fs=fs)

    proc_samples = mm_time_recovery(despread_samples, sps)#*gc_len)
    # plot_time_psd_scat(proc_samples, fs, "MM time recovery")

    proc_samples = course_f_correct(proc_samples, fs)#/len(GC7_2_bip))
    # plot_time_psd_scat(proc_samples, fs, "coarse freq correction")

    proc_samples = costas_loop(proc_samples, fs)#/len(GC7_2_bip))
    # plot_time_psd_scat(proc_samples, fs, "costas loop")
    plt.figure()
    plt.scatter(np.real(proc_samples), np.imag(proc_samples))
    plt.grid('on')
    plt.axis('equal')
    
    plt.figure()
    plt.plot(np.real(proc_samples),':')
    plt.plot(np.imag(proc_samples),'.-')
    # plt.title("Costas loop output"
    plt.legend(['real', 'imag'])
    plt.grid('on')
    
    bits_raw = demod_bpsk(proc_samples)

    # print("Output bits: ", bits_raw[0:100])
    # print_bits(bits_raw,127)

    bit_outs = np.zeros(bits_to_calc)
    for i in range(bits_to_calc):
        temp_sum = np.sum(bits_raw[i*gc_len:(i+1)*gc_len])
        if temp_sum > gc_len/2:
            bit_outs[i] = 1
        else:
            bit_outs[i] = 0
        # bit_outs[i] = np.sum(bits_raw[i*gc_len:(i+1)*gc_len])

    # print
    print_bits(bit_outs, 16)

    bits_out_payload, sync_word_corr = sync_word_sync(bit_outs*2-1, sync_word*2-1, expected_nbits)#100-32)
    
    return(bits_out_payload, sync_word_corr)

def demod_packet_quiet(samples, goldcode, fs, datarate, sps, sync_word, freq_off, expected_nbits):
    
    nbits_preamble = 16

    test_samples = lpf(samples, datarate, fs)
    test_samples = agc(test_samples, 1/np.sqrt(2))

    # plot_time_psd_scat(test_samples,fs, "filter and agc")

    #frequency correction
    t = np.linspace(0,(len(test_samples)-1)/fs, len(test_samples))
    test_samples = test_samples *np.exp(-2j*np.pi*freq_off*t*2)

    #find the start indexes of every sample
    # goldcode = GC7_2_bip
    long_gc = signal.resample_poly(goldcode.astype(np.complex64),sps,1)
    gc_corr = np.abs(np.correlate(test_samples,long_gc,"full"))

    threshold = 0.6 * np.max(gc_corr)
    # gc_peak_inds, _ = signal.find_peaks(gc_corr, height=threshold)
    gc_peak_inds, _ = signal.find_peaks(gc_corr, height=threshold, distance=gc_len*sps/2)
    gc_peak_values = gc_corr[gc_peak_inds]
    
    #find the distance between peaks
    inter_peak_dist = np.diff(gc_peak_inds)



    print("gc_peaks indexes: ",gc_peak_inds)
    print("number of peaks: ",len(gc_peak_inds))
    print("difference in peaks: ",inter_peak_dist)
    print("mean diff: ",np.mean(np.diff(gc_peak_inds)))
    
    if np.any(inter_peak_dist > gc_len*sps*2):
        print("peaks too far apart, exiting")
        return(0,0)
    if len(gc_peak_inds) < 10:
        print("not enough peaks, exiting")
        return(0,0)
    
    # plt.figure()
    # plt.plot(gc_corr)
    # plt.plot(gc_peak_inds, gc_peak_values, 'x')
    # plt.title("Gold code correlation")
    
    # plt.show()
    
    
    bits_to_calc = np.min([expected_nbits+len(sync_word)+nbits_preamble, len(gc_peak_inds)-2])
    print("bits to calculate: ", bits_to_calc)


    despread_samples = np.zeros(sps*gc_len*bits_to_calc).astype(np.complex64)
    for bit in range(bits_to_calc):
        # print("bit: ",bit+1)
        try:
            despread_samples[bit*sps*gc_len:(bit+1)*sps*gc_len] = test_samples[gc_peak_inds[bit+1]:gc_peak_inds[bit+1]+sps*gc_len]*long_gc
        except:
            print("error in despread samples")
            print("bit: ",bit+1)
        
    # plot_time_psd_scat(despread_samples,fs, "Despread samples")

    proc_samples = mm_time_recovery(despread_samples, sps)#*gc_len)
    # plot_time_psd_scat(proc_samples, fs, "MM time recovery")

    proc_samples = course_f_correct(proc_samples, fs)#/len(GC7_2_bip))
    # plot_time_psd_scat(proc_samples, fs, "coarse freq correction")

    proc_samples = costas_loop(proc_samples, fs)#/len(GC7_2_bip))
    # plot_time_psd_scat(proc_samples, fs, "costas loop")

    bits_raw = demod_bpsk(proc_samples)

    # print("Output bits: ", bits_raw[0:100])
    # print_bits(bits_raw,127)

    bit_outs = np.zeros(bits_to_calc)
    for i in range(bits_to_calc):
        temp_sum = np.sum(bits_raw[i*gc_len:(i+1)*gc_len])
        if temp_sum > gc_len/2:
            bit_outs[i] = 1
        else:
            bit_outs[i] = 0
        # bit_outs[i] = np.sum(bits_raw[i*gc_len:(i+1)*gc_len])

    print
    print_bits(bit_outs, 16)

    bits_out_payload, sync_word_corr = sync_word_sync(bit_outs*2-1, sync_word*2-1, expected_nbits)#100-32)
    
    return(bits_out_payload, sync_word_corr)


def read_live_samples():
    sample_rate = 252315
    center_freq = 915.1e6 #- 1e3
    SDR_gain = 40

    gc_len = 127
    sps = 10
    bits_per_packet = 100

    #from Arduino device
    symbol_dur_us = 40
    datarate = 1/(symbol_dur_us*1e-6)
    print(f"Tag data rate: {datarate} bps")

    #catch at least 4 symbols
    target_nsamps = sps*gc_len*bits_per_packet*3
    #round up to nearest multiple of 4096
    nsamps = int(np.ceil(target_nsamps/4096)*4096)

    # nsamps = 2048
    print(f"Number of samples: {nsamps}")


    # Create an SDR object
    # try:
    sdr = RtlSdr()

    print('Found an RTL-SDR device!')
    # print('Current frequency: {:.2f} MHz'.format(sdr.center_freq / 1e6))

    # Set configuration values
    sdr.sample_rate = sample_rate  # Hz
    sdr.center_freq = center_freq # Tune to 101.1 MHz (example)
    # sdr.gain = 'auto'
    sdr.gain = SDR_gain
    # sdr.gain = 10000

    print('Sampling rate: {:.2f} MHz'.format(sdr.sample_rate / 1e6))
    print('Current frequency: {:.2f} MHz'.format(sdr.center_freq / 1e6))
    print(f'Gain: {sdr.gain}')

    # Read samples
    print('Reading samples...')
    # all_samps = sdr.read_samples(4096*16)
    all_rx_samples = sdr.read_samples(nsamps)
    sdr.close()
    # all_samps = []
    # for i in range(10):
    #     samples = sdr.read_samples(4096)
    #     all_samps.append(samples)

    print(f'Read {len(all_rx_samples)} samples.')

    #dynamically generate filename from current time
    from datetime import datetime
    filename = "live_samps/rtlsdr_GC7-2_100bit_"+str(sample_rate)+"k_"+str(int(center_freq/1e6))+"M_40_"+datetime.now().strftime("%Y%m%d")+datetime.now().strftime("%H%M%S")+".dat"

    all_rx_samples.astype(np.complex64).tofile(filename)
    print(f"Saved {len(all_rx_samples)} samples to '{filename}'.")

    return(all_rx_samples)

def read_from_file(fname):
    
    rx_samples_lab = np.fromfile(fname, dtype=np.complex64)
    print(f'Read {len(rx_samples_lab)} samples from file.')
    
    return(rx_samples_lab)

def generate_ideal_packet(Goldcodes, device_id, packet_bits, sps):
    
    gc_len = len(Goldcodes[device_id])
    packet = np.zeros(sps*gc_len*len(packet_bits)).astype(np.complex64)
    for i, bit in enumerate(packet_bits):
        # packet[i*sps*gc_len:(i+1)*sps*gc_len] = signal.resample_poly(Goldcodes[device_id].astype(np.complex64)*(bit*2-1),sps,1)
        packet[i*sps*gc_len:(i+1)*sps*gc_len] = np.repeat(Goldcodes[device_id].astype(np.complex64)*(bit*2-1),sps)
    
    return(packet)

def add_frac_delay(samples, delay):
# Create and apply fractional delay filter
    N = 21 # number of taps
    n = np.arange(-N//2, N//2) # ...-3,-2,-1,0,1,2,3...
    h = np.sinc(n - delay) # calc filter taps
    h *= np.hamming(N) # window the filter to make sure it decays to 0 on both sides
    h /= np.sum(h) # normalize to get unity gain, we don't want to change the amplitude/power
    out_samples = np.convolve(samples, h) # apply filter
    
    return(out_samples)

def extend_int_delay(samples,delay):
    nsamps_in = len(samples)
    out_samples = np.zeros(nsamps_in*2).astype(np.complex64)
    out_samples[delay:delay+nsamps_in] = samples
    
    return(out_samples)

def add_freq_offset(samples, samp_rate, freq_off):
    #frequency offset
    Ts = 1/samp_rate # calc sample period
    # t = np.arange(0, Ts*len(samples), Ts) # create time vector
    t = np.linspace(0,(len(samples)-1)/samp_rate, len(samples))
    out_samples = samples * np.exp(1j*2*np.pi*freq_off*t)
    
    return(out_samples)

def add_carrier(samples, samp_rate, carrier_freq):
    t = np.linspace(0,(len(samples)-1)/samp_rate, len(samples))
    out_samples = samples + 10000*np.exp(1j*2*np.pi*carrier_freq*t)
    
    return(out_samples)

def add_noise_path_loss(samples, tx_pwr, noise_pwr, dist):
    
    Gt = 2 #transmit antenna gain
    Gr = 2 #receive antenna gain
    Gtag = 10 #tag antenna gain
    lamda = 3e8/915.1e6 #wavelength
    n = 2 #path loss exponent
    Theta = 1.23 #on-object gain penalty
    F_alpha = 6.31 #Fade margin

    #receive the signal with path loss and noise
    noise = np.random.normal(0, np.sqrt(noise_pwr), len(samples)) + 1j*np.random.normal(0, np.sqrt(noise_pwr), len(samples))
    rx_pwr = tx_pwr*Gt*Gr*(Gtag**2)*(lamda**(n*2))/((4*np.pi*dist)**(n*2)*(Theta**2)*F_alpha)
    print("lamda: ", lamda)
    print("rx_pwr: ", rx_pwr)
    
    out_samples = np.sqrt(rx_pwr) * samples + noise
    
    return(out_samples)

def add_nonidealities(packet,freq,fsamp,fracdelay,wholedelay,noise_pwr,dist, tx_pwr):
    # t = np.arange(0, (len(packet)-1)/fsamp, len(packet))
    
    tx_packet = packet
    # print("tx_packet length: ", len(tx_packet))
    # print("tx_packet: ", tx_packet[0:10])
    # print("fracdelay: ",fracdelay)
    # print("wholedelay: ",wholedelay)
    tx_packet = add_frac_delay(packet,fracdelay)
    tx_packet = extend_int_delay(tx_packet,wholedelay)
    # print("tx_packet length: ", len(tx_packet))
    # print("tx_packet: ", tx_packet[0:10])
    tx_packet = add_freq_offset(tx_packet,fsamp,freq)
    # tx_packet = add_carrier(tx_packet,fsamp, -915e6)
    tx_packet = add_noise_path_loss(tx_packet, tx_pwr, noise_pwr, dist)
    # print("tx_packet length: ", len(tx_packet))
    # print("tx_packet: ", tx_packet[0:10])
    
    return(tx_packet)
    
    


fs = 252315 #sampling rate
fc = 915.1e6 #carrier frequency
datarate = 25e3 #data rate
# expected_sps = fs/datarate #samples per symbol
sps = 10

# GC7_2_bip = np.array([1, 1, -1, -1, -1, -1, -1, 1, 1, -1, -1, -1, -1, 1, 1, 1, -1, -1, -1, -1, -1, 1, 1, -1, 1, -1, -1, 1, 1, -1, 1, -1, -1, -1, 1, -1, 1, 1, -1, 1, 1, -1, -1, -1, -1, 1, -1, -1, 1, -1, -1, -1, -1, 1, -1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, -1, 1, -1, 1, -1, -1, 1, -1, -1, 1, 1, 1, 1, -1, 1, -1, 1, -1, -1, -1, 1, 1, -1, -1, 1, -1, -1, 1, -1, -1, -1, 1, 1, 1, -1, -1, -1, 1, -1, -1, 1, 1, -1, 1, 1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, -1, 1, -1, 1])
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

gc_len = len(GCs[0])
num_gcs = len(GCs)

# raw_data = np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
#                     0,1,0,1,1,0,0,1,1,1,1,1,0,0,0,0,
#                     1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
#                     1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
#                     1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
#                     1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
#                     1,1,1,1])
# preamble = raw_data[0:16]
# sync_word = raw_data[16:32]
# bits_sent = raw_data[32:]

seed = np.random.randint(0, 2**32-1)
np.random.seed(seed)
print("seed: ", seed)   

#generate a log file to track the BER for different distances and goldcodes
# fname = "sim_results/BER_sim_8Devices_20250223.log"
fname = "PythonSDR/20250820_dev/sim_results/multi_device_play.log"
with open(fname, "a") as f:
    # f.write("gc, distance, freq_offset, frac_delay, whole_delay, noise_power, tx_power, bits_sent, bit_errors\n")
    f.write("ndevices, gc, distance, tx_power, noise_power, freq_offset, frac_delay, whole_delay, bits_sent, bit_errors\n")
f.close()

# ndevices = 1
ndevices_array = [3]#[1,2,3,4,5,6,7,8]#[2,3]#
distance = 10
dist_array = [5]#21,22,23,24,25]#26,27,28,29,31,32,33,34]#[10,15,20,25,30,35]#
additional_distance = 0
additional_distance_array = [0,1,2,3,4,5,6,7,8,9]#[0,1,2,3]#
for ndevices in ndevices_array:

# for distance in dist_array:
# for additional_distance in additional_distance_array:
    
    total_bits_to_sim = 10000
    bits_per_packet = 100
    packets_to_sim = 1#int(np.ceil(total_bits_to_sim/100))
    
    
    for sim_number in range(packets_to_sim):
        print("Sim number: ", sim_number)
        
        bits_sent = np.zeros((num_gcs,bits_per_packet)).astype(np.int8)
        sim_freq_offset = np.zeros(num_gcs)
        sim_frac_delay = np.zeros(num_gcs)
        sim_whole_delay = np.zeros(num_gcs).astype(np.int32)   
        sim_distance = np.zeros(num_gcs)   


        preamble = np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])
        sync_word = np.array([0,1,0,1,1,0,0,1,1,1,1,1,0,0,0,0])
        # sync_word = np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])
        # bits_sent = np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
        #                     1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
        #                     1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
        #                     1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
        #                     1,1,1,1])
        
        
        # rx_samples_sim = np.zeros(0).astype(np.complex64)
        
        for sim_gc in range(ndevices):#range(num_gcs):
            print("Simulating goldcode: ", sim_gc)
            bits_sent[sim_gc] = np.random.randint(0,2,bits_per_packet)

            raw_data = np.concatenate((preamble, sync_word, bits_sent[sim_gc]))
        
            print("raw_data: ")
            print_bits(raw_data, 16)

        # sim_gc = 0#np.random.randint(0,num_gcs)
            tx_packet = generate_ideal_packet(GCs, sim_gc, raw_data, sps)
            tx_packet_len = len(tx_packet)
            print("tx_packet length: ", len(tx_packet))

            # if sim_gc==0:
            #     sim_freq_offset[sim_gc] = 160
            # else:
            sim_freq_offset[sim_gc] = 0#np.random.randint(-1e3,1e3)
            # sim_freq_offset[sim_gc] = np.random.randint(-1e2,1e2)
            sim_frac_delay[sim_gc] = np.random.random() #float between 0 and 1
            # sim_whole_delay[sim_gc] = sps*gc_len*2#
            # sim_whole_delay[sim_gc] = np.random.randint(0,tx_packet_len)
            sim_whole_delay[sim_gc] = np.random.randint(0,sps*gc_len)
            # sim_noise_power = 1.25e-9#1e-9 #mW
            sim_noise_power = 0#1e-9 #mW
        # sim_distance = 20#14.68#2
            sim_tx_power = 30 #mW
            sim_distance[sim_gc] = distance 
            if sim_gc > 0:
                sim_distance[sim_gc] += additional_distance

            print("Non-idealities:")
            print("Frequency offset: ", sim_freq_offset[sim_gc])
            print("Fractional delay: ", sim_frac_delay[sim_gc])
            print("Whole delay: ", sim_whole_delay[sim_gc])
            print("Noise power: ", sim_noise_power)
            print("Distance: ", sim_distance[sim_gc])
            print("Transmit power: ", sim_tx_power)
            
            if sim_gc == 0:
                rx_samples_sim = add_nonidealities(tx_packet, sim_freq_offset[sim_gc],fs, sim_frac_delay[sim_gc], sim_whole_delay[sim_gc], sim_noise_power, sim_distance[sim_gc], sim_tx_power)
            else:
                rx_samples_sim = rx_samples_sim + 1*add_nonidealities(tx_packet, sim_freq_offset[sim_gc],fs, sim_frac_delay[sim_gc], sim_whole_delay[sim_gc], sim_noise_power, sim_distance[sim_gc], sim_tx_power)



        # fname = "/home/backscatter/Documents/Backscatter/PracBackscatter25/PythonSDR/YamanReview20250223/live_samps/rtlsdr_GC7-2_100bit_252315k_915M_40_20250223171627.dat"
        # fname = "/home/backscatter/Documents/Backscatter/PracBackscatter25/PythonSDR/YamanReview20250223/live_samps/rtlsdr_GC7-2_100bit_252315k_915M_40_20250223181622.dat"

        # #devices 0 and 2 transmitting simultaneously
        # fname = "/home/backscatter/Documents/Backscatter/PracBackscatter25/PythonSDR/YamanReview20250223/live_samps/rtlsdr_GC7-0+2_100bit_252315k_915M_40_20250223185135.dat"

        # rx_samples_lab = read_live_samples()
        # rx_samples_lab = read_from_file(fname)

        # plot_time_psd_scat(rx_samples_lab,fs, "raw samples")
        plt.psd(rx_samples_sim, NFFT=1024, Fs=fs)

        # plt.show()

        ## use ~8 CDMA signals to determine frequency offset and whether there is a signal there
        max_freq_dev = 2e3
        nfreq_pts = 101
        # nfreq_pts = 101
        resamp_rate = 2
        # resamp_rate = 4

        # signal_threshold = 2e3
        # signal_threshold = 500
        # signal_threshold = 1000
        signal_threshold = 300*resamp_rate
        sync_corr_threshold =14

        start_offset = 0
        nbits_test = 4
        ntest_samples = nbits_test*gc_len*sps

        #create arrays to track each goldcode
        # next_start_offset = np.zeros(num_gcs)

        rx_samples = rx_samples_sim

        print("rx_samples length: ", len(rx_samples))

        # end_offset = 
        packet_found = np.zeros(num_gcs)
        next_start_offset = np.zeros(num_gcs)
        
        while(start_offset < 1):
        # while((start_offset+gc_len*sps*len(raw_data+1)) < len(rx_samples)):
            print("start offset: ", start_offset)
            test_samples = rx_samples[start_offset:start_offset+ntest_samples]
            
            
            test_samples = lpf(test_samples, datarate, fs)
            test_samples = agc(test_samples, 1/np.sqrt(2))
            
            
            max_corr = np.zeros(num_gcs)
            freq_offset = np.zeros(num_gcs)
            samps_offset = np.zeros(num_gcs)
            
            for gc in range(ndevices):#range(num_gcs):
                
                if next_start_offset[gc] <= start_offset:
                    # max_corr[gc], freq_offset[gc], samps_offset[gc] = freq_cdma_search(test_samples, GCs[gc], fs, max_freq_dev, nfreq_pts, resamp_rate)
                    max_corr[gc], freq_offset[gc], samps_offset[gc] = plot_freq_cdma_search(test_samples, GCs[gc], fs, max_freq_dev, nfreq_pts, resamp_rate)
                    print("*** Goldcode: ",gc, "***\nmax_corr: ",max_corr[gc], "freq_offset: ",freq_offset[gc], "samps_offset: ", samps_offset[gc])
                    # plt.show()
                
                    if max_corr[gc] > signal_threshold:
                        print("Signal detected for goldcode: ",gc, "with correlation: ",max_corr[gc])
                        
                        # plot_freq_cdma_search(test_samples, GC7_2_bip, fs, max_freq_dev, nfreq_pts, resamp_rate)
                        
                        ##advance to the start of the packet and try to demod
                        packet_start = int(start_offset + samps_offset[gc])#-gc_len*sps*nbits_test/2)
                        # packet_start = int(start_offset + samps_offset[gc]-gc_len*sps)
                        packet_end = packet_start + gc_len*sps*len(raw_data+2)
                        
                        print("packet start: ", packet_start)
                        print("packet end: ", packet_end)
                        
                        # bits_out, peak_sync_corr = demod_packet_plot(rx_samples[packet_start:packet_end], GCs[gc], fs, datarate, sps, sync_word, freq_offset[gc], len(bits_sent[gc]))
                        # bits_out, peak_sync_corr = demod_packet(rx_samples[packet_start:packet_end], GCs[gc], fs, datarate, sps, sync_word, freq_offset[gc], len(bits_sent[gc]))
                        bits_out, peak_sync_corr = demod_packet_quiet(rx_samples[packet_start:packet_end], GCs[gc], fs, datarate, sps, sync_word, freq_offset[gc], len(bits_sent[gc]))
                        
                        print("Correlation with sync word: ", peak_sync_corr)
                        # plt.show()
                        # print("Payload bits: ")
                        # print_bits(bits_out, 16)
                        
                        if (np.abs(peak_sync_corr) >= sync_corr_threshold):
                            print("Sync word detected")
                            print("Payload bits: ")
                            print_bits(bits_out, 16)
                            packet_found[gc] = 1
                            
                            
                            try:
                                
                                print("BER: ",np.sum(np.abs(bits_sent[gc]-bits_out))/len(bits_sent[gc]))
                                with open(fname, "a") as f:
                                    f.write(f"{ndevices}, {gc}, {sim_distance[gc]}, {sim_tx_power},{sim_noise_power}, {sim_freq_offset[gc]}, {sim_frac_delay[gc]}, {sim_whole_delay[gc]},  {len(bits_sent[gc])}, {np.sum(np.abs(bits_sent[gc]-bits_out))}\n")
                                f.close()
                            except:
                                print("Error in calculating BER")
                                print("bits_sent: ",bits_sent[gc])
                                print("bits_out: ",bits_out)
                                print("len(bits_sent): ",len(bits_sent[gc]))
                                print("len(bits_out): ",len(bits_out))
                                
                                with open(fname, "a") as f:
                                    f.write(f"{ndevices}, {gc}, {sim_distance[gc]},{sim_tx_power},{sim_noise_power}, {sim_freq_offset[gc]}, {sim_frac_delay[gc]}, {sim_whole_delay[gc]},  {len(bits_sent[gc])}, {len(bits_sent[gc])}\n")
                                f.close()
                                
                                
                            #skip ahead by a whole packet
                            next_start_offset[gc] = start_offset + gc_len*sps*len(raw_data)
                            
                            # plt.show()
                            
                        else:
                            print("Sync word not detected")
                        
                        # plt.show()
                    
                    
                
                
                # print("moving forward to try new bits")
            start_offset = start_offset + gc_len*sps*2
                
            
        for gc in range(ndevices):
            print("next start offset: ", next_start_offset[gc])
            if (packet_found[gc] ==0):
                print("No packet found")
                with open(fname, "a") as f:
                    f.write(f"{ndevices}, {gc}, {sim_distance[gc]}, {sim_tx_power},{sim_noise_power}, {sim_freq_offset[gc]}, {sim_frac_delay[gc]}, {sim_whole_delay[gc]},  {len(bits_sent[gc])}, {len(bits_sent[gc])}\n")
                f.close()
                    
        print("finished reading all samples")
        start_offset = 0

