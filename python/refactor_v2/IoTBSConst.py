from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from scipy import signal

 #Preamble and sync words
# preamble = np.array([1,1,1,1,1,1,1,1,1,0,1,0,1,0,1,0])
# sync_seq = np.array([0,1,0,1,1,0,0,1,1,1,1,1,0,0,0,0])
preamble = np.array([1,1,1,1,1,1,1,1,1,0,1,0,1,0,1,0,1,1,1,1,1,1,1,1,1,0,1,0,1,0,1,0,1,1,1,1,1,1,1,1,1,0,1,0,1,0,1,0,1,1,1,1,1,1,1,1,1,0,1,0,1,0,1,0])
sync_seq = np.array([0,1,1,1,0,0,0,1,0,1,1,1,0,1,0,1])
bitsPerPacket = 16
packet_on_ratio = 0.2

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

# gc_len = len(GCs[0])
# num_gcs = len(GCs)

@dataclass
class TagParam:
    id: int           # sync sequence
    payload_bits: np.ndarray
    
    #computed fields
    goldcode: List[int] = field(init=False)  
    preamble_bits: np.ndarray = field(init=False)  
    sync_bits: np.ndarray = field(init=False)
    all_bits: np.ndarray = field(init=False)
    
    def __post_init__(self):
        self.goldcode = GCs[self.id]
        self.preamble_bits = preamble
        self.sync_bits = sync_seq
        self.all_bits = np.concatenate([self.preamble_bits, self.sync_bits, self.payload_bits])

@dataclass
class AllTagParams:
    tagParams: List[TagParam] = field(default_factory=list)
    ntags: int = field(default=0)
    
    def add_tag(self, tag: TagParam):
        """Add a tag and increment the count"""
        self.tagParams.append(tag)
        self.ntags += 1
    
    def get_tag(self, index: int) -> TagParam:
        """Get tag by index"""
        if 0 <= index < len(self.tagParams):
            return self.tagParams[index]
        raise IndexError(f"Tag index {index} out of range")

    def summary(self):
        sum_str = ""
        for tag in range(self.ntags):
            cur_tag = self.tagParams[tag]
            sum_str = sum_str + f"\nid = {cur_tag.id}\n\tbits={cur_tag.all_bits}\n\tgoldcode={cur_tag.goldcode}"
        
        return sum_str
    
@dataclass
class SimPacket:
    tagParam: TagParam
    ideal_sps: int
    carrier_freq_hz: int
    samplerate_hz: int
    actual_sps: float
    tx_pwr_dbm: float
    noise_pwr_dbm: float
    tag2modem_dist_m: float
    freq_offset_hz: float
    time_offset_sec: float
    
    #derived values
    samples: Optional[np.complex64] = field(default=None)  # computed later
    
    def gen_ideal_samples(self, tot_num_samples):
        self.samples = np.zeros(tot_num_samples).astype(np.complex64)
        sps = self.ideal_sps
        gc = self.tagParam.goldcode
        gc_len = len(gc)
        for i, bit in enumerate(self.tagParam.all_bits):
            self.samples[i*sps*gc_len:(i+1)*sps*gc_len] = np.repeat(gc.astype(np.complex64)*(bit*2-1),sps)
        
    def gen_nonideal_samples(self, tot_num_samples):
        self.gen_ideal_samples(tot_num_samples)
        
        amplitude_mw = 10**(self.tx_pwr_dbm/10)
        self.samples = amplitude_mw * self.samples
        
        print(f"carrier_freq_hz: {self.carrier_freq_hz}")
        print(f"self.tag2modem_dist_m: {self.tag2modem_dist_m}")
        
        # Distance-based attenuation (free space path loss)
        if self.tag2modem_dist_m > 0:
            # Free space path loss: PL(dB) = 20*log10(4π*d*f/c)   
            c = 3e8  # speed of light
            oneway_path_loss_db = 20 * np.log10(4 * np.pi * self.tag2modem_dist_m * self.carrier_freq_hz / c)
            roundtrip_path_loss_db = 2 * oneway_path_loss_db
            attenuation_linear = 10**(-roundtrip_path_loss_db/10)
            self.samples = self.samples * attenuation_linear
        
        # Time delay - integer and fractional parts
        total_delay_samples = self.time_offset_sec * self.samplerate_hz
        self.integer_delay_samples = int(total_delay_samples)
        self.fractional_delay_samples = total_delay_samples - self.integer_delay_samples
        
        # Integer delay using np.roll
        self.samples = np.roll(self.samples, self.integer_delay_samples)
        
        # Fractional delay using interpolation
        if self.fractional_delay_samples != 0:
            delay_filter_length = 41  # odd number for symmetric filter
            n = np.arange(-delay_filter_length//2, delay_filter_length//2) # ...-3,-2,-1,0,1,2,3...
            h = np.sinc(n - self.fractional_delay_samples) # calc filter taps
            h *= np.hamming(delay_filter_length) # window the filter to make sure it decays to 0 on both sides
            h /= np.sum(h) # normalize to get unity gain, we don't want to change the amplitude/power
            # out_samples = np.convolve(samples, h) # apply filter
            self.samples = signal.convolve(self.samples, h, mode='same')
        
        #frequency drift
        time_sec = np.linspace(0, (tot_num_samples-1)/self.samplerate_hz,tot_num_samples)
        self.samples = self.samples*np.exp(1j*2*np.pi*self.freq_offset_hz*time_sec)
        
        #resample
        resamp_ratio = self.actual_sps/self.ideal_sps
        resamp_samples = signal.resample_poly(self.samples, int(resamp_ratio*1000), 1000)
        if len(resamp_samples) > len(self.samples):
            self.samples = resamp_samples[0:len(self.samples)]
        else:
            self.samples = np.concatenate([resamp_samples, np.zeros(len(self.samples)-len(resamp_samples)).astype(np.complex64)])
        
        #add background noise
        noise_pwr_raw = 10**(self.noise_pwr_dbm/10)
        self.samples = self.samples + np.random.normal(0,noise_pwr_raw, self.samples.shape)+1j*np.random.normal(0,noise_pwr_raw, self.samples.shape)
        
        
        
        # temp_samples = np.zeros(tot_num_samples).astype(np.complex64)
        # temp_samples[] = self.samples
        
        # self.samples = temp_samples
    
    def summary(self):
        return str(f"id: {self.tagParam.id}, "\
            f"actual_sps: {self.actual_sps}, "\
            f"tx_pwr_dbm: {self.tx_pwr_dbm}, "\
            f"noise_pwr_dbm: {self.noise_pwr_dbm}, "\
            f"tag2modem_dist_m: {self.tag2modem_dist_m}, "\
            f"freq_offset_hz: {self.freq_offset_hz}, "\
            f"time_offset_sec: {self.time_offset_sec}")
        
@dataclass
class AllSimPackets:
    packetParams: List[SimPacket] = field(default_factory=list)
    npackets: int = field(default=0)
    
    def add_packet_meta(self, packet_meta: SimPacket):
        """Add a packet and increment the count"""
        self.packetParams.append(packet_meta)
        self.npackets += 1
    
    def get_packet_meta(self, index: int) -> TagParam:
        """Get tag by index"""
        if 0 <= index < len(self.packetParams):
            return self.packetParams[index]
        raise IndexError(f"Packet index {index} out of range")

    def summary(self):
        sum_str = ""
        for tag in range(self.npackets):
            cur_packet = self.packetParams[tag]
            sum_str = sum_str + f"\n{cur_packet.summary()}"
        
        return sum_str
    