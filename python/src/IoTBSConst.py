from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from scipy import signal


# Ideal packet
@dataclass
class PacketParams:
    tag_id: int         
    preamble_bits: np.ndarray           # preamble
    sync_bits: np.ndarray               # sync sequence
    payload_bits: np.ndarray           # payload
    pad_bits: np.ndarray                #bits at the end to pad out
    sps: int            # samples per chip/bit
    samplerate_hz: int
    
    
    # computed fields
    actual_bits: np.ndarray = field(init=False)
    all_bits: np.ndarray = field(init=False)   # concatenated bits
    goldcode: List[int] = field(init=False)     
    total_duration_sec: float = field(init=False)
    
    samples: Optional[List[float]] = field(default=None)  # computed later

    def __post_init__(self):
        # Concatenate preamble, sync, and payload
        self.goldcode = GCs[self.tag_id]
        self.actual_bits = np.concatenate([self.preamble_bits, self.sync_bits, self.payload_bits])
        self.all_bits = np.concatenate([self.actual_bits, self.pad_bits])
        self.total_duration_sec = len(self.all_bits)*self.sps*len(self.goldcode)/self.samplerate_hz/packet_on_ratio
        
    def gen_ideal_samples(self):
        # self.samples = np.zeros(self.sps*len(self.goldcode)*len(self.all_bits)).astype(np.complex64)
        self.samples = np.zeros(int(self.total_duration_sec*self.samplerate_hz)).astype(np.complex64)
        for i, bit in enumerate(self.all_bits):
            self.samples[i*self.sps*len(self.goldcode):(i+1)*self.sps*len(self.goldcode)] = np.repeat(self.goldcode.astype(np.complex64)*(bit*2-1),self.sps)


# Non-Ideal Simulated Packet
@dataclass
class SimulatedPacketParams(PacketParams):
    # snr_db: float = 30.0          # Signal-to-noise ratio in dB
    tx_pwr_dbm: float = 30.0
    noise_pwr_dbm: float = 0
    oneway_tag2modem_dist_m: float = 0
    frequency_offset_hz: float = 0.0  # Frequency offset in Hz
    time_delay_mode: str = "rand"      # Time delay in seconds

    # computed fields
    time_delay_sec: float = field(init=False)      # Time delay in seconds
    integer_delay_samples: int = field(init=False)
    fractional_delay_samples: float = field(init=False)
    
    def __post_init__(self):
        # Call parent's __post_init__ first to set up all_bits, goldcode, etc.
        super().__post_init__()
        
        if self.time_delay_mode == "zero":
            self.time_delay_sec = 0
        elif self.time_delay_mode == "rand":
            self.time_delay_sec = np.random.uniform(0.0, self.total_duration_sec*(1-packet_on_ratio))
        else:
            print("**** ERROR: time_delay_mode invalid ****")
            self.time_delay_sec = 0

    def gen_nonideal_samples(self):
        
        # Generate the ideal samples at the tx power
        self.gen_ideal_samples()
        amplitude_mw = 10**(self.tx_pwr_dbm/10)
        self.samples = amplitude_mw * self.samples
        
        # Distance-based attenuation (free space path loss)
        if self.oneway_tag2modem_dist_m > 0:
            # Free space path loss: PL(dB) = 20*log10(4π*d*f/c)
            # Assuming 915 MHz carrier frequency and speed of light
            carrier_freq_hz = 915e6
            c = 3e8  # speed of light
            oneway_path_loss_db = 20 * np.log10(4 * np.pi * self.oneway_tag2modem_dist_m * carrier_freq_hz / c)
            roundtrip_path_loss_db = 2 * oneway_path_loss_db
            attenuation_linear = 10**(-roundtrip_path_loss_db/20)
            self.samples = self.samples * attenuation_linear
        
        # Time delay - integer and fractional parts
        total_delay_samples = self.time_delay_sec * self.samplerate_hz
        self.integer_delay_samples = int(total_delay_samples)
        self.fractional_delay_samples = total_delay_samples - self.integer_delay_samples
        
        # Integer delay using np.roll
        self.samples = np.roll(self.samples, self.integer_delay_samples)
        
        # Fractional delay using interpolation
        if self.fractional_delay_samples != 0:
            # Create fractional delay filter
            
            delay_filter_length = 41  # odd number for symmetric filter
            n = np.arange(-delay_filter_length//2, delay_filter_length//2) # ...-3,-2,-1,0,1,2,3...
            h = np.sinc(n - self.fractional_delay_samples) # calc filter taps
            h *= np.hamming(delay_filter_length) # window the filter to make sure it decays to 0 on both sides
            h /= np.sum(h) # normalize to get unity gain, we don't want to change the amplitude/power
            # out_samples = np.convolve(samples, h) # apply filter
            self.samples = signal.convolve(self.samples, h, mode='same')
            
            # h = np.sinc(np.arange(-delay_filter_length//2, delay_filter_length//2 + 1) - fractional_delay_samples)
            # h *= np.hamming(delay_filter_length)  # windowing to reduce artifacts
            # h /= np.sum(h)  # normalize
            
            # Apply fractional delay filter
            # self.samples = signal.convolve(self.samples, h, mode='same')
        
        #frequency drift
        time_sec = np.linspace(0, self.total_duration_sec -1/self.samplerate_hz,int(self.total_duration_sec *self.samplerate_hz))
        self.samples = self.samples*np.exp(1j*2*np.pi*self.frequency_offset_hz*time_sec)
        
        #add background noise
        noise_pwr_raw = 10**(self.noise_pwr_dbm/10)
        self.samples = self.samples + np.random.normal(0,noise_pwr_raw, self.samples.shape)+1j*np.random.normal(0,noise_pwr_raw, self.samples.shape)
        

    def summary(self):
        return (f"SimPacket: bits={self.all_bits},\n "
                f"Goldcode={self.goldcode},\n"
                f"samples_per_signal={self.sps}, "
                f"time_delay={self.time_delay_sec}s")

@dataclass
class TagParams:
    tagParams: List[SimulatedPacketParams] = field(default_factory=list)
    ntags: int = field(default=0)
    
    def add_tag(self, tag: SimulatedPacketParams):
        """Add a tag and increment the count"""
        self.tagParams.append(tag)
        self.ntags += 1
    
    def get_tag(self, index: int) -> SimulatedPacketParams:
        """Get tag by index"""
        if 0 <= index < len(self.tagParams):
            return self.tagParams[index]
        raise IndexError(f"Tag index {index} out of range")

    def summary(self):
        sum_str = ""
        for tag in range(self.ntags):
            cur_tag = self.tagParams[tag]
            sum_str = sum_str + f"\nid = {cur_tag.tag_id}\tbits={cur_tag.actual_bits}"
        
        return sum_str
    
    def summary_nonideal(self):
        sum_str = ""
        for tag in range(self.ntags):
            cur_tag = self.tagParams[tag]
            sum_str = sum_str + f"\nid = {cur_tag.tag_id}"\
            f"\ttx_pwr_dbm = {cur_tag.tx_pwr_dbm}"\
            f"\tnoise_pwr_dbm = {cur_tag.noise_pwr_dbm}"\
            f"\toneway_tag2modem_dist_m = {cur_tag.oneway_tag2modem_dist_m}"\
            f"\tfrequency_offset_hz={cur_tag.frequency_offset_hz}"\
            f"\ttime_delay_sec={round(cur_tag.time_delay_sec,2)}"\
            f"\tinteger_delay_samples={cur_tag.integer_delay_samples}"\
            f"\tfractional_delay_samples={round(cur_tag.fractional_delay_samples,2)}"
        
        return sum_str
    
    def combined_samples(self):
        comb_samples = np.zeros(len(self.tagParams[0].samples))
        for tag in range(self.ntags):
            comb_samples = comb_samples + self.tagParams[tag].samples
        
        return comb_samples
    
@dataclass
class RadioSettings:
    samplerate_hz: int
    carrier_freq_hz: int

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

# GCs = np.array([
#    [-1, -1, 1, -1] 
    
# ])

gc_len = len(GCs[0])
num_gcs = len(GCs)

#Preamble and sync words
# preamble = np.array([1,1,1,1,1,1,1,1,1,0,1,0,1,0,1,0])
# sync_seq = np.array([0,1,0,1,1,0,0,1,1,1,1,1,0,0,0,0])
preamble = np.array([1,1,1,1,1,1,1,1,1,0,1,0,1,0,1,0,1,1,1,1,1,1,1,1,1,0,1,0,1,0,1,0,1,1,1,1,1,1,1,1,1,0,1,0,1,0,1,0,1,1,1,1,1,1,1,1,1,0,1,0,1,0,1,0])
sync_seq = np.array([0,1,1,1,0,0,0,1,0,1,1,1,0,1,0,1])
bitsPerPacket = 16
packet_on_ratio = 0.2