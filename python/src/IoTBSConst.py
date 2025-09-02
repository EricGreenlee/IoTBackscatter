from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


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
    total_duration_sec: float 
    
    # computed fields
    actual_bits: np.ndarray = field(init=False)
    all_bits: np.ndarray = field(init=False)   # concatenated bits
    goldcode: List[int] = field(init=False)     
    
    samples: Optional[List[float]] = field(default=None)  # computed later

    def __post_init__(self):
        # Concatenate preamble, sync, and payload
        self.goldcode = GCs[self.tag_id]
        self.actual_bits = np.concatenate([self.preamble_bits, self.sync_bits, self.payload_bits])
        self.all_bits = np.concatenate([self.actual_bits, self.pad_bits])
        
    def gen_ideal_samples(self):
        # self.samples = np.zeros(self.sps*len(self.goldcode)*len(self.all_bits)).astype(np.complex64)
        self.samples = np.zeros(int(self.total_duration_sec*self.samplerate_hz)).astype(np.complex64)
        for i, bit in enumerate(self.all_bits):
            self.samples[i*self.sps*len(self.goldcode):(i+1)*self.sps*len(self.goldcode)] = np.repeat(self.goldcode.astype(np.complex64)*(bit*2-1),self.sps)


# Non-Ideal Simulated Packet
@dataclass
class SimulatedPacketParams(PacketParams):
    snr_db: float = 30.0          # Signal-to-noise ratio in dB
    frequency_offset_hz: float = 0.0  # Frequency offset in Hz
    time_delay_sec: float = 0.0       # Time delay in seconds
    oneway_tag2modem_dist_m: float = 0

    def gen_nonideal_samples(self):
        self.gen_ideal_samples()
        
        #attenuate signal
        
        #random time delay by full samples
        samps_2_shift = int(self.time_delay_sec*self.samplerate_hz)
        # print(f"roll by {samps_2_shift} samples")
        self.samples = np.roll(self.samples,samps_2_shift)
        
        #time/phase delay by partial samples
        
        #frequency drift
        time_sec = np.linspace(0, self.total_duration_sec -1/self.samplerate_hz,int(self.total_duration_sec *self.samplerate_hz))
        self.samples = self.samples*np.exp(1j*2*np.pi*self.frequency_offset_hz*time_sec)
        
        #add background noise
        noise_pwr_raw = 10**(-self.snr_db/10)
        self.samples = self.samples + np.random.normal(0,noise_pwr_raw, self.samples.shape)+1j*np.random.normal(0,noise_pwr_raw, self.samples.shape)
        

    def summary(self):
        return (f"SimPacket: bits={self.all_bits},\n "
                f"Goldcode={self.goldcode},\n"
                f"samples_per_signal={self.sps}, "
                f"SNR={self.snr_db} dB, freq_offset={self.frequency_offset_hz} Hz, "
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
            sum_str = sum_str + f"\nid = {cur_tag.tag_id}\ttime_delay_sec={round(cur_tag.time_delay_sec,2)}"\
            f"\tfrequency_offset_hz={cur_tag.frequency_offset_hz}\tsnr_db:{cur_tag.snr_db}"
        
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
preamble = np.array([1,1,1,1,1,1,1,1,1,0,1,0,1,0,1,0])
sync_seq = np.array([0,1,0,1,1,0,0,1,1,1,1,1,0,0,0,0])
bitsPerPacket = 16