#!/usr/bin/env python3
"""
CDMA BPSK Sample Generator for STM32
Generates precomputed samples for flash storage
"""

import numpy as np
import math

# Signal parameters (match STM32 defines)
SAMPLE_RATE = 200000      # 200ksps
CARRIER_FREQ = 50000      # 50kHz carrier
SAMPLES_PER_CYCLE = SAMPLE_RATE // CARRIER_FREQ  # 4 samples per cycle
SAMPLES_PER_BIT = 16      # 16 samples per chip = 80µs = 12.5kbps
DAC_MAX = 4095            # 12-bit DAC

# Frame structure
PREAMBLE_LENGTH = 64
PAYLOAD_LENGTH = 16
GOLDCODE_LENGTH = 127

#device_id determines which goldcode to use
device_id = 0

# Data arrays
preamble = [1,1,1,1,1,1,1,1,1,0,1,0,1,0,1,0,1,1,1,1,1,1,1,1,1,0,1,0,1,0,1,0,
           1,1,1,1,1,1,1,1,1,0,1,0,1,0,1,0,1,1,1,1,1,1,1,1,1,0,1,0,1,0,1,0]

payload = [0,1,0,0,1,0,1,0,0,1,1,0,1,1,1,1]

# goldcode = [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,1,1,-1,-1,-1,1,1,-1,1,1,1,1,1,1,-1,
#            1,1,-1,-1,-1,-1,1,1,-1,-1,1,-1,1,1,-1,-1,1,-1,-1,1,-1,1,-1,-1,-1,1,1,1,1,1,1,1,
#            -1,-1,1,1,1,1,1,1,-1,-1,-1,1,-1,1,-1,1,1,1,1,-1,1,1,1,1,1,1,-1,1,1,-1,1,-1,-1,1,
#            1,1,1,1,-1,1,-1,1,-1,-1,1,1,-1,1,-1,-1,1,1,1,1,1,1,-1,-1,1,-1,1,1,-1]

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

def generate_cdma_samples():
    """Generate two goldcode sequences: normal (for '1' bits) and inverted (for '0' bits)"""
    normal_goldcode_samples = []
    inverted_goldcode_samples = []
    break_samples = []
    
    print(f"Generating optimized CDMA goldcode samples...")
    print(f"Strategy: Store 2 goldcode sequences, select based on data bit polarity")
    print(f"Goldcode length: {GOLDCODE_LENGTH} chips")
    
    # Generate normal goldcode samples (for data bit = 1)
    print("Generating normal goldcode (for data bit '1')...")
    goldcode = GCs[device_id]
    for chip_idx, goldcode_chip in enumerate(goldcode):
        # Normal goldcode: use chip as-is
        modulated_chip = goldcode_chip
        phase_offset = 0.0 if modulated_chip == 1 else math.pi
        
        # Generate samples for this chip
        for sample_idx in range(SAMPLES_PER_BIT):
            phase = (sample_idx % SAMPLES_PER_CYCLE) * (2.0 * math.pi / SAMPLES_PER_CYCLE)
            phase += phase_offset
            
            sine_val = math.sin(phase)
            dac_value = int(2048 + (sine_val * 1862))
            dac_value = max(0, min(DAC_MAX, dac_value))
            normal_goldcode_samples.append(dac_value)
    
    # Generate inverted goldcode samples (for data bit = 0)
    print("Generating inverted goldcode (for data bit '0')...")
    for chip_idx, goldcode_chip in enumerate(goldcode):
        # Inverted goldcode: flip the chip polarity
        modulated_chip = -goldcode_chip
        phase_offset = 0.0 if modulated_chip == 1 else math.pi
        
        # Generate samples for this chip
        for sample_idx in range(SAMPLES_PER_BIT):
            phase = (sample_idx % SAMPLES_PER_CYCLE) * (2.0 * math.pi / SAMPLES_PER_CYCLE)
            phase += phase_offset
            
            sine_val = math.sin(phase)
            dac_value = int(2048 + (sine_val * 1862))
            dac_value = max(0, min(DAC_MAX, dac_value))
            inverted_goldcode_samples.append(dac_value)
            
    # Generate normal goldcode samples (for data bit = 1)
    print("Generating normal goldcode (for data bit '1')...")
    for chip_idx, goldcode_chip in enumerate(goldcode):
        # Normal goldcode: use chip as-is
        modulated_chip = goldcode_chip
        phase_offset = 0.0 if modulated_chip == 1 else math.pi
        
        # Generate samples for this chip
        for sample_idx in range(SAMPLES_PER_BIT):
            phase = (sample_idx % SAMPLES_PER_CYCLE) * (2.0 * math.pi / SAMPLES_PER_CYCLE)
            phase += phase_offset
            
            sine_val = math.sin(phase)
            dac_value = int(2048 + (sine_val * 1862))
            dac_value = max(0, min(DAC_MAX, dac_value))
            normal_goldcode_samples.append(dac_value)
            
    # Generate normal goldcode samples (for data bit = 1)
    print("Generating DC for when taking a break...")
    DC_val = 2048
    for chip_idx, goldcode_chip in enumerate(goldcode):
        # Normal goldcode: use chip as-is
        modulated_chip = goldcode_chip
        phase_offset = 0.0 if modulated_chip == 1 else math.pi
        
        # Generate samples for this chip
        for sample_idx in range(SAMPLES_PER_BIT):
            # phase = (sample_idx % SAMPLES_PER_CYCLE) * (2.0 * math.pi / SAMPLES_PER_CYCLE)
            # phase += phase_offset
            
            # sine_val = math.sin(phase)
            # dac_value = int(2048 + (sine_val * 1862))
            # dac_value = max(0, min(DAC_MAX, dac_value))
            break_samples.append(DC_val)
    
    samples_per_goldcode = len(normal_goldcode_samples)
    total_storage = samples_per_goldcode * 2  # Two sequences
    goldcode_time_ms = (samples_per_goldcode / SAMPLE_RATE) * 1000
    
    # Calculate full frame timing
    total_data_bits = PREAMBLE_LENGTH + PAYLOAD_LENGTH
    full_frame_samples = total_data_bits * samples_per_goldcode
    frame_time_ms = (full_frame_samples / SAMPLE_RATE) * 1000
    
    print(f"Generated {samples_per_goldcode} samples per goldcode sequence")
    print(f"Storage: {total_storage * 2} bytes ({total_storage * 2 / 1024:.1f} KB)")
    print(f"Goldcode time: {goldcode_time_ms:.2f} ms")
    print(f"Full frame: {total_data_bits} bits × {goldcode_time_ms:.2f}ms = {frame_time_ms:.2f} ms")
    
    return normal_goldcode_samples, inverted_goldcode_samples, break_samples
    
    # # Combine preamble and payload
    # data_bits = preamble + payload
    
    # # Process each data bit
    # for bit_idx, data_bit in enumerate(data_bits):
    #     if bit_idx < PREAMBLE_LENGTH:
    #         bit_type = "preamble"
    #     else:
    #         bit_type = "payload"
            
    #     print(f"Processing {bit_type} bit {bit_idx}: {data_bit}")
        
    #     # For each data bit, modulate the entire goldcode
    #     for chip_idx, goldcode_chip in enumerate(goldcode):
    #         # Apply data bit modulation to goldcode chip
    #         if data_bit == 1:
    #             modulated_chip = goldcode_chip      # Normal goldcode
    #         else:
    #             modulated_chip = -goldcode_chip     # Inverted goldcode
            
    #         # Convert chip to phase offset
    #         phase_offset = 0.0 if modulated_chip == 1 else math.pi
            
    #         # Generate samples for this chip (4 cycles of 50kHz carrier)
    #         for sample_idx in range(SAMPLES_PER_BIT):
    #             # Calculate phase with zero-crossing transitions
    #             phase = (sample_idx % SAMPLES_PER_CYCLE) * (2.0 * math.pi / SAMPLES_PER_CYCLE)
    #             phase += phase_offset
                
    #             # Generate ±1.5V amplitude around 1.65V center
    #             # 1.65V = 2048 DAC counts, ±1.5V = ±1862 DAC counts
    #             sine_val = math.sin(phase)
    #             dac_value = int(2048 + (sine_val * 1862))
                
    #             # Clamp to valid DAC range
    #             dac_value = max(0, min(DAC_MAX, dac_value))
    #             samples.append(dac_value)
    
    # total_samples = len(samples)
    # frame_time_ms = (total_samples / SAMPLE_RATE) * 1000
    
    # print(f"Generated {total_samples} samples")
    # print(f"Frame time: {frame_time_ms:.2f} ms")
    # print(f"Memory usage: {total_samples * 2} bytes ({total_samples * 2 / 1024:.1f} KB)")
    
    # return samples

def write_header_file(normal_samples, inverted_samples, break_samples, filename="precomputed_samples.h"):
    """Write goldcode samples to C header file"""
    
    filename = "scripts/precomputed_samples/"+filename
    print(f"Writing goldcode samples to {filename}...")
    
    with open(filename, 'w') as f:
        f.write("/* Precomputed CDMA BPSK goldcode samples for STM32 */\n")
        f.write("/* Generated by generate_samples.py */\n")
        f.write("/* Strategy: Store normal + inverted goldcode, select based on data bit */\n\n")
        
        f.write("#ifndef PRECOMPUTED_SAMPLES_H\n")
        f.write("#define PRECOMPUTED_SAMPLES_H\n\n")
        
        f.write("#include <stdint.h>\n\n")
        
        # Write parameters as defines
        f.write("/* Sample parameters */\n")
        f.write(f"#define PRECOMPUTED_SAMPLE_RATE {SAMPLE_RATE}\n")
        f.write(f"#define PRECOMPUTED_CARRIER_FREQ {CARRIER_FREQ}\n")
        f.write(f"#define PRECOMPUTED_SAMPLES_PER_BIT {SAMPLES_PER_BIT}\n")
        f.write(f"#define PRECOMPUTED_GOLDCODE_LENGTH {GOLDCODE_LENGTH}\n")
        f.write(f"#define PRECOMPUTED_SAMPLES_PER_GOLDCODE {len(normal_samples)}\n")
        f.write(f"#define PRECOMPUTED_GOLDCODE_TIME_MS {int((len(normal_samples) / SAMPLE_RATE) * 1000)}\n\n")
        
        # Write data bit arrays for reference
        f.write("/* Data sequences */\n")
        f.write("const uint8_t preamble_bits[] = {")
        f.write(", ".join(map(str, preamble)))
        f.write("};\n")
        f.write(f"#define PREAMBLE_LENGTH {PREAMBLE_LENGTH}\n\n")
        
        f.write("const uint8_t payload_bits[] = {")
        f.write(", ".join(map(str, payload)))
        f.write("};\n")
        f.write(f"#define PAYLOAD_LENGTH {PAYLOAD_LENGTH}\n\n")
        
        # Write normal goldcode samples
        f.write("/* Normal goldcode samples (for data bit '1') - stored in flash */\n")
        f.write("const uint16_t normal_goldcode_samples[] = {\n")
        
        for i, sample in enumerate(normal_samples):
            if i % 16 == 0:
                f.write("    ")
            
            f.write(f"{sample:4d}")
            
            if i < len(normal_samples) - 1:
                f.write(",")
                
            if i % 16 == 15:
                f.write("\n")
            elif i < len(normal_samples) - 1:
                f.write(" ")
        
        if len(normal_samples) % 16 != 0:
            f.write("\n")
            
        f.write("};\n\n")
        
        # Write inverted goldcode samples
        f.write("/* Inverted goldcode samples (for data bit '0') - stored in flash */\n")
        f.write("const uint16_t inverted_goldcode_samples[] = {\n")
        
        for i, sample in enumerate(inverted_samples):
            if i % 16 == 0:
                f.write("    ")
            
            f.write(f"{sample:4d}")
            
            if i < len(inverted_samples) - 1:
                f.write(",")
                
            if i % 16 == 15:
                f.write("\n")
            elif i < len(inverted_samples) - 1:
                f.write(" ")
        
        if len(inverted_samples) % 16 != 0:
            f.write("\n")
            
        f.write("};\n\n")
        
        #write break samples
        f.write("/* Break samples (DC) - stored in flash */\n")
        f.write("const uint16_t break_samples[] = {\n")
        
        for i, sample in enumerate(break_samples):
            if i % 16 == 0:
                f.write("    ")
            
            f.write(f"{sample:4d}")
            
            if i < len(break_samples) - 1:
                f.write(",")
                
            if i % 16 == 15:
                f.write("\n")
            elif i < len(break_samples) - 1:
                f.write(" ")
        
        if len(break_samples) % 16 != 0:
            f.write("\n")
            
        f.write("};\n\n")
        
        #write low samples
        f.write("/* Low samples (DC) - stored in flash */\n")
        f.write("const uint16_t low_samples[] = {\n")
        
        for i, sample in enumerate(break_samples):
            if i % 16 == 0:
                f.write("    ")
            
            f.write(f"{sample-1024:4d}")
            
            if i < len(break_samples) - 1:
                f.write(",")
                
            if i % 16 == 15:
                f.write("\n")
            elif i < len(break_samples) - 1:
                f.write(" ")
        
        if len(break_samples) % 16 != 0:
            f.write("\n")
            
        f.write("};\n\n")
        
        # Helper macro for selecting samples based on bit value
        f.write("/* Helper macro to select goldcode based on data bit */\n")
        f.write("#define GET_GOLDCODE_SAMPLES(bit) ((bit) ? normal_goldcode_samples : inverted_goldcode_samples)\n\n")
        
        f.write("#endif /* PRECOMPUTED_SAMPLES_H */\n")
    
    print(f"Header file written successfully!")
    print(f"Total storage: {(len(normal_samples) + len(inverted_samples)) * 2} bytes")

def write_debug_info(normal_samples, inverted_samples, break_samples, filename="sample_info.txt"):
    """Write debug information about the generated samples"""
    print(f"Writing debug info to {filename}...")
    
    with open(filename, 'w') as f:
        f.write("CDMA BPSK Goldcode Sample Generation Report\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Signal Parameters:\n")
        f.write(f"  Sample Rate: {SAMPLE_RATE} Hz\n")
        f.write(f"  Carrier Frequency: {CARRIER_FREQ} Hz\n")
        f.write(f"  Samples per Cycle: {SAMPLES_PER_CYCLE}\n")
        f.write(f"  Samples per Chip: {SAMPLES_PER_BIT}\n")
        f.write(f"  Chip Rate: {SAMPLE_RATE / SAMPLES_PER_BIT} Hz\n\n")
        
        f.write(f"Goldcode:\n")
        f.write(f"  Length: {GOLDCODE_LENGTH} chips\n")
        f.write(f"  Samples per goldcode: {len(normal_samples)}\n")
        f.write(f"  Goldcode time: {(len(normal_samples) / SAMPLE_RATE) * 1000:.2f} ms\n\n")
        
        f.write(f"Frame Structure:\n")
        f.write(f"  Preamble: {PREAMBLE_LENGTH} bits\n")
        f.write(f"  Payload: {PAYLOAD_LENGTH} bits\n")
        f.write(f"  Total Data Bits: {PREAMBLE_LENGTH + PAYLOAD_LENGTH}\n")
        f.write(f"  Frame Time: {((PREAMBLE_LENGTH + PAYLOAD_LENGTH) * len(normal_samples) / SAMPLE_RATE) * 1000:.2f} ms\n\n")
        
        f.write(f"Storage Optimization:\n")
        f.write(f"  Strategy: Store 2 goldcode sequences, select based on bit polarity\n")
        f.write(f"  Normal goldcode samples: {len(normal_samples)} ({len(normal_samples) * 2} bytes)\n")
        f.write(f"  Inverted goldcode samples: {len(inverted_samples)} ({len(inverted_samples) * 2} bytes)\n")
        f.write(f"  Total storage: {(len(normal_samples) + len(inverted_samples)) * 2} bytes ({(len(normal_samples) + len(inverted_samples)) * 2 / 1024:.1f} KB)\n")
        f.write(f"  Flash usage: ~{(len(normal_samples) + len(inverted_samples)) * 2 / 1024:.1f}KB vs ~406KB (98% reduction!)\n\n")
        
        f.write(f"Data Sequences:\n")
        f.write(f"  Preamble: {''.join(map(str, preamble))}\n")
        f.write(f"  Payload:  {''.join(map(str, payload))}\n\n")
        
        # Sample statistics for both sequences
        normal_min, normal_max = min(normal_samples), max(normal_samples)
        normal_mean = sum(normal_samples) / len(normal_samples)
        
        inverted_min, inverted_max = min(inverted_samples), max(inverted_samples)
        inverted_mean = sum(inverted_samples) / len(inverted_samples)
        
        f.write(f"Normal Goldcode Statistics:\n")
        f.write(f"  Min: {normal_min} ({normal_min * 3.3 / 4095:.3f}V)\n")
        f.write(f"  Max: {normal_max} ({normal_max * 3.3 / 4095:.3f}V)\n")
        f.write(f"  Mean: {normal_mean:.1f} ({normal_mean * 3.3 / 4095:.3f}V)\n\n")
        
        f.write(f"Inverted Goldcode Statistics:\n")
        f.write(f"  Min: {inverted_min} ({inverted_min * 3.3 / 4095:.3f}V)\n")
        f.write(f"  Max: {inverted_max} ({inverted_max * 3.3 / 4095:.3f}V)\n")
        f.write(f"  Mean: {inverted_mean:.1f} ({inverted_mean * 3.3 / 4095:.3f}V)\n\n")
        
        f.write("Implementation Strategy:\n")
        f.write("1. Store two goldcode sequences in flash\n")
        f.write("2. Use DMA to stream one goldcode at a time\n")
        f.write("3. Switch between normal/inverted based on data bit\n")
        f.write("4. Use DMA complete callback to load next goldcode\n")

if __name__ == "__main__":
    print("CDMA BPSK Goldcode Sample Generator")
    print("=" * 40)
    
    # Generate goldcode samples
    normal_samples, inverted_samples, break_samples = generate_cdma_samples()
    
    # Write header file
    fname = f"precomputed_samples_GC{device_id}.h"
    write_header_file(normal_samples, inverted_samples, break_samples, filename = fname)
    
    # Write debug info
    write_debug_info(normal_samples, inverted_samples, break_samples)
    
    storage_kb = (len(normal_samples) + len(inverted_samples)+ len(break_samples)) * 2 / 1024
    
    print("\nGeneration complete!")
    print(f"Include 'precomputed_samples.h' in your STM32 project")
    print(f"Total flash usage: {storage_kb:.1f}KB (fits in 64KB flash!)")
    print(f"Memory efficiency: 98% reduction vs full frame storage")