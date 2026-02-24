#!/usr/bin/env python3
"""
Create a binary test file for preamble detection testing.
Generates: preamble + 16 custom bits + padding zeros
"""

import numpy as np

def hex_to_bits(hex_string):
    """Convert hex string to list of bits (MSB first)"""
    hex_string = hex_string.replace(' ', '').replace('0x', '')

    bits = []
    for hex_char in hex_string:
        val = int(hex_char, 16)
        # Convert to 4 bits (MSB first)
        for i in range(3, -1, -1):
            bits.append((val >> i) & 1)
    return bits

def create_test_file():
    """Create binary test file with preamble, payload, and padding"""

    # Preamble: AAAAAAAAAAAAAAAA (alternating 1010...)
    preamble_hex = "AAAAAAAAAAAAAAAA"
    preamble_bits = hex_to_bits(preamble_hex)

    # 16-bit custom payload - let's use 0xDEAD = 1101111010101101
    payload_bits = hex_to_bits("DEAD")

    # Add some padding zeros (1000 bits)
    padding_bits = [0] * 1000

    # Combine all bits
    all_bits = preamble_bits + payload_bits + padding_bits

    print(f"Preamble ({len(preamble_bits)} bits): {preamble_bits}")
    print(f"Payload ({len(payload_bits)} bits): {payload_bits}")
    print(f"Payload hex: 0xDEAD = {0xDEAD} decimal")
    print(f"Total bits: {len(all_bits)}")

    # Convert to numpy array of bytes (0 or 1)
    bit_array = np.array(all_bits, dtype=np.uint8)

    # Save as binary file
    output_file = '/home/backscatter/Documents/Backscatter/GNURadio/2026/test_preamble.bin'
    bit_array.tofile(output_file)

    print(f"\nTest file created: {output_file}")
    print(f"File size: {len(bit_array)} bytes")

    # Also create a text version for verification
    text_file = '/home/backscatter/Documents/Backscatter/GNURadio/2026/test_preamble.txt'
    with open(text_file, 'w') as f:
        f.write("Preamble (64 bits):\n")
        f.write(''.join(map(str, preamble_bits)) + "\n\n")
        f.write("Payload (16 bits):\n")
        f.write(''.join(map(str, payload_bits)) + "\n\n")
        f.write("All bits:\n")
        # Write in groups of 64 for readability
        for i in range(0, len(all_bits), 64):
            chunk = all_bits[i:i+64]
            f.write(''.join(map(str, chunk)) + "\n")

    print(f"Text version created: {text_file}")

    return preamble_bits, payload_bits, all_bits

def create_multiple_packets():
    """Create test file with multiple preamble+payload sequences"""

    # Preamble: AAAAAAAAAAAAAAAA
    preamble_hex = "AAAAAAAAAAAAAAAA"
    preamble_bits = hex_to_bits(preamble_hex)

    # Multiple payloads to test
    payloads = ["DEAD", "BEEF", "CAFE", "BABE"]

    all_bits = []

    for payload_hex in payloads:
        payload_bits = hex_to_bits(payload_hex)

        # Add some random noise/zeros between packets
        noise = [0] * 50  # 50 zero bits between packets

        all_bits.extend(preamble_bits + payload_bits + noise)

        print(f"Added packet: Preamble + {payload_hex} + padding")

    # Final padding
    all_bits.extend([0] * 500)

    # Convert to numpy array
    bit_array = np.array(all_bits, dtype=np.uint8)

    # Save multi-packet file
    output_file = '/home/backscatter/Documents/Backscatter/GNURadio/2026/test_multiple_packets.bin'
    bit_array.tofile(output_file)

    print(f"\nMultiple packet test file created: {output_file}")
    print(f"File size: {len(bit_array)} bytes")
    print(f"Expected detections: {len(payloads)} packets")

    return all_bits

if __name__ == "__main__":
    print("=== Creating Test Files for Preamble Detection ===")

    # Create single packet test file
    preamble, payload, all_bits = create_test_file()

    print("\n" + "="*50)

    # Create multiple packet test file
    create_multiple_packets()

    print("\n=== Files Ready ===")
    print("Use test_preamble.bin for single packet testing")
    print("Use test_multiple_packets.bin for multiple packet testing")
    print("Both files use preamble: AAAAAAAAAAAAAAAA (alternating 1010...)")