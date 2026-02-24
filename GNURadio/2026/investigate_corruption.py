#!/usr/bin/env python3
"""
Investigate payload corruption by correlating synchronized output with raw decoder output.
Focuses on specific payload instances like payload 15 with corruption.
"""

import re

def read_sync_payloads(filename):
    """Read synchronized payload output"""
    payloads = []
    try:
        with open(filename, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line.startswith('Payload'):
                    # Extract payload number, bits, and value
                    match = re.search(r'Payload (\d+): ([01]{16}) \(Value: (\d+)\)', line)
                    if match:
                        payload_num = int(match.group(1))
                        bits = match.group(2)
                        value = int(match.group(3))
                        payloads.append({
                            'payload_num': payload_num,
                            'bits': bits,
                            'value': value,
                            'line_num': line_num,
                            'raw_line': line
                        })
    except FileNotFoundError:
        print(f"File {filename} not found")
    return payloads

def read_raw_decoder_output(filename, max_chars=10000):
    """Read raw decoder output as a continuous bit stream"""
    try:
        with open(filename, 'rb') as f:
            raw_data = f.read(max_chars)

        # Convert bytes to bits (assuming each byte is 0 or 1)
        bit_stream = []
        for byte in raw_data:
            if byte in [0, 1, ord('0'), ord('1')]:
                if byte == ord('0'):
                    bit_stream.append(0)
                elif byte == ord('1'):
                    bit_stream.append(1)
                else:
                    bit_stream.append(int(byte))

        return bit_stream
    except FileNotFoundError:
        print(f"File {filename} not found")
        return []

def find_preamble_in_stream(bit_stream, preamble_hex="AAAAAAAAAAAAAAAA"):
    """Find all preamble occurrences in bit stream"""
    # Convert hex preamble to bits
    preamble_bits = []
    for hex_char in preamble_hex:
        val = int(hex_char, 16)
        for i in range(3, -1, -1):
            preamble_bits.append((val >> i) & 1)

    # Also check inverted preamble
    inverted_preamble = [1-b for b in preamble_bits]

    matches = []

    # Search for normal preamble
    for i in range(len(bit_stream) - len(preamble_bits) + 1):
        if bit_stream[i:i+len(preamble_bits)] == preamble_bits:
            matches.append({
                'position': i,
                'type': 'normal',
                'preamble_bits': preamble_bits,
                'payload_start': i + len(preamble_bits)
            })

    # Search for inverted preamble
    for i in range(len(bit_stream) - len(inverted_preamble) + 1):
        if bit_stream[i:i+len(inverted_preamble)] == inverted_preamble:
            matches.append({
                'position': i,
                'type': 'inverted',
                'preamble_bits': inverted_preamble,
                'payload_start': i + len(inverted_preamble)
            })

    # Sort by position
    matches.sort(key=lambda x: x['position'])
    return matches, preamble_bits, inverted_preamble

def analyze_payload_corruption(payload_info, bit_stream, preamble_matches):
    """Analyze specific payload corruption"""
    expected_bits = "1001001110001111"  # Expected payload pattern
    expected_value = 37775  # Decimal value of 1001001110001111

    print(f"\n=== ANALYZING PAYLOAD {payload_info['payload_num']} ===")
    print(f"Synchronized output: {payload_info['bits']} (Value: {payload_info['value']})")
    print(f"Expected output:     {expected_bits} (Value: {expected_value})")
    print(f"Raw line: {payload_info['raw_line']}")

    # Calculate error metrics
    bit_errors = sum(1 for a, b in zip(payload_info['bits'], expected_bits) if a != b)
    print(f"Bit errors: {bit_errors}/16")

    # Check if it's a simple inversion
    inverted_bits = ''.join('1' if b == '0' else '0' for b in payload_info['bits'])
    inverted_value = int(inverted_bits, 2)
    inverted_errors = sum(1 for a, b in zip(inverted_bits, expected_bits) if a != b)

    print(f"If inverted: {inverted_bits} (Value: {inverted_value}, Errors: {inverted_errors}/16)")

    # Check for bit shifts
    print("\nChecking bit shifts:")
    for shift in range(-3, 4):
        if shift == 0:
            continue

        if shift > 0:
            # Right shift - pad with zeros at start
            shifted = '0' * shift + payload_info['bits'][:-shift]
        else:
            # Left shift - pad with zeros at end
            shifted = payload_info['bits'][-shift:] + '0' * (-shift)

        if len(shifted) == 16:
            shifted_errors = sum(1 for a, b in zip(shifted, expected_bits) if a != b)
            shifted_value = int(shifted, 2)
            print(f"  Shift {shift:+d}: {shifted} (Value: {shifted_value}, Errors: {shifted_errors}/16)")

    # Try to find this payload in the raw stream
    print(f"\nSearching for payload pattern in raw stream...")
    payload_pattern = [int(b) for b in payload_info['bits']]

    found_positions = []
    for i in range(len(bit_stream) - 15):
        if bit_stream[i:i+16] == payload_pattern:
            found_positions.append(i)

    if found_positions:
        print(f"Found exact payload pattern at positions: {found_positions}")

        # Check context around each position
        for pos in found_positions[:3]:  # Limit to first 3 matches
            print(f"\nContext around position {pos}:")
            start = max(0, pos - 80)  # 64 bits before for preamble + 16 extra
            end = min(len(bit_stream), pos + 32)  # 16 bits payload + 16 extra

            context = bit_stream[start:end]
            context_str = ''.join(map(str, context))

            # Mark the payload region
            payload_start_in_context = pos - start
            marked = (context_str[:payload_start_in_context] +
                     '[' + context_str[payload_start_in_context:payload_start_in_context+16] + ']' +
                     context_str[payload_start_in_context+16:])

            print(f"  Context: {marked}")

            # Check for preamble before this payload
            preamble_search_start = max(0, pos - 70)
            preamble_region = bit_stream[preamble_search_start:pos]
            print(f"  Looking for preamble in 70 bits before payload...")

            # Check for normal preamble (64 bits = AAAA...)
            preamble_pattern = [1,0,1,0,1,0,1,0] * 8  # AAAAAAAAAAAAAAAA
            inverted_pattern = [0,1,0,1,0,1,0,1] * 8   # Inverted

            for i in range(len(preamble_region) - 63):
                if preamble_region[i:i+64] == preamble_pattern:
                    print(f"    Found normal preamble at offset {i} from payload")
                elif preamble_region[i:i+64] == inverted_pattern:
                    print(f"    Found inverted preamble at offset {i} from payload")

def investigate_specific_payloads(sync_file, raw_file, target_payloads=[15]):
    """Investigate specific payload numbers"""
    print("=== PAYLOAD CORRUPTION INVESTIGATION ===")

    # Read synchronized payloads
    sync_payloads = read_sync_payloads(sync_file)
    print(f"Found {len(sync_payloads)} synchronized payloads")

    # Read raw decoder output
    bit_stream = read_raw_decoder_output(raw_file, max_chars=50000)
    print(f"Read {len(bit_stream)} bits from raw decoder output")

    if not bit_stream:
        print("Could not read raw decoder output - trying as text file")
        try:
            with open(raw_file, 'r') as f:
                text_content = f.read(10000)  # Read first 10k chars
                # Extract 0s and 1s from text
                bit_stream = [int(c) for c in text_content if c in '01']
            print(f"Extracted {len(bit_stream)} bits from text content")
        except:
            print("Could not read raw file as text either")
            return

    # Find preambles in raw stream
    preamble_matches, normal_preamble, inverted_preamble = find_preamble_in_stream(bit_stream)
    print(f"Found {len(preamble_matches)} preamble matches in raw stream")

    print(f"Normal preamble:   {''.join(map(str, normal_preamble))}")
    print(f"Inverted preamble: {''.join(map(str, inverted_preamble))}")

    # Show preamble positions
    for i, match in enumerate(preamble_matches[:10]):  # Show first 10
        payload_end = min(match['payload_start'] + 16, len(bit_stream))
        payload_bits = bit_stream[match['payload_start']:payload_end]
        if len(payload_bits) == 16:
            payload_str = ''.join(map(str, payload_bits))
            payload_value = int(payload_str, 2)
            print(f"  Match {i+1}: pos {match['position']}, {match['type']}, "
                  f"payload: {payload_str} (value: {payload_value})")

    # Analyze specific target payloads
    for target in target_payloads:
        target_payload = next((p for p in sync_payloads if p['payload_num'] == target), None)
        if target_payload:
            analyze_payload_corruption(target_payload, bit_stream, preamble_matches)
        else:
            print(f"\nPayload {target} not found in synchronized output")

    # Summary of all payloads
    print(f"\n=== SUMMARY OF ALL {len(sync_payloads)} SYNCHRONIZED PAYLOADS ===")
    expected_value = 37775

    for payload in sync_payloads:
        status = "OK" if payload['value'] == expected_value else "CORRUPTED"
        print(f"Payload {payload['payload_num']:2d}: {payload['bits']} "
              f"(Value: {payload['value']:5d}) - {status}")

if __name__ == "__main__":
    sync_file = '/home/backscatter/Documents/Backscatter/GNURadio/2026/sync_payload_output_2.txt'
    raw_file = '/home/backscatter/Documents/Backscatter/GNURadio/2026/Simple_demod_output_20260224.txt'

    investigate_specific_payloads(sync_file, raw_file, target_payloads=[15, 1, 5, 10])