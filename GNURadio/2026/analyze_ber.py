#!/usr/bin/env python3
"""
Bit Error Rate (BER) analysis for preamble detection output.
Analyzes sync_payload_output.txt and diagnoses common error types.
"""

import re
import numpy as np

def hamming_distance(bits1, bits2):
    """Calculate Hamming distance between two bit strings"""
    if len(bits1) != len(bits2):
        return float('inf')
    return sum(b1 != b2 for b1, b2 in zip(bits1, bits2))

def analyze_bit_shift(received, expected):
    """Check for bit shifts (timing errors)"""
    min_errors = float('inf')
    best_shift = 0

    # Try shifts from -8 to +8 bits
    for shift in range(-8, 9):
        if shift == 0:
            shifted = received
        elif shift > 0:
            # Right shift (delay) - pad with zeros at start
            shifted = '0' * shift + received[:-shift] if len(received) > shift else '0' * len(received)
        else:
            # Left shift (advance) - pad with zeros at end
            shift_amount = abs(shift)
            shifted = received[shift_amount:] + '0' * shift_amount if len(received) > shift_amount else '0' * len(received)

        # Ensure same length
        if len(shifted) != len(expected):
            continue

        errors = hamming_distance(shifted, expected)
        if errors < min_errors:
            min_errors = errors
            best_shift = shift

    return best_shift, min_errors

def analyze_inversion(received, expected):
    """Check for bit inversion (phase errors)"""
    # Invert all bits
    inverted = ''.join('1' if b == '0' else '0' for b in received)
    errors = hamming_distance(inverted, expected)
    return errors

def analyze_partial_inversion(received, expected):
    """Check for partial inversions (part of sequence inverted)"""
    best_errors = float('inf')
    best_pattern = None

    # Try different inversion patterns
    for start in range(len(received)):
        for length in range(1, min(8, len(received) - start + 1)):
            # Create partially inverted version
            modified = list(received)
            for i in range(start, min(start + length, len(received))):
                modified[i] = '1' if modified[i] == '0' else '0'

            modified_str = ''.join(modified)
            errors = hamming_distance(modified_str, expected)

            if errors < best_errors:
                best_errors = errors
                best_pattern = f"bits {start}-{start+length-1} inverted"

    return best_pattern, best_errors

def extract_bits_from_line(line):
    """Extract bit pattern from a payload line"""
    # Look for patterns like "1001001110001111 (Value: 12345)"
    match = re.search(r'([01]{16})', line)
    if match:
        return match.group(1)
    return None

def analyze_ber():
    """Main BER analysis function"""
    expected_bits = "1001001110001111"  # Expected payload
    expected_value = int(expected_bits, 2)

    print("=== Bit Error Rate (BER) Analysis ===")
    print(f"Expected payload: {expected_bits}")
    print(f"Expected value: {expected_value}")
    print(f"Expected hex: 0x{expected_value:04X}")
    print()

    try:
        with open('/home/backscatter/Documents/Backscatter/GNURadio/2026/sync_payload_output.txt', 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print("Error: sync_payload_output.txt not found")
        return

    if not lines:
        print("Error: sync_payload_output.txt is empty")
        return

    total_payloads = 0
    total_errors = 0
    error_analysis = []

    for line_num, line in enumerate(lines, 1):
        line = line.strip()
        if not line or not line.startswith('Payload'):
            continue

        received_bits = extract_bits_from_line(line)
        if not received_bits:
            print(f"Line {line_num}: Could not extract bits from: {line}")
            continue

        total_payloads += 1

        # Basic error count
        bit_errors = hamming_distance(received_bits, expected_bits)
        total_errors += bit_errors

        print(f"Payload {total_payloads}, Received: {received_bits}, Errors: {bit_errors}/{len(expected_bits)} bits")

        if bit_errors > 0:
            # Analyze error types
            analysis = {
                'raw_errors': bit_errors,
                'received': received_bits,
                'line_num': line_num
            }

            # Check for bit shifts
            best_shift, shift_errors = analyze_bit_shift(received_bits, expected_bits)
            analysis['best_shift'] = best_shift
            analysis['shift_errors'] = shift_errors

            # Check for inversion
            inversion_errors = analyze_inversion(received_bits, expected_bits)
            analysis['inversion_errors'] = inversion_errors

            # Check for partial inversion
            partial_pattern, partial_errors = analyze_partial_inversion(received_bits, expected_bits)
            analysis['partial_pattern'] = partial_pattern
            analysis['partial_errors'] = partial_errors

            # Diagnose most likely cause
            min_errors = min(bit_errors, shift_errors, inversion_errors, partial_errors)

            if min_errors == 0:
                print("  → Perfect match found with correction!")
            elif shift_errors < bit_errors and best_shift != 0:
                print(f"  → Likely timing error: {best_shift} bit shift reduces errors to {shift_errors}")
            elif inversion_errors < bit_errors:
                print(f"  → Likely phase error: inversion reduces errors to {inversion_errors}")
            elif partial_errors < bit_errors:
                print(f"  → Likely partial phase error: {partial_pattern} reduces errors to {partial_errors}")
            else:
                print("  → Random bit errors detected")

            # Show bit-by-bit comparison
            print("  Bit-by-bit:")
            print("    Pos: " + "".join(f"{i%10}" for i in range(16)))
            print("    Exp: " + expected_bits)
            print("    Rec: " + received_bits)
            print("    Err: " + "".join("^" if r != e else " " for r, e in zip(received_bits, expected_bits)))

            error_analysis.append(analysis)
        # else:
        #     print("  → Perfect match!")

        print()

    # Summary statistics
    print("=== SUMMARY ===")
    print(f"Total payloads analyzed: {total_payloads}")
    print(f"Total bit errors: {total_errors}")
    print(f"Total bits: {total_payloads * 16}")

    if total_payloads > 0:
        ber = total_errors / (total_payloads * 16)
        per = len(error_analysis) / total_payloads

        print(f"Bit Error Rate (BER): {ber:.6f} ({ber*100:.4f}%)")
        print(f"Packet Error Rate (PER): {per:.6f} ({per*100:.2f}%)")

        if error_analysis:
            print("\n=== ERROR PATTERN ANALYSIS ===")

            # Count error types
            shift_improvements = sum(1 for a in error_analysis if a['shift_errors'] < a['raw_errors'])
            inversion_improvements = sum(1 for a in error_analysis if a['inversion_errors'] < a['raw_errors'])

            print(f"Packets improved by shift correction: {shift_improvements}/{len(error_analysis)}")
            print(f"Packets improved by inversion: {inversion_improvements}/{len(error_analysis)}")

            # Most common shifts
            shifts = [a['best_shift'] for a in error_analysis if a['shift_errors'] < a['raw_errors']]
            if shifts:
                from collections import Counter
                shift_counts = Counter(shifts)
                print("Most common timing shifts:")
                for shift, count in shift_counts.most_common(3):
                    print(f"  {shift:+d} bits: {count} occurrences")
    else:
        print("No payloads found to analyze")

if __name__ == "__main__":
    analyze_ber()