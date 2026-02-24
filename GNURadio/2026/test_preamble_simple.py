#!/usr/bin/env python3
"""
Simple test of preamble detection logic without GNURadio dependencies
"""

import numpy as np

STATE_SEARCH = 0
STATE_PAYLOAD = 1

class PreambleDetector:
    def __init__(self, preamble_hex="DEADBEEFCAFEBABE"):
        """
        Preamble detector and payload extractor

        Args:
            preamble_hex: 64-bit preamble as hex string (16 hex characters)
        """
        # Convert hex preamble to bit array (64 bits)
        self.preamble_bits = self.hex_to_bits(preamble_hex)

        # Create inverted preamble for 180-degree phase detection
        self.inverted_preamble_bits = [1-b for b in self.preamble_bits]

        # State variables
        self.payload_count = 0

        # File output
        try:
            self.output_file = open('/home/backscatter/Documents/Backscatter/GNURadio/2026/sync_payload_output.txt', 'w')
        except Exception as e:
            print(f"Could not open output file: {e}")
            self.output_file = None

        print(f"Preamble detector initialized with 64-bit preamble")
        print(f"Preamble hex: {preamble_hex}")
        print(f"Preamble bits: {self.preamble_bits}")
        print(f"Inverted bits: {self.inverted_preamble_bits}")

        self.start_search()

    def hex_to_bits(self, hex_string):
        """Convert hex string to list of bits (MSB first)"""
        # Remove any whitespace or 0x prefix
        hex_string = hex_string.replace(' ', '').replace('0x', '')

        bits = []
        for hex_char in hex_string:
            val = int(hex_char, 16)
            # Convert to 4 bits (MSB first)
            for i in range(3, -1, -1):
                bits.append((val >> i) & 1)
        return bits

    def process_bits(self, input_bits):
        """Process input bit stream"""
        results = []

        for bit in input_bits:
            bit = int(bit)

            if self.state == STATE_SEARCH:
                # Shift in new bit to search buffer
                self.search_bits = self.search_bits[1:] + [bit]

                # Check for preamble match (normal)
                if self.search_bits == self.preamble_bits:
                    print("Preamble found (normal phase)")
                    self.inverted = False
                    self.start_payload_extraction()

                # Check for preamble match (inverted)
                elif self.search_bits == self.inverted_preamble_bits:
                    print("Preamble found (inverted phase)")
                    self.inverted = True
                    self.start_payload_extraction()

            elif self.state == STATE_PAYLOAD:
                # Apply phase correction if needed
                corrected_bit = bit if not self.inverted else (1 - bit)

                self.payload_bits.append(corrected_bit)
                results.append(corrected_bit)

                # Check if we have collected 16 bits
                if len(self.payload_bits) == 16:
                    payload_value = self.process_payload()
                    self.start_search()

        return results

    def start_search(self):
        """Reset to search state"""
        self.state = STATE_SEARCH
        self.inverted = False
        self.search_bits = [0] * 64  # 64-bit rolling buffer
        self.payload_bits = []

    def start_payload_extraction(self):
        """Switch to payload extraction state"""
        self.state = STATE_PAYLOAD
        self.payload_bits = []

    def process_payload(self):
        """Process the extracted 16-bit payload"""
        # Convert bits to integer value (MSB first)
        payload_value = 0
        for i, bit in enumerate(self.payload_bits):
            payload_value += bit * (2 ** (15 - i))

        self.payload_count += 1

        # Write to file
        if self.output_file:
            self.output_file.write(f"Payload {self.payload_count}: ")
            self.output_file.write(''.join(map(str, self.payload_bits)))
            self.output_file.write(f" (Value: {payload_value})\n")
            self.output_file.flush()

        print(f"Extracted payload {self.payload_count}: {payload_value} ({''.join(map(str, self.payload_bits))})")

        return payload_value

    def __del__(self):
        """Cleanup when object is destroyed"""
        if hasattr(self, 'output_file') and self.output_file:
            self.output_file.close()


def test_preamble_detector():
    """Test function for the preamble detector"""
    print("=== Testing Preamble Detector ===")

    # Example usage
    detector = PreambleDetector(preamble_hex="DEADBEEFCAFEBABE")

    # Test data: preamble + payload
    test_payload = [1,0,1,1,0,0,1,0,1,0,1,0,1,1,0,0]  # 16-bit payload (45772 decimal)
    test_bits = detector.preamble_bits + test_payload

    print(f"\nTest input length: {len(test_bits)}")
    print(f"Preamble: {detector.preamble_bits}")
    print(f"Payload: {test_payload}")

    # Process the test data
    results = detector.process_bits(test_bits)
    print(f"Extracted bits: {results}")

    # Test inverted case
    print("\n=== Testing Inverted Preamble ===")
    inverted_test_bits = detector.inverted_preamble_bits + test_payload
    results_inverted = detector.process_bits(inverted_test_bits)
    print(f"Extracted bits (inverted): {results_inverted}")


if __name__ == "__main__":
    test_preamble_detector()