"""
Embedded Python Blocks:

Preamble detector and payload extractor for backscatter signals.
Searches for 64-bit preamble and extracts following 16-bit payload.
Outputs to both file and QT GUI number sink.
"""

import numpy as np
from gnuradio import gr

STATE_SEARCH = 0
STATE_PAYLOAD = 1

class blk(gr.basic_block):

    def __init__(self, preamble_hex="AAAAAAAAAAAAAAAA"):
        """
        Preamble detector and payload extractor

        Args:
            preamble_hex: 64-bit preamble as hex string (16 hex characters)
        """
        gr.basic_block.__init__(
            self,
            name='Preamble Detector and Payload Extractor',
            in_sig=[np.byte],
            out_sig=[np.byte, np.float32]  # bits output, payload value output
        )

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

    def general_work(self, input_items, output_items):
        """Main processing function"""
        input_bits = input_items[0]
        output_bits = output_items[0] if len(output_items) > 0 else None
        output_payload = output_items[1] if len(output_items) > 1 else None

        added = 0
        consumed = 0
        to_consume = len(input_bits)

        for i in range(to_consume):
            bit = int(input_bits[i])
            consumed += 1

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

                if output_bits is not None and added < len(output_bits):
                    output_bits[added] = corrected_bit
                    added += 1

                # Check if we have collected 16 bits
                if len(self.payload_bits) == 16:
                    payload_value = self.process_payload()

                    # Send payload value to second output
                    if output_payload is not None and added < len(output_payload):
                        output_payload[added-1] = float(payload_value)

                    self.start_search()

        self.consume_each(consumed)
        return added

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
        """Cleanup when block is destroyed"""
        if hasattr(self, 'output_file') and self.output_file:
            self.output_file.close()


# Helper function for standalone testing
def test_preamble_detector():
    """Test function for the preamble detector"""
    # Example usage
    detector = blk(preamble_hex="DEADBEEFCAFEBABE")  # 64-bit hex preamble

    # Test data: preamble + payload
    test_bits = detector.preamble_bits + [1,0,1,1,0,0,1,0,1,0,1,0,1,1,0,0]  # 16-bit payload

    print(f"Test input: {test_bits}")

    # Simulate processing
    input_items = [np.array(test_bits, dtype=np.byte)]
    output_items = [np.zeros(len(test_bits), dtype=np.byte),
                   np.zeros(len(test_bits), dtype=np.float32)]

    detector.general_work(input_items, output_items)

if __name__ == "__main__":
    test_preamble_detector()
