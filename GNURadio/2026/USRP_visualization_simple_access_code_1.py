"""
Embedded Python Blocks:

Each time this file is saved, GRC will instantiate the first class it finds
to get ports and parameters of your block. The arguments to __init__  will
be the parameters. All of them are required to have default values!
"""

import numpy as np
from gnuradio import gr

STATE_SEARCH = 0
STATE_HEADER = 1
STATE_DATA = 2

class blk(gr.basic_block):  # other base classes are basic_block, decim_block, interp_block

    def __init__(self, preamble="", data_len_bits=8, header_len_bits=0):  # only default arguments here
        """arguments to this function show up as parameters in GRC"""
        gr.basic_block.__init__(
            self,
            name='Preamble Detector',   # will show up in GRC
            in_sig=[np.byte],
            out_sig=[np.byte]
        )
        # if an attribute with the same name as a parameter is found,
        # a callback is registered (properties work, too).
        self.preamble = preamble
        self.data_len_bits = data_len_bits
        self.header_len_bits = header_len_bits

        self.preamble_bits = [np.byte(c) for c in preamble]
        self.inverted_preamble_bits = [np.byte(b^1) for b in self.preamble_bits]
        self.header_half = int(self.header_len_bits / 2)

        self.start_search()

    def general_work(self, input_items, output_items):
        added = 0
        to_consume = min(len(input_items[0]), len(output_items[0]))

        for i in range(to_consume):
            bit = input_items[0][i]
            if self.inverted:
                bit = np.byte(bit^1)

            if self.state == STATE_SEARCH:
                self.search_bits = self.search_bits[1:] + [bit]

                if self.search_bits == self.preamble_bits:
                    self.inverted = False
                    self.start_header()
                elif self.search_bits == self.inverted_preamble_bits:
                    self.inverted = True
                    self.start_header()

            elif self.state == STATE_HEADER:
                self.header_bits.append(bit)

                if len(self.header_bits) == self.header_len_bits:
                    len1, len2 = self.header_bits[:self.header_half], self.header_bits[self.header_half:]
                    if len1 == len2:
                        self.data_len_bits = binary_digits_to_int(len1) * 8
                        self.start_data()
                    else:
                        self.start_search()

            elif self.state == STATE_DATA:
                self.data_bits.append(bit)
                output_items[0][added] = bit
                added += 1

                if len(self.data_bits) == self.data_len_bits:
                    self.start_search()

        self.consume_each(to_consume)
        return added

    def start_search(self):
        self.state = STATE_SEARCH
        self.inverted = False
        self.search_bits = [None for _ in self.preamble_bits]

    def start_header(self):
        if self.header_len_bits > 0:
            self.state = STATE_HEADER
            self.header_bits = []
        else:
            self.start_data()

    def start_data(self):
        self.state = STATE_DATA
        self.data_bits = []

def binary_digits_to_int(digits):
    return sum([v * 2 ** i for i, v in enumerate(reversed(digits))])
