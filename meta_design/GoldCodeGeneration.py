#Generate CDMA gold codes into a text file

import numpy as np
import matplotlib.pyplot as plt

# parameters

ndevices = 10


goldcode_n = 7
goldcode_length = 2**goldcode_n - 1

#Gold code Parameters
register1 = [1] * goldcode_n  # Initial state of LFSR 1
register2 = [1] * goldcode_n  # Initial state of LFSR 2
taps1 = [goldcode_n, goldcode_n-1]         # Feedback taps for LFSR 1
taps2 = [goldcode_n, goldcode_n-2]         # Feedback taps for LFSR 2
# taps1 = [3, 2]         # Feedback taps for LFSR 1
# taps2 = [3, 1]         # Feedback taps for LFSR 2
# taps1 = [6, 5]         # Feedback taps for LFSR 1
# taps2 = [6, 3]         # Feedback taps for LFSR 2
# taps1 = [6, 5]         # Feedback taps for LFSR 1
# taps2 = [4, 2]         # Feedback taps for LFSR 2
# taps1 = [8, 5]         # Feedback taps for LFSR 1
# taps2 = [7, 2]         # Feedback taps for LFSR 2
# taps1 = [7, 6]         # Feedback taps for LFSR 1
# taps2 = [7, 5]         # Feedback taps for LFSR 2
# sequence_length = 2**len(register1) - 1  # Length o

def generate_m_sequence(register, taps, length):
    """
    Generate an m-sequence using an LFSR.
    Args:
        register: Initial state of the shift register (list of 0s and 1s).
        taps: List of tap positions for feedback (1-based index).
        length: Length of the m-sequence.
    Returns:
        m_sequence: Generated m-sequence (list).
    """
    n = len(register)
    m_sequence = []
    for _ in range(length):
        m_sequence.append(register[-1])
        feedback = sum(register[tap - 1] for tap in taps) % 2
        register = [feedback] + register[:-1]
    return m_sequence

def generate_gold_code(mseq1, mseq2, phase_shift):
    """
    Generate a Gold code from two m-sequences.
    Args:
        mseq1: First m-sequence.
        mseq2: Second m-sequence.
        phase_shift: Number of positions to shift mseq2.
    Returns:
        gold_code: Generated Gold code (list).
    """
    mseq2_shifted = np.roll(mseq2, phase_shift)
    gold_code = [(bit1 ^ bit2) for bit1, bit2 in zip(mseq1, mseq2_shifted)]
    return gold_code


# Generate m-sequences
mseq1 = generate_m_sequence(register1, taps1, goldcode_length)
mseq2 = generate_m_sequence(register2, taps2, goldcode_length)

# Generate ndevices Gold codes with different phase shifts
zero_gold_codes = []
for phase_shift in range(ndevices):  # Generate a gold code for each device
    gold_code = generate_gold_code(mseq1, mseq2, phase_shift)
    zero_gold_codes.append(gold_code)

# zero_gold_codes = [code for code in zero_gold_codes]

gold_codes = np.empty((ndevices, goldcode_length))
for i in range(ndevices):
    gold_codes[i] = [2*bit-1 for bit in zero_gold_codes[i]]

# Print the Gold codes
for i in range(ndevices):
    print(f"Gold Code {i}: {gold_codes[i]}")
    
#save the goldcodes to a file for import into Arduino code
filename = 'goldcodes_Arduino.txt'
with open(filename, 'w') as f:
    f.write("{\n")
    for i in range(ndevices):
        f.write("{")
        for j in range(goldcode_length):
            f.write(f"{int(gold_codes[i][j])}")
            if (j != goldcode_length-1):
                f.write(", ")
        f.write("},\n")
    f.write("};")

# with open(filename, 'w') as f:
    # f.write("{\n")
    # for i in range(ndevices):
    #     # f.write(f"goldcode[{i}] = {{")
    #     f.write("{")
    #     for j in range(goldcode_length):
    #         f.write(f"{int(gold_codes[i][j])}, ")
    #     f.write("},\n")
    # f.write("};")
        
#save the goldcodes to a file for import into Python code 
filename = 'goldcodes_Python.txt'
with open(filename, 'w') as f:
    f.write("np.array([\n")
    for i in range(ndevices):
        f.write("[")
        for j in range(goldcode_length):
            f.write(f"{int(gold_codes[i][j])}")
            if (j != goldcode_length-1):
                f.write(", ")
        f.write("],\n")
    f.write("])")
    
# with open(filename, 'w') as f:
#     f.write("gold_codes = np.array([\n")
#     for i in range(ndevices):
#         # f.write(f"goldcode_{i} = [")
#         f.write("[")
#         for j in range(goldcode_length-1):
#             f.write(f"{int(gold_codes[i][j])}, ")
#         f.write(f"{int(gold_codes[i][j])} ")
#         if i == ndevices-1:
#             f.write("]\n")
#         else:
#             f.write("],\n")
#     f.write("])")