# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an IoT backscatter communication system with three main components:
- **Hardware**: PCB designs for backscatter tags and receivers (empty directory)
- **Firmware**: Embedded software for the hardware (empty directory)  
- **Python**: Signal processing software for packet demodulation and simulation

The Python modem software is the primary active component, implementing CDMA-based backscatter signal processing with multiple tag support.

## Python Environment Setup

Navigate to the `python/` directory and activate the virtual environment:
```bash
cd python
source .venv/bin/activate
```

## Core Architecture

### Signal Processing Pipeline (`demod.py`)
The demodulation pipeline processes received samples through:
1. Low-pass filtering (`lpf()`)
2. Automatic gain control (`agc()`)
3. Placeholder stages for: despread, time recovery, frequency correction, Costas loop, BPSK demod

### Data Structures (`IoTBSConst.py`)
- `Packet`: Base class for ideal packets with preamble/sync/payload bits
- `SimulatedPacket`: Extends Packet with RF impairments (SNR, frequency offset, time delay)
- `RadioSettings`: Sample rate and carrier frequency configuration
- Pre-defined Gold codes for 10 CDMA tags (`GCs` array)
- Standard preamble (16 ones) and sync sequence (16-bit pattern)

### Main Application (`main.py`)
Command-line interface supporting three sample sources:
- `sim`: Generate simulated packets with configurable tag count and RF impairments
- `file`: Load samples from file (not implemented)
- `sdr`: Stream from SDR hardware (not implemented)

Includes comprehensive logging to both console (configurable verbosity) and `demodulator.log` file.

## Running the Code

### Quick Start - Full Experiment
Use the automated script to run a complete backscatter experiment:
```bash
./scripts/start_modem.sh
```
This script opens two terminals:
- HackRF transmitter: `hackrf_transfer -f 915000000 -x 47 -c 127`
- Python receiver: `python/src/main.py`

### Individual Components

#### Python Demodulator (`python/src/main.py`)
The main demodulator accepts three sample sources:

```bash
cd python
# Simulate packets from multiple tags
python src/main.py --source sim --n_tags 2 -vv

# Process samples from file (not implemented)
python src/main.py --source file --file samples.dat

# Stream from SDR hardware (not implemented) 
python src/main.py --source sdr
```

Verbosity levels: `-v` (WARNING), `-vv` (INFO), `-vvv` (DEBUG)

#### HackRF Transmitter
Start the HackRF One transmitter for 915 MHz experiments:
```bash
hackrf_transfer -f 915000000 -x 47 -c 127
```

#### Arduino Firmware
Flash and run the embedded firmware for backscatter tags:
```bash
# Navigate to arduino directory and flash firmware
cd arduino/
# (Upload via Arduino IDE or command line tools)
```

#### USRP N210 Receiver Demo
Capture samples from a USRP N210 SDR for analysis and testing:
```bash
cd python/playground/
python USRP_N210_RX_demo.py --freq 915e6 --rate 1e6 --gain 50 --num_samps 10000 --plot
```

The script automatically saves samples with timestamp and radio settings to the `samples/` folder.

#### Meta Design Script
Run the meta design optimization script:
```bash
cd python/
python meta_design.py
```

## Code Style

The project uses Black formatting with 88-character line length and isort for import organization. Configuration is in `pyproject.toml`.

## Development Status

The signal processing pipeline has placeholder implementations for most DSP blocks. The file and SDR sample sources are not implemented. Hardware and firmware directories are currently empty.