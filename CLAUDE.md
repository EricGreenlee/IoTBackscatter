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

The main demodulator accepts three sample sources:

```bash
# Simulate packets from multiple tags
python src/main.py --source sim --n_tags 2 -vv

# Process samples from file (not implemented)
python src/main.py --source file --file samples.dat

# Stream from SDR hardware (not implemented) 
python src/main.py --source sdr
```

Verbosity levels: `-v` (WARNING), `-vv` (INFO), `-vvv` (DEBUG)

## Code Style

The project uses Black formatting with 88-character line length and isort for import organization. Configuration is in `pyproject.toml`.

## Development Status

The signal processing pipeline has placeholder implementations for most DSP blocks. The file and SDR sample sources are not implemented. Hardware and firmware directories are currently empty.