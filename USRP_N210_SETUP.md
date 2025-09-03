# USRP N210 Setup Guide

## Network Configuration

The USRP N210 connects via Ethernet and can interfere with internet connectivity. To avoid this:

1. **Connect USRP N210**: Plug the USRP into the USB-to-Ethernet adapter
2. **Configure network**: Run the provided script to set up networking without breaking internet:
   ```bash
   ./scripts/setup_usrp_network.sh
   ```

This script:
- Finds the USB-to-Ethernet interface  
- Sets a static IP (192.168.10.1) for your computer
- Adds a route specifically for the USRP subnet (192.168.10.0/24)
- Preserves your WiFi internet connection
- Tests connectivity to the USRP at 192.168.10.2

## Python Demo Usage

The demo script is located at `python/playground/USRP_N210_RX_demo.py` and provides a complete example of:
- Connecting to the USRP N210
- Configuring frequency, sample rate, and gain
- Receiving samples
- Basic signal analysis
- Plotting results

### Basic Usage

```bash
cd python
source .venv/bin/activate

# Simple test - receive 10000 samples at 915 MHz (auto-saves with timestamp)
python3 playground/USRP_N210_RX_demo.py

# Custom parameters
python3 playground/USRP_N210_RX_demo.py --freq 915e6 --rate 2e6 --gain 40 --num_samps 10000

# With plotting
python3 playground/USRP_N210_RX_demo.py --freq 915e6 --plot

# Disable automatic sample saving
python3 playground/USRP_N210_RX_demo.py --freq 915e6 --no_save --save_plot
```

### Command Line Options

- `--addr`: USRP IP address (default: 192.168.10.2)
- `--freq`: Center frequency in Hz (default: 915 MHz)
- `--rate`: Sample rate in Hz (default: 1 MS/s)  
- `--gain`: RX gain in dB (default: 50)
- `--num_samps`: Number of samples to receive (default: 10000)
- `--plot`: Show interactive plots
- `--save_plot`: Save plot to PNG file
- `--no_save`: Disable automatic sample saving (samples are saved by default)

### Sample Storage

By default, the script automatically saves all captured samples to `python/playground/samples/` with filenames that include:
- Timestamp (YYYYMMDD_HHMMSS)
- Center frequency (MHz)
- Sample rate (Msps) 
- Gain (dB)
- Number of samples

Example filename: `usrp_n210_20250903_143022_915MHz_1.0Msps_50dB_10000samps.npy`

### Analysis Features

The script provides:
- Time domain I/Q plots
- Power vs time analysis
- Power spectral density (frequency domain)
- Constellation plot
- Basic statistics (average/peak power, dynamic range)

## Integration with Main Codebase

To integrate USRP N210 support into your main demodulator:

1. **Add USRP source option** to `python/src/main.py`
2. **Import UHD** in the main application
3. **Add USRP configuration** parameters
4. **Create USRP streamer** similar to the demo script
5. **Feed samples** to the existing demodulation pipeline in `demod.py`

The USRP will provide complex float32 samples that can be processed by your existing `lpf()`, `agc()`, and other DSP functions.

## Performance Notes

- The USRP N210 with UBX daughterboard supports:
  - Frequency range: 10 MHz - 6 GHz
  - Sample rates: up to ~25 MS/s (depending on host interface)
  - 14-bit ADC resolution
- For high sample rates, you may need to optimize network buffers as suggested in the warnings
- The Ethernet interface limits sustained throughput compared to USB 3.0 models

## Troubleshooting

### No device found
- Check network cable connections
- Verify USRP power LED is on
- Run: `uhd_find_devices` to detect the device
- Check that the network is configured correctly

### Network connectivity issues
- Use the provided `setup_usrp_network.sh` script
- Ensure your WiFi interface remains the default route
- Check routes with: `ip route show`

### Performance warnings
- The UDP buffer warnings are normal for Ethernet USRPs
- For better performance, consider increasing kernel network buffers
- Thread priority warnings can be ignored for basic operation