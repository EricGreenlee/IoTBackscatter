#!/usr/bin/env python3
"""
USRP N210 Receiver Demo Script

This script demonstrates how to connect to and receive samples from a USRP N210
connected via Ethernet at IP address 192.168.10.2.

Usage:
    python3 usrp_n210_rx.py --freq 915e6 --rate 1e6 --gain 50 --num_samps 10000
"""

import uhd
import numpy as np
import argparse
import time
import matplotlib.pyplot as plt
import os
from datetime import datetime


def setup_usrp(device_addr, sample_rate, center_freq, rx_gain):
    """
    Initialize and configure the USRP N210
    
    Args:
        device_addr (str): USRP device address (e.g., "addr=192.168.10.2")
        sample_rate (float): Sample rate in Hz
        center_freq (float): Center frequency in Hz
        rx_gain (float): RX gain in dB
        
    Returns:
        uhd.usrp.MultiUSRP: Configured USRP object
    """
    print(f"Connecting to USRP at {device_addr}")
    
    # Create USRP object
    usrp = uhd.usrp.MultiUSRP(device_addr)
    
    # Set sample rate
    usrp.set_rx_rate(sample_rate, 0)
    actual_rate = usrp.get_rx_rate(0)
    print(f"Requested rate: {sample_rate/1e6:.2f} MS/s")
    print(f"Actual rate: {actual_rate/1e6:.2f} MS/s")
    
    # Set center frequency
    tune_request = uhd.libpyuhd.types.tune_request(center_freq)
    tune_result = usrp.set_rx_freq(tune_request, 0)
    actual_freq = usrp.get_rx_freq(0)
    print(f"Requested freq: {center_freq/1e6:.2f} MHz")
    print(f"Actual freq: {actual_freq/1e6:.2f} MHz")
    
    # Set RX gain
    usrp.set_rx_gain(rx_gain, 0)
    actual_gain = usrp.get_rx_gain(0)
    print(f"Requested gain: {rx_gain} dB")
    print(f"Actual gain: {actual_gain:.2f} dB")
    
    # Get device info
    print(f"Using device: {usrp.get_pp_string()}")
    
    return usrp


def receive_samples(usrp, num_samps, timeout=3.0):
    """
    Receive samples from the USRP
    
    Args:
        usrp: USRP object
        num_samps (int): Number of samples to receive
        timeout (float): Receive timeout in seconds
        
    Returns:
        numpy.ndarray: Complex samples
    """
    print(f"Receiving {num_samps} samples...")
    
    # Create receive streamer
    st_args = uhd.usrp.StreamArgs("fc32", "sc16")  # Complex float32, over-the-wire sc16
    rx_streamer = usrp.get_rx_stream(st_args)
    
    # Allocate buffer
    recv_buffer = np.zeros(num_samps, dtype=np.complex64)
    
    # Set up streaming
    stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.num_done)
    stream_cmd.num_samps = num_samps
    stream_cmd.stream_now = True
    rx_streamer.issue_stream_cmd(stream_cmd)
    
    # Receive samples
    metadata = uhd.types.RXMetadata()
    num_rx_samps = rx_streamer.recv(recv_buffer, metadata, timeout)
    
    if num_rx_samps != num_samps:
        print(f"Warning: Requested {num_samps} samples, received {num_rx_samps}")
        
    if metadata.error_code != uhd.types.RXMetadataErrorCode.none:
        print(f"RX metadata error: {metadata.strerror()}")
    
    return recv_buffer[:num_rx_samps]


def analyze_samples(samples, sample_rate):
    """
    Perform basic analysis on received samples
    
    Args:
        samples (numpy.ndarray): Complex samples
        sample_rate (float): Sample rate in Hz
    """
    print(f"\nSample Analysis:")
    print(f"Number of samples: {len(samples)}")
    print(f"Sample rate: {sample_rate/1e6:.2f} MS/s")
    print(f"Duration: {len(samples)/sample_rate*1000:.2f} ms")
    
    # Power statistics
    power = np.abs(samples) ** 2
    avg_power = np.mean(power)
    peak_power = np.max(power)
    
    print(f"Average power: {10*np.log10(avg_power):.2f} dB")
    print(f"Peak power: {10*np.log10(peak_power):.2f} dB")
    print(f"Dynamic range: {10*np.log10(peak_power/avg_power):.2f} dB")
    
    # Frequency domain analysis
    fft = np.fft.fftshift(np.fft.fft(samples))
    freqs = np.fft.fftshift(np.fft.fftfreq(len(samples), 1/sample_rate))
    
    return power, fft, freqs


def plot_results(samples, power, fft, freqs, save_plot=False):
    """
    Plot time domain and frequency domain results
    
    Args:
        samples: Complex samples
        power: Power samples  
        fft: FFT of samples
        freqs: Frequency bins
        save_plot: Whether to save plot to file
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    
    # Time domain - I/Q
    time_axis = np.arange(len(samples)) / len(samples) * 1000  # ms
    ax1.plot(time_axis[:1000], np.real(samples[:1000]), 'b-', label='I', alpha=0.7)
    ax1.plot(time_axis[:1000], np.imag(samples[:1000]), 'r-', label='Q', alpha=0.7)
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Amplitude')
    ax1.set_title('I/Q Time Series (first 1000 samples)')
    ax1.legend()
    ax1.grid(True)
    
    # Time domain - Power
    ax2.plot(time_axis[:1000], 10*np.log10(power[:1000]), 'g-')
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('Power (dB)')
    ax2.set_title('Power vs Time (first 1000 samples)')
    ax2.grid(True)
    
    # Frequency domain
    ax3.plot(freqs/1e6, 10*np.log10(np.abs(fft)**2), 'b-')
    ax3.set_xlabel('Frequency (MHz)')
    ax3.set_ylabel('Power Spectral Density (dB)')
    ax3.set_title('Power Spectral Density')
    ax3.grid(True)
    
    # Constellation plot
    ax4.scatter(np.real(samples[::100]), np.imag(samples[::100]), alpha=0.5, s=1)
    ax4.set_xlabel('I')
    ax4.set_ylabel('Q')
    ax4.set_title('Constellation Plot (decimated)')
    ax4.grid(True)
    ax4.axis('equal')
    
    plt.tight_layout()
    
    if save_plot:
        plt.savefig('usrp_n210_samples.png', dpi=150, bbox_inches='tight')
        print("Plot saved as 'usrp_n210_samples.png'")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='USRP N210 Sample Receiver Demo')
    parser.add_argument('--addr', default='192.168.10.2', 
                       help='USRP IP address (default: 192.168.10.2)')
    parser.add_argument('--freq', type=float, default=915e6,
                       help='Center frequency in Hz (default: 915 MHz)')
    parser.add_argument('--rate', type=float, default=1e6,
                       help='Sample rate in Hz (default: 1 MS/s)')
    parser.add_argument('--gain', type=float, default=50,
                       help='RX gain in dB (default: 50)')
    parser.add_argument('--num_samps', type=int, default=10000,
                       help='Number of samples to receive (default: 10000)')
    parser.add_argument('--plot', action='store_true',
                       help='Plot received samples')
    parser.add_argument('--save_plot', action='store_true',
                       help='Save plot to file')
    parser.add_argument('--no_save', action='store_true',
                       help='Do not save samples (saves by default)')
    
    args = parser.parse_args()
    
    try:
        # Setup USRP
        device_addr = f"addr={args.addr}"
        usrp = setup_usrp(device_addr, args.rate, args.freq, args.gain)
        
        # Allow hardware to settle
        print("Letting hardware settle...")
        time.sleep(0.5)
        
        # Receive samples
        start_time = time.time()
        samples = receive_samples(usrp, args.num_samps)
        rx_time = time.time() - start_time
        
        print(f"Received {len(samples)} samples in {rx_time:.3f} seconds")
        print(f"Effective data rate: {len(samples)*8/rx_time/1e6:.2f} MB/s")
        
        # Analyze samples
        power, fft, freqs = analyze_samples(samples, args.rate)
        
        # Save samples with timestamp and radio settings (unless disabled)
        if not args.no_save:
            # Create samples directory if it doesn't exist
            samples_dir = os.path.join(os.path.dirname(__file__), 'local_samples')
            os.makedirs(samples_dir, exist_ok=True)
            
            # Generate filename with timestamp and radio settings
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            freq_mhz = int(args.freq / 1e6)
            rate_msps = args.rate / 1e6
            filename = f"usrp_n210_{timestamp}_{freq_mhz}MHz_{rate_msps:.1f}Msps_{args.gain}dB_{args.num_samps}samps.npy"
            filepath = os.path.join(samples_dir, filename)
            
            np.save(filepath, samples)
            print(f"Samples saved to {filepath}")
        
        # Plot if requested
        if args.plot or args.save_plot:
            plot_results(samples, power, fft, freqs, args.save_plot)
            
    except Exception as e:
        print(f"Error: {e}")
        return 1
        
    return 0


if __name__ == "__main__":
    exit(main())