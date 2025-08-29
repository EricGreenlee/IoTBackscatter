#!/usr/bin/env python3
import argparse
import logging
import sys

import demod
import IoTBSConst
import numpy as np
import sim_cdma

np.set_printoptions(linewidth=120, threshold=np.inf)

# ----------------------------
# Global logger
# ----------------------------
logger = logging.getLogger("demodulator")  # single global logger

def configure_logging(verbosity_level: int):
    """
    Configure logging programmatically for both console and file.
    verbosity_level: 0=ERROR, 1=WARNING, 2=INFO, 3=DEBUG
    """
    logger.setLevel(logging.DEBUG)  # capture all levels; filter in handlers

    # Remove existing handlers to avoid duplicates
    if logger.hasHandlers():
        logger.handlers.clear()

    # --- Console handler ---
    ch = logging.StreamHandler(sys.stdout)
    if verbosity_level == 0:
        ch.setLevel(logging.ERROR)
    elif verbosity_level == 1:
        ch.setLevel(logging.WARNING)
    elif verbosity_level == 2:
        ch.setLevel(logging.INFO)
    else:
        ch.setLevel(logging.DEBUG)
    ch_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")
    ch.setFormatter(ch_formatter)
    logger.addHandler(ch)

    # --- File handler ---
    fh = logging.FileHandler("demodulator.log", mode='a')
    fh.setLevel(logging.DEBUG)  # log everything to file
    fh_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")
    fh.setFormatter(fh_formatter)
    logger.addHandler(fh)

# ----------------------------
# Sample functions (stubs)
# ----------------------------
def get_simulated_samples(n_tags: int, seed: int):
    logger.info("Generating simulated samples for %s tags with seed %s", n_tags, seed)
    
    # Set random seed for reproducible results
    np.random.seed(seed)

    # Hold all simulated packet parameters
    all_pp = IoTBSConst.TagParams()

    for tag in range(n_tags):
        spp = IoTBSConst.SimulatedPacketParams(
            tag_id = tag,
            preamble_bits = IoTBSConst.preamble,
            # preamble_bits = np.array([1, 1, 1, 1, 1, 0, 1, 0]),
            # sync_bits = IoTBSConst.sync_seq,
            sync_bits = [],
            # n_payload_bits = 100,
            # payload_bits=np.array([1, 0, 1, 0, 1, 0, 0, 1]),
            payload_bits=np.random.randint(0,2,IoTBSConst.bitsPerPacket),
            pad_bits=np.zeros(10),
            sps=2,
            snr_db=15.0,
            frequency_offset_hz=500.0,
            time_delay_sec=0.002
        )
        spp.gen_ideal_samples()
        
        all_pp.add_tag(spp)
    
    # print(sp.summary())
    logger.debug("simulated packet metadata: %s", all_pp.summary())
     
    sim_packet = all_pp.combined_samples()
    
    return sim_packet, all_pp

def get_file_samples(path):
    logger.info(f"Reading samples from file: {path}")
    return []  # TODO: implement

def get_sdr_samples():
    logger.info("Streaming live samples from SDR hardware")
    return []  # TODO: implement

# Command line interace
def parse_args():
    parser = argparse.ArgumentParser(description="RF Demodulator skeleton")
    parser.add_argument("-s", "--source", choices=["sim", "file", "sdr"], required=True,
                        help="Sample source: sim, file, sdr")
    parser.add_argument("-f", "--file", type=str, help="Path to input file (required if --source=file)")
    parser.add_argument("-v", "--verbose", action="count", default=1,
                        help="-v=WARNING, -vv=INFO, -vvv=DEBUG")
    parser.add_argument("-t","--n_tags", type=int, default=1, help="Number of backscatter tags simultaneously transmitting")
    parser.add_argument("-i","--iterations", type=int, default=1, help="Number of simulation iterations to run")
    return parser.parse_args()

def main():
    args = parse_args()
    configure_logging(args.verbose)

    logger.debug(f"**** New run with CLI arguments: {args}")
    
    # trial
    
    radio_params = IoTBSConst.RadioSettings(
        samplerate_hz = 252315,
        carrier_freq_hz = 915100000
    )

    # Initialize tracking variables
    all_results = []  # Store results from each iteration
    total_errors_per_tag = {}
    total_bits_per_tag = {}
    
    # Run multiple iterations
    for iteration in range(args.iterations):
        logger.info("=== Starting iteration %d/%d ===", iteration + 1, args.iterations)
        # seed = iteration + 1000  # Use different seed for each iteration
        seed = np.random.randint(0,100000)
        
        # Select sample source
        if args.source == "sim":
            samples, tag_params = get_simulated_samples(args.n_tags, seed)
        elif args.source == "file":
            if not args.file:
                logger.error("--file must be specified when --source=file")
                sys.exit(1)
            samples = get_file_samples(args.file)
        elif args.source == "sdr":
            samples = get_sdr_samples()
        else:
            logger.error(f"Unknown source: {args.source}")
            sys.exit(1)

        logger.debug("Packet to demod: %s", samples)
        iteration_results = demod.demodulate_packet(samples, tag_params, radio_params)
        all_results.append(iteration_results)
        
        # Accumulate totals for each tag
        for tag_id, tag_result in iteration_results.items():
            if tag_id not in total_errors_per_tag:
                total_errors_per_tag[tag_id] = 0
                total_bits_per_tag[tag_id] = 0
            total_errors_per_tag[tag_id] += tag_result['num_errors']
            total_bits_per_tag[tag_id] += tag_result['num_bits']
    
    # Calculate and display comprehensive statistics table
    logger.info("=== COMPREHENSIVE STATISTICS OVER %d ITERATIONS ===", args.iterations)
    
    # Collect error rate statistics per tag across all iterations
    error_rates_by_tag = {}
    for tag_id in range(args.n_tags):
        error_rates_by_tag[tag_id] = []
    
    for results in all_results:
        for tag_id, tag_result in results.items():
            error_rate = tag_result['num_errors'] / tag_result['num_bits']
            error_rates_by_tag[tag_id].append(error_rate)
    
    # Print comprehensive table header
    logger.info("Tag ID | Total Errors | Total Bits | Overall BER | Min BER    | Max BER    | Mean BER   | Median BER")
    logger.info("-------|--------------|------------|-------------|------------|------------|------------|------------")
    
    # Print statistics for each tag
    overall_total_errors = 0
    overall_total_bits = 0
    
    for tag_id in range(args.n_tags):
        if tag_id in total_errors_per_tag and tag_id in error_rates_by_tag:
            # Overall statistics across all iterations
            total_errors = total_errors_per_tag[tag_id]
            total_bits = total_bits_per_tag[tag_id]
            overall_ber = total_errors / total_bits
            
            # Per-iteration BER statistics
            error_rates = error_rates_by_tag[tag_id]
            min_ber = min(error_rates) if error_rates else 0
            max_ber = max(error_rates) if error_rates else 0
            mean_ber = sum(error_rates) / len(error_rates) if error_rates else 0
            median_ber = sorted(error_rates)[len(error_rates)//2] if error_rates else 0
            
            logger.info("  %3d  |        %5d |      %5d |    %8.6f |   %8.6f |   %8.6f |   %8.6f |   %8.6f", 
                       tag_id, total_errors, total_bits, overall_ber, min_ber, max_ber, mean_ber, median_ber)
            
            overall_total_errors += total_errors
            overall_total_bits += total_bits
    
    # Overall combined statistics
    if overall_total_bits > 0:
        combined_ber = overall_total_errors / overall_total_bits
        logger.info("-------|--------------|------------|-------------|------------|------------|------------|------------")
        logger.info("TOTAL  |        %5d |      %5d |    %8.6f |            |            |            |            ", 
                   overall_total_errors, overall_total_bits, combined_ber) 

if __name__ == "__main__":
    main()
