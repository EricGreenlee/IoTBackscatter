#!/usr/bin/env python3
import argparse
import logging
import sys

import IoTBSConst
import sim_cdma

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
def get_simulated_samples():
    logger.debug("Generating simulated samples")
    
    # p = Packet(goldcode=[1,0,1,1,0,1,0,1], preamble = [1, 1, 1, 1], sync_word = [1, 0, 1, 0], bits= [0, 0, 0, 0], samples_per_signal=4)
    # print(p)

    # Non-ideal simulated packet
    sp = IoTBSConst.SimulatedPacket(
        goldcode_number = 0,
        goldcode=IoTBSConst.GCs[0],
        preamble_bits = [1, 1, 1, 1],
        sync_bits = [1, 0, 1, 0],
        payload_bits=[0, 0, 0, 0],
        samples_per_signal=4,
        snr_db=15.0,
        frequency_offset_hz=500.0,
        time_delay_sec=0.002
    )
    # print(sp.summary())
    logger.info("simulated packet metadata: %s", sp)
    
    sim_packet = sim_cdma.gen_ideal_packet()
    return sim_packet

def get_file_samples(path):
    logger.info(f"Reading samples from file: {path}")
    return []  # TODO: implement

def get_sdr_samples():
    logger.info("Streaming live samples from SDR hardware")
    return []  # TODO: implement

def run_demodulator(samples):
    logger.info(f"Processing {len(samples)} samples")


# ----------------------------
# CLI
# ----------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="RF Demodulator skeleton")
    parser.add_argument("-s", "--source", choices=["sim", "file", "sdr"], required=True,
                        help="Sample source: sim, file, sdr")
    parser.add_argument("-f", "--file", type=str, help="Path to input file (required if --source=file)")
    parser.add_argument("-v", "--verbose", action="count", default=1,
                        help="-v=WARNING, -vv=INFO, -vvv=DEBUG")
    return parser.parse_args()

# ----------------------------
# Main entrypoint
# ----------------------------
def main():
    args = parse_args()
    configure_logging(args.verbose)

    logger.debug(f"**** New run with CLI arguments: {args}")

    # Select sample source
    if args.source == "sim":
        samples = get_simulated_samples()
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

    logger.info("Packet to demod: %s", samples)
    run_demodulator(samples)

if __name__ == "__main__":
    main()
if __name__ == "__main__":
    main()
    main()
