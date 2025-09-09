import argparse
import logging
import sys
import matplotlib.pyplot as plt
import numpy as np
import json

logger = logging.getLogger("demodulator")  # single global logger

import demod_samples
import get_samples
import IoTBSConst

#configure logging
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

# Command line interace
def parse_args():
    parser = argparse.ArgumentParser(description="RF Demodulator skeleton")
    parser.add_argument("-s", "--source", choices=["sim", "file", "sdr"], required=True,
                        help="Sample source: sim, file, sdr")
    parser.add_argument("-f", "--file", type=str, help="Path to input file (required if --source=file)")
    parser.add_argument("-v", "--verbose", action="count", default=1,
                        help="[blank]=WARNING, -v=INFO, -vv=DEBUG")
    parser.add_argument("-t","--n_tags", type=int, default=1, help="Number of backscatter tags simultaneously transmitting")
    parser.add_argument("-i","--iterations", type=int, default=1, help="Number of simulation iterations to run")
    parser.add_argument("-p","--plot", action="store_true", help="Enable plotting of figures")
    return parser.parse_args()

def load_json_settings(fpath):
    try:
        with open(fpath, 'r') as f:
            config = json.load(f)
        logger.info("Loaded settings from IoTBSSettings.json")
        
        # Extract settings
        radio_settings = config.get('radio_settings', {})
        sim_settings = config.get('sim_settings',{})
        demod_settings = config.get('demodulator_settings', {})
        
    except Exception as e:
        logger.error(f"Error loading settings: {e}")
        raise 
    
    logger.debug(f"radio_settings: {radio_settings}")
    logger.debug(f"sim_settings: {sim_settings}")
    logger.debug(f"demod_settings: {demod_settings}")
    
    return radio_settings, demod_settings, sim_settings

def gen_tag_params(ntags):
    all_tag_params = IoTBSConst.AllTagParams()

    for tag in range(ntags):
        single_tag_param = IoTBSConst.TagParam( 
            id = tag,
            payload_bits = np.array([0,1,0,0,1,0,1,0,0,1,1,0,1,1,1,1])
        ) 
        
        all_tag_params.add_tag(single_tag_param)
        
    return(all_tag_params)

def print_bit_error_summary(tot_bits_error, tot_bits_sent, ntags):
    logger.info("=== Bit Error Summary ===")
    for tag_id in range(ntags):
        if tot_bits_sent[tag_id] > 0:
            ber = tot_bits_error[tag_id] / tot_bits_sent[tag_id]
            logger.info(f"Tag {tag_id}: {int(tot_bits_error[tag_id])}/{int(tot_bits_sent[tag_id])} errors, BER: {ber:.4f}")
        else:
            logger.info(f"Tag {tag_id}: No bits sent")

def generate_samples(source, fname, ntags, radio_settings, sim_settings):
    
    tag_params = gen_tag_params(ntags)
    
    logger.debug(f"Tag summary: {tag_params.summary()}")
    
    if source == "sim":
        logger.debug("Simulating samples")
        samples = get_samples.simulate_samples(ntags, tag_params, sim_settings, radio_settings)
        
    elif source == "file":
        if not fname:
            logger.error("--file must be specified when --source=file")
            sys.exit(1)
        logger.debug(f"Pulling samples from file {fname}")
        samples = get_samples.file_samples(fname)

    elif source == "sdr":
        logger.debug("Pulling samples from SDR")
        samples = get_samples.sdr_samples(radio_settings)
    else:
        logger.error(f"Unknown source: {source}")
        sys.exit(1)
        
    return samples, tag_params
        
def plot():
    # Set up keyboard event handler to close all plots
    def on_key(event):
        if event.key == 'q':
            plt.close('all')
            logger.info("All plot windows closed")
    
    # Connect the key event handler to all figures
    for fig_num in plt.get_fignums():
        fig = plt.figure(fig_num)
        fig.canvas.mpl_connect('key_press_event', on_key)
    
    logger.info("Press 'q' in any plot window to close all plots")
    plt.show()

def main():
    args = parse_args()
    configure_logging(args.verbose)

    logger.debug(f"**** New run with CLI arguments: {args}")
    
    # Load settings from JSON configuration file
    radio_settings, demod_settings, sim_settings = load_json_settings("refactor/IoTBSSettings.json")
    
    #initialize data structure to keep track of results
    tot_bits_error = np.zeros(args.n_tags)
    tot_bits_sent = np.zeros(args.n_tags)
   
    # Run multiple iterations
    for iteration in range(args.iterations): 
        logger.info("=== Starting iteration %d/%d ===", iteration + 1, args.iterations)
        
        samples, tag_params  = generate_samples(args.source, args.file, args.n_tags, radio_settings, sim_settings)
        
        packet_bits, bit_errors, bits_sent = demod_samples.demodulate_packet(samples, tag_params, radio_settings, demod_settings, enable_plotting=args.plot)
        # demod_results = demod.demodulate_packet(samples, tag_params, radio_params, plotting_level, demod_settings)
        logger.debug(f"Demodulated bits:\n {packet_bits}")
        
        tot_bits_error += bit_errors
        tot_bits_sent += bits_sent
        
    
    #display demod_results
    print_bit_error_summary(tot_bits_error, tot_bits_sent, args.n_tags)
    
    if args.plot: 
        plot()


if __name__ == "__main__":
    main()