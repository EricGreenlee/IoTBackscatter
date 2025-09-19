import matplotlib.pyplot as plt
import numpy as np
import os

from main import logger
import IoTBSConst

def gen_nonidealities(sim_settings, ideal_sps):
    # ideal_sps = radio_settings.get('sps', {})
    
    enable_nonideal = sim_settings.get('enable_nonideal', {})
    nonideal_vals = sim_settings.get('nonideal_vals', {})
    
    if enable_nonideal.get('sps', {}):
        sps_ratio = nonideal_vals.get('sps_ratio', {})
        sps = ideal_sps*np.random.uniform((1-sps_ratio), (1+sps_ratio))
    else:
        sps = ideal_sps
        
    if enable_nonideal.get('noise', {}):
        tx_pwr_dbm = nonideal_vals.get('tx_pwr_dbm', {})
        noise_pwr_dbm = nonideal_vals.get('noise_pwr_dbm', {})
    else:
        tx_pwr_dbm = 100
        noise_pwr_dbm = -100
        
    if enable_nonideal.get('attenuation', {}):
        tag2modem_dist_m = nonideal_vals.get('tag2modem_dist_m', {})
    else:
        tag2modem_dist_m = 0
        
    if enable_nonideal.get('frequency', {}):
        max_freq_offset_hz = nonideal_vals.get('max_freq_offset_hz', {})
        freq_offset_hz = np.random.uniform(-1*max_freq_offset_hz, max_freq_offset_hz)
    else:
        freq_offset_hz = 0
        
    if enable_nonideal.get('time', {}):
        time_delay_sec = np.random.uniform(0,nonideal_vals.get('max_time_delay_sec', {}))
    else:
        time_delay_sec = 0
        
    return sps, tx_pwr_dbm, noise_pwr_dbm, tag2modem_dist_m, freq_offset_hz, time_delay_sec

def simulate_samples(ntags, tag_params, sim_settings, radio_settings):
    seed = np.random.randint(0,100000)
    np.random.seed(seed)
    logger.debug(f"Generating simulated samples with seed {seed}")
    
    cur_tag = tag_params.get_tag(0)
    
    packet_num_samples = len(cur_tag.goldcode)*len(cur_tag.all_bits)*sim_settings.get('sps',{})
    tot_num_samples=  packet_num_samples*sim_settings.get('duration_ratio',{})
    
    packet_duration_sec = packet_num_samples/radio_settings.get('samplerate_hz',{})
    tot_duration_sec = packet_duration_sec*sim_settings.get('duration_ratio',{})
    
    sim_samples = np.zeros(tot_num_samples).astype(np.complex64)
    logger.debug(f"packet_num_samples: {packet_num_samples}, tot_num_samples: {packet_num_samples}, packet_duration_sec: {packet_duration_sec}, total_sim_time_sec: {tot_duration_sec}")
    
    all_sim_packets = IoTBSConst.AllSimPackets()
    
    #make a structure for simulated non-idealities for each tag 
    for id in range(ntags):
        cur_tag = tag_params.get_tag(id)
        
        target_sps = radio_settings.get('sps', {})
        carrier_freq_hz = radio_settings.get('carrier_freq_hz', {})
        samplerate_hz = radio_settings.get('samplerate_hz', {})
        actual_sps, tx_pwr_dbm, noise_pwr_dbm, tag2modem_dist_m, freq_offset_hz, time_offset_sec = gen_nonidealities(sim_settings, target_sps)
        
        cur_sim_packet = IoTBSConst.SimPacket(
            tagParam = cur_tag,
            ideal_sps = target_sps,
            carrier_freq_hz = carrier_freq_hz,
            samplerate_hz = samplerate_hz,
            actual_sps = actual_sps,
            tx_pwr_dbm = tx_pwr_dbm,
            noise_pwr_dbm = noise_pwr_dbm,
            tag2modem_dist_m = tag2modem_dist_m,
            freq_offset_hz = freq_offset_hz,
            time_offset_sec = time_offset_sec
        )
        
        cur_sim_packet.gen_nonideal_samples(tot_num_samples)
        
        sim_samples = sim_samples + cur_sim_packet.samples
        all_sim_packets.add_packet_meta(cur_sim_packet)
        
    logger.debug(f"Simulated packet metadata: {all_sim_packets.summary()}")
        
    return sim_samples

def file_samples(path):
    logger.info(f"Reading samples from file: {path}")


    # If path is just a filename, look in src/cloud_samples directory
    if not os.path.isabs(path) and not os.path.dirname(path):
        samples_dir = os.path.join(os.path.dirname(__file__), 'cloud_samples')
        full_path = os.path.join(samples_dir, path)
    else:
        full_path = path
    
    try:
        samples = np.load(full_path)
        logger.info(f"Loaded {len(samples)} samples from {full_path}")
        return samples
    except FileNotFoundError:
        logger.error(f"File not found: {full_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading samples from {full_path}: {e}")
        raise


def sdr_samples(radio_setting):
    # logger.debug("Simulating samples in file")
    return 0