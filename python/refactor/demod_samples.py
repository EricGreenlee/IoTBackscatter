import matplotlib.pyplot as plt
import numpy as np
from main import logger
from scipy import signal


def demodulate_packet(input_samples, tag_params, radio_settings, demod_settings):
    logger.debug("Demodulating packet")
    return 1,1