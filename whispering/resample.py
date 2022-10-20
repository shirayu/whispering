#!/usr/bin/env python3
import numpy as np
from scipy.interpolate import interp1d
from whisper.audio import SAMPLE_RATE


def resample(x: np.ndarray, source_sr: int) -> np.ndarray:
    """Resample a numpy array to match the SAMPLE_RATE
    of Whisper (16 000 Hz).

    Args:
        x (np.ndarray): Source numpy array
        source_sr (int): Source sample rate

    Returns:
        np.ndarray: The resampled array
    """

    if source_sr == SAMPLE_RATE:
        return x
    factor = source_sr / SAMPLE_RATE
    n = int(np.ceil(x.size / factor))
    f = interp1d(np.linspace(0, 1, x.size), x, "linear")
    return f(np.linspace(0, 1, n))
