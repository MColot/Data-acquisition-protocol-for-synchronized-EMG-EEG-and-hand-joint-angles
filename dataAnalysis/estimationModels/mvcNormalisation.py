# https://www.sciencedirect.com/science/article/pii/S1350453399000545?via%3Dihub#BIB3
# this article shows that the best mvc normalization technique is to divide each channel by the maximum absolute
# value of the isometric contraction

import numpy as np


def normalize(pathToMvc, pathToSignal):
    """
    Normalize EMG signal based on a MVC record
    """
    mvc = np.load(pathToMvc)
    signal = np.load(pathToSignal)
    return np.array([signal[i] / max(abs(mvc[i])) for i in range(len(signal))])
