from pandas import MultiIndex
import numpy as np


def permute_channels(mi):
    shank, channel = mi.levels
    return MultiIndex.from_arrays((shank, np.random.permutation(channel)))
