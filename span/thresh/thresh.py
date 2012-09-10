#!/usr/bin/env python
import numpy as np
import pylab as np


from ..clear_refrac import thresh_and_clear
from ..utils import cast


def make_threshold(data, threshes=None, sc=5.0, const=0.6745):
    """
    Make a threshold from a data set.

    Parameters
    ----------
    data : array_like
    threshes : array_like, optional
    sc : float, optional
    const : float, optional

    Returns
    -------
    threshes : array_like
    """
    minshape = np.min(data.shape)
    if threshes is None:
        threshes = sc * data.median(axis=0) / constant
    elif np.isscalar(threshes):
        threshes = cast(np.repeat(threshes, minshape), data.dtype)
    elif pl.isvector(threshes):
        assert threshes.size == minshape, 'invalid threshold shape'
        if threshes.ndim > 1:
            threshes = threshes.squeeze()
    else:
        raise TypeError("invalid type of threshold, must be ndarray, "
                        "scalar, or None")
    return threshes


def spike_window(ms, fs, const=1e3):
    """Perform a transparent conversion from time to samples.

    Parameters
    ----------
    ms : int
    fs : float
    const : float, optional

    Returns
    -------
    Conversion of milliseconds to samples
    """
    return cast(np.floor(ms / const * fs), dtype=np.int32)
