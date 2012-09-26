#!/usr/bin/env python

"""
Summary
-------


Extended Summary
----------------


Routine Listing
---------------


See Also
--------


Notes
-----


References
----------


Examples
--------

"""

from future_builtins import map, zip

import abc
import itertools

import numpy as np
import pandas as pd

try:
    import pylab
    subplots = pylab.subplots
except RuntimeError:
    subplots = None

from span.xcorr import xcorr
from span.tdt.spikeglobals import Indexer

from span.utils.decorate import cached_property, thunkify
from span.utils import (
    bin_data, cast, clear_refrac, detrend_mean, group_indices, ndtuples,
    remove_legend)


class SpikeDataFrameAbstractBase(pd.DataFrame):
    """Abstract base class for spike data frames.

    Parameters
    ----------
    data : array_like

    meta : array_like, optional
        TDT tsq file meta data. Defaults to None

    args, kwargs : tuple, dict
        Arguments to pd.DataFrame

    Attributes
    ----------
    raw
    channels
    fs
    nchans

    See Also
    --------
    pandas.DataFrame
        The superclass of this class.

    numpy.ndarray
        The basis for numerics in Python
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self, data, meta, *args, **kwargs):
        """Constructor.

        """
        super(SpikeDataFrameAbstractBase, self).__init__(data, *args, **kwargs)
        self.meta = meta

    @abc.abstractproperty
    def raw(self):
        """Retrieve the underlying raw NumPy array."""
        raise NotImplementedError

    @abc.abstractproperty
    def channels(self):
        """Retrieve the data organized as a samples by channels DataFrame."""
        raise NotImplementedError

    @abc.abstractproperty
    def fs(self):
        """The sampling rate of the event."""
        raise NotImplementedError

    @abc.abstractproperty
    def nchans(self):
        """The number of channels in the array."""
        raise NotImplementedError

    @abc.abstractmethod
    def threshold(self, threshes):
        """Thresholding function for spike detection."""
        raise NotImplementedError


class SpikeDataFrameBase(SpikeDataFrameAbstractBase):
    """Base class implementing basic spike data set properties and methods.

    Attributes
    ----------

    Parameters
    ----------

    See Also
    --------
    """
    def __init__(self, *args, **kwargs):
        super(SpikeDataFrameBase, self).__init__(*args, **kwargs)

    @cached_property
    def fs(self): return self.meta.fs.max()

    @cached_property
    def nchans(self): return cast(self.meta.channel.max() + 1, int)


    @property
    @thunkify
    def _channels(self):
        # get the channel indices
        inds = self.channel_indices

        # get the 3D array of raw values
        vals = self.values[inds.values]

        # number of channels
        nch = inds.columns.values.size

        # get indices of the sorted (descending) dimensions of vals
        shpsort = np.asanyarray(vals.shape).argsort()[::-1]

        # transpose vals to make a reshape into a samples x channels array
        valsr = vals.transpose(shpsort).reshape(vals.size // nch, nch)

        # columns = inds.columns
        return pd.DataFrame(valsr, columns=self.channel_index)

    @cached_property
    def channel_index(self):
        srt_idx = Indexer.sort('channel').reset_index(drop=True)
        channel, shank, side = srt_idx.channel, srt_idx.shank, srt_idx.side
        return pd.MultiIndex.from_arrays((channel, shank, side))

    @cached_property
    def channels(self): return self._channels()

    @property
    def channel_indices(self): return group_indices(self.channel_group)

    @property
    def shank_indices(self): return group_indices(self.shank_group)

    @property
    def side_indices(self): return group_indices(self.side_group, str)

    @property
    def channel_group(self): return self.groupby(level=self.meta.channel.name)

    @property
    def shank_group(self): return self.groupby(level=self.meta.shank.name)

    @property
    def side_group(self): return self.groupby(level=self.meta.side.name)

    @property
    def raw(self): return self.channels.values

    def threshold(self, threshes): return self.channels > threshes


class SpikeDataFrame(SpikeDataFrameBase):
    """Class encapsulting a Pandas DataFrame with extensions for analyzing spike
    train data.

    Attributes
    ----------

    Parameters
    ----------

    See Also
    --------
    pandas.DataFrame
        Root class of this class
    """
    def __init__(self, *args, **kwargs):
        super(SpikeDataFrame, self).__init__(*args, **kwargs)

    @property
    def _constructor(self):
        return lambda *args, **kwargs: SpikeDataFrame(*args, meta=self.meta, **kwargs)

    def bin(self, threshes, ms=2, binsize=1000):
        """Bin spike data by `ms` millisecond bins.

        Parameters
        ----------
        threshes: array_like
        ms : float, optional
            Refractory period
        binsize : float
            The size of the bins to use, in milliseconds
        conv : float
            The conversion factor to convert the the binsize to samples

        Returns
        -------
        df : pandas.DataFrame

        See Also
        --------
        span.utils.bin
        """
        conv = 1e3
        ms, binsize = map(float, (ms, binsize))
        cleared = self.cleared(threshes, ms)
        max_sample = self.channels.index[-1]
        bin_samples = int(np.floor(binsize * self.fs / conv))
        bins = np.r_[:max_sample:bin_samples]
        binned = bin_data(cleared.values, bins)
        return pd.DataFrame(binned, columns=self.channel_index)

    def refrac_window(self, ms):
        """Compute the refractory period in samples given a period of `ms`
        milliseconds.

        Parameters
        ----------
        ms : float
            The refractory period in milliseconds.

        Returns
        -------
        win : int
            The refractory period in samples.
        """
        conv = 1e3
        return int(np.floor(float(ms) / conv * self.fs))

    def cleared(self, threshes, ms=2):
        """Remove spikes from the refractory period of all channels.

        Parameters
        ----------
        threshes : array_like
            A single number for the whole array or an array_like object
            that has shape == (number of columns of self,).

        ms : float, optional
            The length of the refractory period in milliseconds.

        Returns
        -------
        clr : pd.DataFrame
            The thresholded and refractory-period-cleared array of booleans
            indicating the sample point at which a spike was above threshold.
        """
        clr = self.threshold(threshes)

        # TODO: fragile indexing here make sure samples by channels is shape of
        # input
        clear_refrac(clr.values, self.refrac_window(ms))
        return clr

    def fr(self, threshes, level='channel', axis=1, binsize=1000, ms=2):
        """Compute the firing rate over a given level.

        Parameters
        ----------
        threshes : array_like
        level : str, optional
        binsize : int, optional
        ms : int, optional

        Returns
        -------
        fr : array_like
        """
        binned = self.bin(threshes, binsize=binsize, ms=ms)
        group = binned.groupby(axis=axis, level=level)
        s = group.sum()
        sqrtn = np.sqrt(s.sum())
        return group.mean().mean(), s.std() / sqrtn


    def xcorr(self, threshes, ms=2, binsize=1000, maxlags=100,
              detrend=detrend_mean, scale_type='normalize'):
        """Compute the cross correlation of binned data.

        Parameters
        ----------
        threshes : array_like
            Threshold(s) to use for spike detection

        ms : float, optional
            The refractory period of a channel. Defaults to 2 ms.

        binsize : float, optional
            The size of the bins to use to count up spikes, in milliseconds.
            Defaults to 1000 ms (1 s).

        maxlags : int, optional
            Maximum number of lags to return from the cross correlation. Defaults
            to 100.

        detrend : callable, optional
            Callable used to detrend. Defaults to detrend_mean

        scale_type : str, optional
            Method of scaling. Defaults to 'normalize'.

        Raises
        ------
        AssertionError
           If detrend is not a callable object
           If scale_type is not a string

        Returns
        -------
        xc : DataFrame
            The cross correlation of all the columns of the data.

        See Also
        --------
        span.tdt.xcorr.xcorr
            General cross correlation function.

        SpikeDataFrame.bin
            Binning function.

        SpikeDataFrame.cleared
            Clear the refractory period of a channel.
        """
        assert callable(detrend), 'detrend must be a callable class or function'
        assert isinstance(scale_type, basestring), 'scale_type must be a string'

        ms, binsize = map(float, (ms, binsize))
        binned = self.bin(threshes, ms=ms, binsize=binsize)
        nchannels = binned.columns.values.size
        left, right = ndtuples(nchannels, nchannels).T
        left, right = map(pd.Series, (left, right))
        left.name, right.name = 'channel i', 'channel j'
        sorted_indexer = Indexer.sort('channel').reset_index(drop=True)
        lshank, rshank = sorted_indexer.shank[left], sorted_indexer.shank[right]
        lshank.name, rshank.name = 'shank i', 'shank j'
        index = pd.MultiIndex.from_arrays((left, right, lshank, rshank))
        xc = xcorr(binned, maxlags=maxlags, detrend=detrend,
                   scale_type=scale_type).T
        xc.index = index
        return xc
