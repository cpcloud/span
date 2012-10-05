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

from future_builtins import map

from abc import ABCMeta, abstractproperty, abstractmethod

import numpy as np
from pandas import Series, DataFrame, MultiIndex

import span
from span.xcorr import xcorr
from span.tdt.spikeglobals import Indexer, ChannelIndex
from span.utils.decorate import cached_property, thunkify


class SpikeDataFrameAbstractBase(DataFrame):
    """Abstract base class for spike data frames.

    Parameters
    ----------
    data : array_like

    meta : array_like, optional
        TDT tsq file meta data. Defaults to None

    args, kwargs : tuple, dict
        Arguments to DataFrame

    Attributes
    ----------
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

    __metaclass__ = ABCMeta

    def __init__(self, data, meta, *args, **kwargs):
        super(SpikeDataFrameAbstractBase, self).__init__(data, *args, **kwargs)
        self.meta = meta

    @abstractproperty
    def channels(self):
        """Retrieve the data organized as a samples by channels DataFrame."""
        pass

    @abstractproperty
    def fs(self):
        """The sampling rate of the event."""
        pass

    @abstractproperty
    def nchans(self):
        """The number of channels in the array."""
        pass

    @abstractmethod
    def threshold(self, threshes):
        """Thresholding function for spike detection."""
        pass


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

    @property
    def nsamples(self): return self.channels.index[-1]

    @cached_property
    def nchans(self): return span.utils.cast(self.meta.channel.max() + 1, int)

    @property
    @thunkify
    def _channels(self):
        vals = self.values[self.channel_indices]
        shpsort = np.argsort(vals.shape)[::-1]
        newshp = int(vals.size // self.nchans), self.nchans
        valsr = vals.transpose(shpsort).reshape(newshp)
        return DataFrame(valsr, columns=ChannelIndex)

    @cached_property
    def channels(self): return self._channels()

    @property
    def channel_indices(self):
        return span.utils.group_indices(self.channel_group)

    @property
    def shank_indices(self): return span.utils.group_indices(self.shank_group)

    @property
    def side_indices(self): return span.utils.group_indices(self.side_group, str)

    @property
    def channel_group(self): return self.groupby(level=self.meta.channel.name)

    @property
    def shank_group(self): return self.groupby(level=self.meta.shank.name)

    @property
    def side_group(self): return self.groupby(level=self.meta.side.name)

    def threshold(self, threshes):
        cols = self.channels.columns
        threshes = np.asanyarray(threshes)

        assert threshes.size == 1 or threshes.size == cols.size, \
            'number of threshold values must be 1 (same for all channels) or {}'\
            ', different threshold for each channel'

        return self.channels.gt(Series(threshes, index=cols), axis='columns')


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
        binned = span.utils.bin_data(cleared.values, bins)
        return DataFrame(binned, columns=ChannelIndex)

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
        clr : DataFrame
            The thresholded and refractory-period-cleared array of booleans
            indicating the sample point at which a spike was above threshold.
        """
        clr = self.threshold(threshes)

        # TODO: make sure samples by channels is shape of clr
        span.utils.clear_refrac(clr.values, self.refrac_window(ms))
        return clr

    def fr(self, threshes, level='channel', axis=1, binsize=1000, ms=2):
        """Compute the firing rate over a given level.

        Parameters
        ----------
        threshes : array_like
            Threshold or threshold array.

        level : str, optional
            The level of the data set on which to run the analyses.

        axis : int, optional
            The axis on which to look for the level, defaults to 1.

        binsize : int, optional
            Bin size in milliseconds.

        ms : int, optional
            Length of the refractory period in milliseconds

        Returns
        -------
        fr, sem : array_like, array_like
            The average firing rate in spikes per binsize milliseconds and the
            standard error of the mean of the spike counts for a given level.
        """
        binned = self.bin(threshes, binsize=binsize, ms=ms)
        group = binned.groupby(axis=axis, level=level)
        sqrtn = np.sqrt(max(binned.shape))
        return group.mean().mean(), group.sum().std() / sqrtn

    def xcorr(self, threshes, ms=2, binsize=1000, maxlags=100,
              detrend=span.utils.detrend_mean, scale_type='normalize',
              reject_count=100, sortlevel='shank i', dropna=True,
              nan_auto=False):
        """Compute the cross correlation of binned data.

        Parameters
        ----------
        threshes : array_like
            Threshold(s) to use for spike detection.

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

        reject_count : int, optional
            Bins whose count is less than this will be assigned NaN. Defaults to
            100.

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
        assert isinstance(scale_type, basestring) or scale_type is None, \
            'scale_type must be a string or None'

        ms, binsize = float(ms), float(binsize)
        binned = self.bin(threshes, ms=ms, binsize=binsize).astype(float)
        assert reject_count >= 0, 'reject count must be a positive integer'

        if reject_count:
            rec_len_s = float(self.nsamples) / self.fs
            min_sp_per_s = reject_count / rec_len_s
            sp_per_s = binned.mean() * (1e3 / binsize)
            binned.ix[:, sp_per_s < min_sp_per_s] = np.nan

        nchannels = binned.columns.values.size

        channel_i, channel_j = 'channel i', 'channel j'
        channel_names = channel_i, channel_j
        lr = DataFrame(span.utils.ndtuples(nchannels, nchannels),
                       columns=channel_names)
        left, right = lr[channel_i], lr[channel_j]

        srt_idx = Indexer.sort('channel').reset_index(drop=True)

        lshank, rshank = srt_idx.shank[left], srt_idx.shank[right]
        lshank.name, rshank.name = 'shank i', 'shank j'

        lside, rside = srt_idx.side[left], srt_idx.side[right]
        lside.name, rside.name = 'side i', 'side j'

        index = MultiIndex.from_arrays((left, right, lshank, rshank, lside,
                                        rside))

        xc = xcorr(binned, maxlags=maxlags, detrend=detrend,
                   scale_type=scale_type)
        xc.columns = index

        if nan_auto:
            sz = xc.shape[1]
            sqrtsz = int(np.sqrt(sz))
            auto_inds = np.diag(np.r_[:sz].reshape(sqrtsz, sqrtsz))
            xc.ix[0, auto_inds] = np.nan

        if dropna:
            xc = xc.dropna(axis=1)
            binned = binned.dropna(axis=1)

        if sortlevel is not None:
            sl = np.array(sortlevel).item()
            assert isinstance(sl, (int, basestring)), \
                'sortlevel must be an int or a string'
                
            if isinstance(sl, int):
                nlevels = xc.columns.nlevels
                assert 0 <= sl < nlevels, \
                    'sortlevel: {0} not in {1}'.format(sl, range(nlevels))
            else:
                names = xc.columns.names
                assert sl in names, "'{0}' not in {1}".format(sl, names)

            xc = xc.sortlevel(level=sl, axis=1)

        return xc, binned
