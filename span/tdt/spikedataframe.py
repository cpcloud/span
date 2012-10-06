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

from abc import ABCMeta, abstractproperty, abstractmethod

from functools import partial
from operator import lt, gt

import numpy as np
from pandas import Series, DataFrame, MultiIndex

import span
from span.xcorr import xcorr
from span.tdt.spikeglobals import Indexer, ChannelIndex
from span.utils.decorate import cached_property, thunkify
from span.utils import clear_refrac, refrac_window


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

        assert meta is not None, 'meta cannot be None'
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
    def fs(self): return self.meta.fs.unique().item()

    @cached_property
    def chunk_size(self): return self.meta.size.unique().item()

    @cached_property
    def sort_code(self): return self.meta.sort_code.unique().item()

    @cached_property
    def fmt(self): return self.meta.format.unique().item()

    @cached_property
    def tdt_type(self): return self.meta.type.unique().item()

    @cached_property
    def nchans(self): return self.meta.channel.dropna().nunique()

    @property
    def nsamples(self): return max(self.channel_indices.shape) * self.chunk_size

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
    def channel_group(self): return self.groupby(level=self.meta.channel.name)

    def threshold(self, threshes):
        threshes = np.asanyarray(threshes)

        assert threshes.size == 1 or threshes.size == self.nchans, \
            'number of threshold values must be 1 (same for all channels) or {}'\
            ', different threshold for each channel'

        is_neg = np.all(threshes < 0)

        if threshes.size == self.nchans:
            threshes = Series(threshes, index=self.channels.columns)
            chn = self.channels
            f = partial(chn.lt if is_neg else chn.gt, axis='columns')
        else:
            f = partial(lt if is_neg else gt, self.channels)
            
        return f(threshes)


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

    def bin(self, cleared, binsize=1000, reject_count=100, dropna=False):
        """Bin spike data by `ms` millisecond bins.

        Parameters
        ----------
        cleared : array_like
            The refractory-period-cleared array of spike booleans to bin.
        
        binsize : float, optional
            The size of the bins to use, in milliseconds

        reject_count : int, optional

        Returns
        -------
        binned : pandas.DataFrame

        See Also
        --------
        span.utils.bin
        """
        assert reject_count >= 0, 'reject count must be a positive integer'

        conv = 1e3
        bin_samples = int(np.floor(binsize * self.fs / conv))
        bins = np.r_[:self.nsamples - 1:bin_samples]
        btmp = span.utils.bin_data(cleared.values, bins)
        binned = DataFrame(btmp, columns=cleared.columns, dtype=float)

        if reject_count:
            rec_len_s = self.nsamples / float(self.fs)
            min_sp_per_s = reject_count / rec_len_s
            sp_per_s = binned.mean() * (1e3 / binsize)
            binned.ix[:, sp_per_s < min_sp_per_s] = np.nan

        if dropna:
            binned = binned.dropna(axis=1)
        
        return binned

    def cleared(self, threshed, ms=2):
        """Remove spikes from the refractory period of all channels.

        Parameters
        ----------
        threshed : array_like

        ms : float, optional
            The length of the refractory period in milliseconds.

        Raises
        ------
        AssertionError
            If `ms` is less than 0 or is not None

        Returns
        -------
        r : DataFrame
            The thresholded and refractory-period-cleared array of booleans
            indicating the sample point at which a spike was above threshold.
        """
        assert ms >= 0 or ms is None, \
            'refractory period must be a positive integer or None'

        if ms:
            clr = threshed.values.copy()

            # TODO: make sure samples by channels is shape of clr
            clear_refrac(clr, refrac_window(self.fs, ms))
            r = DataFrame(clr, index=threshed.index, columns=threshed.columns)
        else:
            r = threshed
        
        return r

    def fr(self, binned, level='channel', axis=1, sem=False):
        """Compute the firing rate over a given level.

        Parameters
        ----------
        binned : array_like
            Threshold or threshold array.

        level : str, optional
            The level of the data set on which to run the analyses.

        axis : int, optional
            The axis on which to look for the level, defaults to 1.

        Returns
        -------
        fr, sem : array_like, array_like
            The average firing rate in spikes per binsize milliseconds and the
            standard error of the mean of the spike counts for a given level.
        """
        group = binned.groupby(axis=axis, level=level)
        sqrtn = np.sqrt(max(binned.shape))
        r = group.mean().mean()

        if sem:
            r = r, group.sum().std() / sqrtn

        return r

    def xcorr(self, binned, maxlags=100, detrend=span.utils.detrend_mean,
              scale_type='normalize', sortlevel='shank i', dropna=False,
              nan_auto=False):
        """Compute the cross correlation of binned data.

        Parameters
        ----------
        binned : array_like
            Data of which to compute the cross-correlation.

        maxlags : int, optional
            Maximum number of lags to return from the cross correlation. Defaults
            to 100.

        detrend : callable, optional
            Callable used to detrend. Defaults to detrend_mean

        scale_type : str, optional
            Method of scaling. Defaults to 'normalize'.

        sortlevel : str, optional
            How to sort the index of the returned cross-correlation(s). Defaults
            to "shank i" so the the xcorrs are ordered by their physical
            ordering.

        dropna : bool, optional
            If True this will drop all channels whose cross correlation is NaN.
            Cross-correlations will be NaN if any of the columns of `binned` are
            NaN.

        Raises
        ------
        AssertionError
           If detrend is not a callable object
           If scale_type is not a string

        ValueError
           If sortlevel is not None and is not a string or a number in the
               list of level names or level indices

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

        if sortlevel is not None:
            try:
                sl = int(sortlevel)
                nlevels = xc.columns.nlevels
                assert 0 <= sl < nlevels, \
                    'sortlevel {0} not in {1}'.format(sl, range(nlevels))
            except ValueError:
                try:
                    sl = str(sortlevel)
                    names = xc.columns.names
                    assert sl in names, "'{0}' not in {1}".format(sl, names)
                except ValueError:
                    raise ValueError('sortlevel must be an int or a string')

            xc = xc.sortlevel(level=sl, axis=1)

        return xc
