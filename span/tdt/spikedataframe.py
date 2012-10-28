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

import gc
import numbers
import collections
import functools
import abc
import operator

import numpy as np
import scipy
import scipy.signal

import pandas as pd
from pandas import Series, DataFrame, MultiIndex, date_range, datetools

import span
from span.xcorr import xcorr
from span.utils.decorate import cached_property, thunkify

try:
    from pylab import subplots
except RuntimeError:
    subplots = None


def is_valid_array(a):
    assert ((hasattr(a, 'values') and not isinstance(a, collections.Mapping))
            or isinstance(a, np.ndarray)), \
        '{} must be an instance of '


def find_names(obj):
    return [k for ref in gc.get_referrers(obj) if isinstance(ref, dict)
            for k, v in ref.iteritems() if v is obj]


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

    __metaclass__ = abc.ABCMeta

    def __init__(self, data, meta, *args, **kwargs):
        super(SpikeDataFrameAbstractBase, self).__init__(data, *args, **kwargs)

        assert meta is not None, 'meta cannot be None'
        self.meta = meta
        self.date = meta.timestamp[0]

    @abc.abstractproperty
    def channels(self):
        """Retrieve the data organized as a samples by channels DataFrame."""
        pass

    @abc.abstractproperty
    def fs(self):
        """The sampling rate of the event."""
        pass

    @abc.abstractproperty
    def nchans(self):
        """The number of channels in the array."""
        pass

    @abc.abstractproperty
    def nsamples(self):
        pass

    @abc.abstractmethod
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

    @property
    def index_values(self):
        return span.utils.index_values(self.index)

    def downsample(self, factor, n=None, ftype='iir', axis=-1):
        """Downsample the data by an integer factor.

        Parameters
        ----------
        factor
        n
        ftype
        axis

        Returns
        -------
        dns : DataFrame
            Downsampled data in a DataFrame
        """
        dns = scipy.signal.decimate(self.channels.values.T, factor, n, ftype,
                                    axis)
        return DataFrame(dns.T, columns=self.channels.columns)

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
        from span.tdt.spikeglobals import ChannelIndex as columns

        vals = self.values[self.channel_indices]

        shpsort = np.argsort(vals.shape)[::-1]
        newshp = int(vals.size / self.nchans), self.nchans

        valsr = vals.transpose(shpsort).reshape(newshp)

        us_per_sample = (1e6 / self.fs) * datetools.Micro()
        index = date_range(self.date, periods=valsr.shape[0], freq=us_per_sample,
                           name='time')
        return DataFrame(valsr, columns=columns, index=index)

    @cached_property
    def channels(self): return self._channels()

    @property
    def channel_indices(self):
        return span.utils.group_indices(self.channel_group)

    @property
    def all_indices(self):
        gb = self.groupby(level=('shank', 'channel', 'side'))
        return DataFrame(gb.indices)

    @property
    def channel_group(self): return self.groupby(level=self.meta.channel.name)

    def spike_times(self, threshed):
        joiners = [self.channels.ix[threshed[k]] for k in threshed.columns]
        concated = pd.concat(joiners)
        concated.drop_duplicates(inplace=True)
        return Series(concated.index)

    def threshold(self, threshes):
        """Threshold spikes.

        Parameters
        ----------
        threshes : array_like

        Returns
        -------
        threshed : array_like
        """
        threshes = np.asanyarray(threshes)

        assert threshes.size == 1 or threshes.size == self.nchans, \
            'number of threshold values must be 1 (same for all channels) or {}'\
            ', different threshold for each channel'

        is_neg = np.all(threshes < 0)

        if threshes.size == self.nchans:
            threshes = Series(threshes, index=self.channels.columns)
            chn = self.channels
            f = functools.partial(chn.lt if is_neg else chn.gt, axis='columns')
        else:
            cmpf = operator.lt if is_neg else operator.gt
            f = functools.partial(cmpf, self.channels)

        return f(threshes)


class SpikeDataFrame(SpikeDataFrameBase):
    """Class encapsulting a Pandas DataFrame with extensions for analyzing spike
    train data.

    Attributes
    ----------

    See Also
    --------
    pandas.DataFrame
        Root class of this class

    span.tdt.SpikeDataFrameBase
        Base class
    """
    def __init__(self, *args, **kwargs):
        super(SpikeDataFrame, self).__init__(*args, **kwargs)

    @property
    def _constructor(self):
        return lambda *args, **kwargs: SpikeDataFrame(*args, meta=self.meta, **kwargs)

    def bin(self, cleared, binsize, reject_count=100, dropna=False):
        """Bin spike data by `ms` millisecond bins.

        Parameters
        ----------
        cleared : array_like
            The refractory-period-cleared array of spike booleans to bin.

        binsize : float, optional
            The size of the bins to use, in milliseconds

        reject_count : int, optional

        dropna : bool, optional

        Returns
        -------
        binned : DataFrame

        See Also
        --------
        span.utils.bin
        """
        assert hasattr(cleared, 'values') or isinstance(cleared, np.ndarray), \
            '"cleared" must be have the "values" attribute or be an instance of'\
            ' numpy.ndarray'
        assert binsize > 0 and isinstance(binsize, numbers.Real), \
            '"binsize" must be a positive number'
        assert reject_count >= 0, '"reject_count" must be a non negative integer'

        conv = 1e3
        bin_samples = span.utils.cast(np.floor(binsize * self.fs / conv), int)
        bins = np.arange(0, self.nsamples - 1, bin_samples, np.uint64)
        btmp = span.utils.bin_data(cleared.values.view(np.uint8), bins)

        # make a datetime index of seconds
        freq = binsize * datetools.Milli()
        index = date_range(start=self.date, periods=btmp.shape[0], freq=freq,
                           name='time')

        binned = DataFrame(btmp, index, cleared.columns, float)

        # samples / (samples / s) == s
        rec_len_s = self.nsamples / self.fs

        # spikes / s
        min_sp_per_s = reject_count / rec_len_s

        # spikes / s * ms / ms == spikes / s
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

        if ms is None or ms > 0:
            clr = threshed.values.copy()

            # TODO: make sure samples by channels is shape of clr
            ms_fs = span.utils.fs2ms(self.fs, ms)
            span.utils.clear_refrac(clr.view(np.uint8), ms_fs)
            r = DataFrame(clr, threshed.index, threshed.columns)
        else:
            r = threshed

        return r

    def fr(self, binned, level='channel', axis=1, sem=False):
        """Compute the firing rate over a given level.

        Parameters
        ----------
        binned : array_like
            Threshold scalar array.

        level : str, optional
            The level of the data set on which to run the analyses.

        axis : int, optional
            The axis on which to look for the level, defaults to 1.

        sem : bool, optional
            Whether to return the standard error of the mean along with the
            average firing rate.

        Returns
        -------
        fr, sem : array_like, array_like
            The average firing rate in spikes per binsize milliseconds and the
            standard error of the mean of the spike counts for a given level.
        """
        group = binned.groupby(axis=axis, level=level)
        r = group.mean().mean()

        if sem:
            sqrtn = np.sqrt(binned.index.shape[0])
            r = r, group.sum().std() / sqrtn

        return r

    def xcorr(self, binned, maxlags=None, detrend=span.utils.detrend_mean,
              scale_type='normalize', sortlevel='shank i', dropna=False,
              nan_auto=False):
        """Compute the cross correlation of binned data.

        Parameters
        ----------
        binned : array_like
            Data of which to compute the cross-correlation.

        maxlags : int, optional
            Maximum number of lags to return from the cross correlation.
            Defaults to None and computes the full cross correlation.

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

        nan_auto : bool, optional
            If True then the autocorrelation values will be NaN. Defaults to
            False

        Raises
        ------
        AssertionError
           If detrend is not a callable object
           If scale_type is not a string or is not None

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

        xc = xcorr(binned, maxlags=maxlags, detrend=detrend,
                   scale_type=scale_type)

        xc.columns = _create_xcorr_inds(self.nchans)

        if nan_auto:
            npairs = self.nchans ** 2
            auto_inds = np.diag(np.r_[:npairs].reshape(self.nchans, self.nchans))
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


def _create_xcorr_inds(nchannels):
    from span.utils import ndtuples
    from span.tdt.spikeglobals import Indexer

    channel_i, channel_j = 'channel i', 'channel j'
    channel_names = channel_i, channel_j

    lr = DataFrame(ndtuples(nchannels, nchannels), columns=channel_names)
    left, right = lr[channel_i], lr[channel_j]

    srt_idx = Indexer.sort('channel').reset_index(drop=True)

    lshank, rshank = srt_idx.shank[left], srt_idx.shank[right]
    lshank.name, rshank.name = 'shank i', 'shank j'

    lside, rside = srt_idx.side[left], srt_idx.side[right]
    lside.name, rside.name = 'side i', 'side j'

    return MultiIndex.from_arrays((left, right, lshank, rshank, lside, rside))
