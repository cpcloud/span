#!/usr/bin/env python

# spikedataframe.py ---

# Copyright (C) 2012 Copyright (C) 2012 Phillip Cloud <cpcloud@gmail.com>

# Author: Phillip Cloud <cpcloud@gmail.com>

# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.


"""
Examples
--------
>>> import span
>>> tank = span.tdt.PandasTank('some/tank/file/folder')
>>> sp = tank.spikes
>>> assert isinstance(sp, span.tdt.SpikeDataFrame)
"""

import numbers
import functools as fntools
import operator

import numpy as np
import scipy
import scipy.signal

from pandas import (Series, DataFrame, MultiIndex, date_range, datetools,
                    Timestamp)

import span
from span.xcorr import xcorr
from span.utils.decorate import cached_property


try:
    from pylab import subplots
except RuntimeError:
    subplots = None


class _ChannelGetter(object):
    def __init__(self, obj):
        super(_ChannelGetter, self).__init__()
        self.obj = obj

    def __getitem__(self, i):
        return self.obj.ix[:, i]


class _ShankGetter(object):
    def __init__(self, df):
        super(_ShankGetter, self).__init__()
        self.df = df

    def __getitem__(self, i):
        return self.df.select(lambda x: x[1] == i, axis=1)


class _SideGetter(object):
    def __init__(self, df):
        super(_SideGetter, self).__init__()
        self.df = df

    def __getitem__(self, i):
        return self.df.select(lambda x: x[-1] == i, axis=1)


class SpikeGrouper(type):
    def __new__(cls, name, parents, dct):
        dct['channel'] = property(fget=_ChannelGetter)
        dct['shank'] = property(fget=_ShankGetter)
        dct['side'] = property(fget=_SideGetter)

        return type.__new__(cls, name, parents, dct)


class GroupedDataFrame(DataFrame):

    __metaclass__ = SpikeGrouper

    def __init__(self, *args, **kwargs):
        super(GroupedDataFrame, self).__init__(*args, **kwargs)

    @property
    def _constructor(self):
        return GroupedDataFrame


class SpikeDataFrameBase(GroupedDataFrame):
    """Base class implementing basic spike data set properties and methods.

    Attributes
    ----------
    fs (float) : Sampling rate
    nchans (int) : Number of channels
    nsamples (int) : Number of samples per channel
    chunk_size (int) : Samples per chunk
    sort_code (int) : I have no idea what the sort code is
    fmt (dtype) : The dtype of the event
    tdt_type (int) : The integer corresponding to the type of event

    See Also
    --------
    span.tdt.spikedataframe.SpikeDataFrame
    """

    def __init__(self, data, meta, *args, **kwargs):
        super(SpikeDataFrameBase, self).__init__(data, *args, **kwargs)

        assert meta is not None, 'meta cannot be None'

        self.meta = meta
        self.date = Timestamp(self.meta.timestamp[0])

    def downsample(self, factor, n=None, ftype='iir', axis=-1):
        """Downsample the data by an integer factor.

        Parameters
        ----------
        factor : int
            Factor by which to downsample

        n : int, optional

        ftype : str, optional
            Type of filter to use to downsample

        axis : int, optional
            Axis over which to downsample

        Returns
        -------
        dns : DataFrame
            Downsampled data.

        See Also
        --------
        scipy.signal.decimate
        """
        dec_s = scipy.signal.decimate(self.values.T, factor, n, ftype, axis)
        return DataFrame(dec_s.T, columns=self.columns)

    @cached_property
    def fs(self):
        return self.meta.fs.unique().item()

    @cached_property
    def chunk_size(self):
        return self.meta.size.unique().item()

    @cached_property
    def sort_code(self):
        return self.meta.sort_code.unique().item()

    @cached_property
    def fmt(self):
        return self.meta.format.unique().item()

    @cached_property
    def tdt_type(self):
        return self.meta.type.unique().item()

    @property
    def nchans(self):
        return min(self.shape)

    @property
    def nsamples(self):
        return max(self.shape)

    def threshold(self, threshes):
        """Threshold spikes.

        Parameters
        ----------
        threshes : array_like

        Raises
        ------
        AssertionError
            If `threshes` is not a scalar or a vector of length == number of
            channels

        Returns
        -------
        threshed : array_like
        """
        threshes = np.asanyarray(threshes)

        assert threshes.size == 1 or threshes.size == self.nchans, \
            'number of threshold values must be 1 (same for all channels) or '\
            '{0}, different threshold for each channel'.format(self.nchans)

        is_neg = np.all(threshes < 0)

        if threshes.size == self.nchans:
            threshes = Series(threshes, index=self.columns)
            cmpf = self.lt if is_neg else self.gt
            f = fntools.partial(cmpf, axis=1)
        else:
            cmpf = operator.lt if is_neg else operator.gt
            f = fntools.partial(cmpf, self)

        return f(threshes)


class SpikeDataFrame(SpikeDataFrameBase):
    """Class encapsulting a Pandas DataFrame with extensions for analyzing
    spike train data.

    Attributes
    ----------

    See Also
    --------
    pandas.DataFrame
    span.tdt.SpikeDataFrameBase
    """

    def __init__(self, *args, **kwargs):
        super(SpikeDataFrame, self).__init__(*args, **kwargs)

    @cached_property
    def _constructor(self):
        self_t = type(self)
        return lambda *args, **kwargs: self_t(*args, meta=self.meta, **kwargs)

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

        Raises
        ------
        AssertionError

        Returns
        -------
        binned : DataFrame

        See Also
        --------
        span.utils.bin
        """
        assert hasattr(cleared, 'values') or isinstance(cleared, np.ndarray), \
            '"cleared" must be have the "values" attribute or be an instance '\
            'of numpy.ndarray'
        assert binsize > 0 and isinstance(binsize, numbers.Real), \
            '"binsize" must be a positive number'
        assert reject_count >= 0, '"reject_count" must be a nonnegative integer'

        conv = 1e3
        bin_samples = span.utils.cast(np.floor(binsize * self.fs / conv), int)
        bins = np.arange(0, self.nsamples - 1, bin_samples, np.uint64)
        btmp = span.utils.bin_data(cleared.values.view(np.uint8), bins)

        # make a datetime index of seconds
        freq = binsize * datetools.Milli()
        index = date_range(start=self.date, periods=btmp.shape[0], freq=freq,
                           name='time', tz='US/Eastern')

        binned = DataFrame(btmp, index=index, columns=cleared.columns,
                           dtype=float)

        # samples / (samples / s) == s
        rec_len_s = self.nsamples / self.fs

        # spikes / s
        min_sp_per_s = reject_count / rec_len_s

        # spikes / s * ms / ms == spikes / s
        sp_per_s = binned.mean() * (1e3 / binsize)

        binned.ix[:, sp_per_s < min_sp_per_s] = np.nan

        if dropna:
            binned = binned.dropna(axis=1)

        return GroupedDataFrame(binned)

    def clear_refrac(self, threshed, ms=2):
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
        r : SpikeDataFrame
            The thresholded and refractory-period-cleared array of booleans
            indicating the sample point at which a spike was above threshold.
        """
        assert ms > 0 or ms is None, \
            'refractory period must be a positive integer or None'

        if ms > 0:
            clr = threshed.values.copy()

            # TODO: make sure samples by channels is shape of clr
            ms_fs = span.utils.fs2ms(self.fs, ms)
            span.utils.clear_refrac(clr.view(np.uint8), ms_fs)
            r = SpikeDataFrame(clr, self.meta, index=threshed.index,
                               columns=threshed.columns)
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
            The level of the data set on which to run the analyses. Defaults
            to "channel".

        axis : int, optional
            The axis on which to look for the level; defaults to 1.

        sem : bool, optional
            Whether to return the standard error of the mean along with the
            average firing rate; defaults to ``False``.

        Returns
        -------
        fr, sem : array_like, array_like
            The average firing rate in spikes per binsize milliseconds and the
            standard error of the mean of the spike counts for a given level.
        """
        group = binned.groupby(axis=axis, level=level)
        gm = group.mean()

        r = gm.mean()

        if sem:
            r = r, gm.apply(span.utils.math.sem)

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
            How to sort the index of the returned cross-correlation(s).
            Defaults to "shank i" so the the xcorrs are ordered by their
            physical ordering.

        dropna : bool, optional
            If True this will drop all channels whose cross correlation is NaN.
            Cross-correlations will be NaN if any of the columns of `binned`
            are NaN.

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
        assert callable(detrend), 'detrend must be a callable class or '\
            'function'
        assert isinstance(scale_type, basestring) or scale_type is None, \
            'scale_type must be a string or None'

        xc = xcorr(binned, maxlags=maxlags, detrend=detrend,
                   scale_type=scale_type)

        xc.columns = _create_xcorr_inds(self.nchans)

        if nan_auto:
            npairs = self.nchans ** 2
            auto_inds = np.diag(np.r_[:npairs].reshape(self.nchans,
                                                       self.nchans))
            xc.ix[0, auto_inds] = np.nan

        if dropna:
            xc = xc.dropna(axis=1)

        if sortlevel is not None:
            fmt_str = 'sortlevel {0} not in {1}'

            try:
                sl = int(sortlevel)
                nlevels = xc.columns.nlevels
                assert 0 <= sl < nlevels, fmt_str.format(sl, range(nlevels))
            except ValueError:
                try:
                    sl = str(sortlevel)
                    names = xc.columns.names
                    assert sl in names, fmt_str.format(sl, names)
                except ValueError:
                    raise ValueError('sortlevel must be an int or a string')

            xc = xc.sortlevel(level=sl, axis=1)

        xc.index.name = r'$\ell$'

        return xc


def _create_xcorr_inds(nchannels):
    """Create a ``MultiIndex`` for `nchannels` channels.

    Parameters
    ----------
    nchannels : int

    Returns
    -------
    xc_inds : MultiIndex
    """
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
