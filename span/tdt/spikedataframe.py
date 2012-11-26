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

import numpy as np
import scipy
import scipy.signal

from pandas import (Series, DataFrame, MultiIndex, date_range, datetools,
                    Timestamp)

import span
from span.xcorr import xcorr
from span.utils.decorate import cached_property
from span.utils import sem, cast, fs2ms, bin_data, clear_refrac, ndtuples

try:
    from pylab import subplots
except RuntimeError:  # pragma: no cover
    subplots = NotImplemented


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
    """Metaclass for creating grouping selectors.

    This is mostly for convenience, and could probably be made more general.
    That is, have a function to create classes on the fly that index into the
    ix or call select to retreive a subset of the data.
    """
    def __new__(cls, name, parents, dct):
        dct['channel'] = property(fget=_ChannelGetter)
        dct['shank'] = property(fget=_ShankGetter)
        dct['side'] = property(fget=_SideGetter)

        return type.__new__(cls, name, parents, dct)


class SpikeGroupedDataFrame(DataFrame):
    """Base class for spike data frames."""

    __metaclass__ = SpikeGrouper

    def __init__(self, *args, **kwargs):
        super(SpikeGroupedDataFrame, self).__init__(*args, **kwargs)

    @property
    def _constructor(self):
        return SpikeGroupedDataFrame

    def sem(self, axis=0, ddof=1):
        r"""Return the standard error of the mean of array along `axis`.

        Parameters
        ----------
        axis : int, optional
            The axis along which to compute the standard error of the mean.

        ddof : int, optional
            Delta degrees of freedom. 0 computes the sem using the population
            standard deviation, 1 computes the sem using the sample standard
            deviation.

        Returns
        -------
        sem : Series
            The standard error of the mean of the object along `axis`.

        Notes
        -----
        The standard error of the mean is defined as:

        .. math::

           \operatorname{sem}\left(\mathbf{x}\right)=\sqrt{\frac{\frac{1}{n -
           \textrm{ddof}}\sum_{i=1}^{n}\left(x_{i} -
           \bar{\mathbf{x}}\right)^{2}}{n}}

        where :math:`n` is the number of elements along the axis `axis`.
        """
        return self.apply(sem, axis=axis, ddof=ddof)


class SpikeDataFrameBase(SpikeGroupedDataFrame):
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

    @property
    def _constructor(self):
        return SpikeDataFrameBase

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
        return self._constructor(dec_s.T, self.meta, columns=self.columns)

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
        threshes = np.asanyarray(np.atleast_1d(threshes))

        assert threshes.size == 1 or threshes.size == self.nchans, \
            'number of threshold values must be 1 (same for all channels) or '\
            '{0}, different threshold for each channel'.format(self.nchans)

        is_neg = np.all(threshes < 0)

        threshes = Series(threshes, index=self.columns)
        cmpf = self.lt if is_neg else self.gt
        f = fntools.partial(cmpf, axis=1)

        return f(threshes)


class SpikeDataFrame(SpikeDataFrameBase):
    """Class encapsulting a Pandas DataFrame with extensions for analyzing
    spike train data.

    See Also
    --------
    pandas.DataFrame
    SpikeDataFrameBase
    """

    def __init__(self, *args, **kwargs):
        super(SpikeDataFrame, self).__init__(*args, **kwargs)

    @property
    def _constructor(self):
        self_t = type(self)
        return lambda *args, **kwargs: self_t(*args, meta=self.meta, **kwargs)

    def bin(self, cleared, binsize, reject_count=100, dropna=False):
        """Bin spike data by `ms` millisecond bins.

        Roughly, sum up the ones (and zeros) in the data using bins of size
        `binsize`.

        See ``span.utils.utils.bin_data`` for the actual loop that
        executes this binning. This method is a wrapper around that function.

        Parameters
        ----------
        cleared : array_like
            The refractory-period-cleared array of booleans to bin.

        binsize : numbers.Real
            The size of the bins to use, in milliseconds

        reject_count : numbers.Real, optional, default 100
            NaN channels whose firing rates are less than this number / sec

        dropna : bool, optional
            Whether to drop NaN'd values if any

        Raises
        ------
        AssertionError
            If `binsize` is not a positive number or if `reject_count` is not a
            nonnegative number

        Returns
        -------
        binned : SpikeGroupedDataFrame of float64

        See Also
        --------
        span.utils.utils.bin_data
        """
        assert binsize > 0 and isinstance(binsize, numbers.Real), \
            '"binsize" must be a positive number'
        assert reject_count >= 0 and isinstance(reject_count, numbers.Real), \
            '"reject_count" must be a nonnegative real number'

        ms_per_s = 1e3
        bin_samples = cast(np.floor(binsize * self.fs / ms_per_s), np.uint64)
        bins = np.arange(start=0, stop=self.nsamples - 1, step=bin_samples,
                         dtype=np.uint64)

        shape = bins.shape[0] - 1, cleared.shape[1]
        btmp = np.empty(shape, np.uint64)

        bin_data(cleared.values.view(np.uint8), bins, btmp)

        # make a datetime index of seconds
        freq = binsize * datetools.Milli()
        index = date_range(start=self.date, periods=btmp.shape[0], freq=freq,
                           name='time', tz='US/Eastern')

        binned = DataFrame(btmp, index=index, columns=cleared.columns,
                           dtype=np.float64)

        # samples / (samples / s) == s
        rec_len_s = self.nsamples / self.fs

        # spikes / s
        min_sp_per_s = reject_count / rec_len_s

        # spikes / s * ms / ms == spikes / s
        sp_per_s = binned.mean() * (1e3 / binsize)

        # nan out the counts that are not about reject_count / s
        # firing rate
        binned.ix[:, sp_per_s < min_sp_per_s] = np.nan

        if dropna:
            binned = binned.dropna(axis=1)

        return SpikeGroupedDataFrame(binned)

    def clear_refrac(self, threshed, ms=2):
        """Remove spikes from the refractory period of all channels.

        Parameters
        ----------
        threshed : array_like

        ms : float, optional, default 2
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
            # copy so we don't write over the values
            clr = threshed.values.copy()

            # get the number of samples in ms milliseconds
            ms_fs = fs2ms(self.fs, ms)

            # TODO: make sure samples by channels is shape of clr
            # WARNING: you must pass a np.uint8 type array (view or otherwise)
            clear_refrac(clr.view(np.uint8), ms_fs)

            r = SpikeDataFrame(clr, self.meta, index=threshed.index,
                               columns=threshed.columns)
        else:
            r = threshed

        return r

    def fr(self, binned, level='channel', axis=1, return_sem=False):
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

        return_sem : bool, optional
            Whether to return the standard error of the mean along with the
            average firing rate; defaults to ``False``.

        Returns
        -------
        fr, sem : array_like, array_like
            The average firing rate in spikes per `binsize` milliseconds and
            the standard error of the mean of the spike counts for a given
            level.
        """
        group = binned.groupby(axis=axis, level=level)
        gm = group.mean()

        r = gm.mean()

        if return_sem:
            r = r, gm.apply(sem)

        return r

    def xcorr(self, binned, maxlags=None, detrend=span.utils.detrend_mean,
              scale_type='normalize', sortlevel='shank i', dropna=False,
              nan_auto=False, lag_name=r'$\ell$'):
        """Compute the cross correlation of binned data.

        Parameters
        ----------
        binned : array_like
            Data of which to compute the cross-correlation.

        maxlags : int, optional
            Maximum number of lags to return from the cross correlation.
            Defaults to None and computes the full cross correlation.

        detrend : callable, optional
            Callable used to detrend. Defaults to
            :func:`span.utils.detrend_mean`.

        scale_type : str, optional
            Method of scaling. Defaults to ``'normalize'``.

        sortlevel : str, optional
            How to sort the index of the returned cross-correlation(s).
            Defaults to "shank i" so the the xcorrs are ordered by their
            physical ordering.

        dropna : bool, optional
            If ``True`` this will drop all channels whose cross correlation is
            ``NaN``. Cross-correlations will be ``NaN`` if any of the columns
            of `binned` are ``NaN``.

        nan_auto : bool, optional
            If ``True`` then the autocorrelation values will be ``NaN``.
            Defaults to ``False``.

        lag_name : str, optional
            Name to give to the lag index for plotting. Defaults to ``$\ell$``.

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
        xc : DataFrame or Series
            The cross correlation of all the columns of the data.

        See Also
        --------
        span.xcorr.xcorr
            General cross correlation function.

        SpikeDataFrame.bin
            Binning function.

        SpikeDataFrame.clear_refrac
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

        xc.index.name = lag_name

        return xc


# TODO: hack to make it so nans are allowed when creating indices
def _create_xcorr_inds(nchannels):
    """Create a ``MultiIndex`` for `nchannels` channels.

    Parameters
    ----------
    nchannels : int

    Returns
    -------
    xc_inds : MultiIndex
    """
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
