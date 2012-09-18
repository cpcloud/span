"""
"""

from __future__ import division
from future_builtins import map, zip

import abc
import itertools

import numpy as np
import scipy.stats
import pandas as pd

try:
    from pylab import subplots
except RuntimeError:
    subplots = None

from span.tdt.decorate import cached_property
from span.tdt.xcorr import xcorr

from span.utils import (summary, group_indices, flatten, cast, ndtuples,
                        detrend_mean, clear_refrac_out, remove_legend)
from span.tdt.spikeglobals import Indexer


class SpikeDataFrameAbstractBase(pd.DataFrame):
    """Abstract base class for spike data frames."""

    __metaclass__ = abc.ABCMeta

    def __init__(self, spikes, meta=None, *args, **kwargs):
        """Constructor.

        Parameters
        ----------
        spikes : array_like
        meta : array_like, optional
        args, kwargs : args, kwargs to pandas.DataFrame
        """
        super(SpikeDataFrameAbstractBase, self).__init__(spikes, *args, **kwargs)
        self.meta = spikes.meta if meta is None else meta
        self.fs = self.meta.fs.max()
        self.nchans = int(self.meta.channel.max() + 1)

    @abc.abstractproperty
    def raw(self):
        """Retrieve the underlying raw NumPy array."""
        pass

    @abc.abstractproperty
    def channels(self):
        """Retrieve the data organized as samples by channels."""
        pass


class SpikeDataFrameBase(SpikeDataFrameAbstractBase):
    """Base class implementing basic spike data set properties and methods."""
    def __init__(self, *args, **kwargs):
        super(SpikeDataFrameBase, self).__init__(*args, **kwargs)

    @cached_property
    def channels(self):
        # get the channel indices
        inds = self.channel_indices

        # get the 3D array of raw values
        vals = self.values[inds.values]

        # number of channels
        nch = inds.columns.size

        # get indices of the sorted (descending) dimensions of vals
        shpsort = np.asanyarray(vals.shape).argsort()[::-1]

        # transpose vals to make a reshape into a samples x channels array
        valsr = vals.transpose(shpsort).reshape(vals.size // nch, nch)
        return pd.DataFrame(valsr, columns=inds.columns)

    @property
    def shanks(self):
        inds = self.shank_indices
        vals = self.values[inds.values]
        nsh = inds.columns.size
        shpsort = np.asanyarray(vals.shape).argsort()[::-1]
        valsr = vals.transpose(shpsort).reshape(vals.size // nsh, nsh)
        return pd.DataFrame(valsr, columns=inds.columns)

    @cached_property
    def channels_slow(self):
        channels = self.channel_group.apply(flatten).T
        channels.columns = cast(channels.columns, int)
        return channels

    @property
    def channel_indices(self): return group_indices(self.channel_group)

    @property
    def shank_indices(self): return group_indices(self.shank_group)

    @property
    def side_indices(self): return group_indices(self.side_group)

    @property
    def channel_group(self): return self.groupby(level=self.meta.channel.name)

    @property
    def shank_group(self): return self.groupby(level=self.meta.shank.name)

    @property
    def side_group(self): return self.groupby(level=self.meta.side.name)

    @property
    def raw(self): return self.channels.values

    def channel(self, i): return self.channels[i]

    def mean(self, axis=0, skipna=True, level=None):
        return self.channel_summary('mean', axis=axis, skipna=skipna,
                                    level=level)

    def var(self, axis=0, skipna=True, level=None, ddof=1):
        return self.channels.var(axis=axis, skipna=skipna, level=level, ddof=ddof)

    def std(self, axis=0, skipna=True, level=None, ddof=1):
        return self.channels.std(axis=axis, skipna=skipna, level=level, ddof=ddof)

    def mad(self, axis=0, skipna=True, level=None):
        return self.channels.mad(axis=axis, skipna=skipna, level=level)

    def sem(self, axis=0, skipna=True, level=None):
        return pd.Series(scipy.stats.sem(np.ma.masked_invalid(self.raw, np.nan),
                                         axis=axis))

    def median(self, axis=0, skipna=True, level=None):
        return self.channels.median(axis=axis, skipna=skipna, level=level)

    def sum(self, axis=0, skipna=True, level=None, numeric_only=None):
        return self.channel_summary('sum', axis=axis, skipna=skipna, level=level)

    def max(self, axis=0, skipna=True, level=None):
        return self.channel_summary('max', axis=axis, skipna=skipna, level=level)

    def min(self, axis=0, skipna=True, level=None):
        return self.channel_summary('min', axis=axis, skipna=skipna, level=level)

    def channel_summary(self, func, axis, skipna, level):
        return summary(self.channel_group, func, axis, skipna, level)


class SpikeDataFrame(SpikeDataFrameBase):
    """ """
    def __init__(self, spikes, meta=None, *args, **kwargs):
        super(SpikeDataFrame, self).__init__(spikes, meta=meta, *args, **kwargs)

    def __lt__(self, other): return self.lt(other)
    def __le__(self, other): return self.le(other)
    def __gt__(self, other): return self.gt(other)
    def __ge__(self, other): return self.ge(other)
    def __ne__(self, other): return self.ne(other)
    def __eq__(self, other): return self.eq(other)

    threshold = __gt__

    def bin(self, threshes, ms=2.0, binsize=1000, conv=1e3):
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
        df : DataFrame
        """
        cleared = self.cleared(threshes, ms=ms).channels
        max_sample = self.channels.index[-1]
        bin_samples = cast(np.floor(binsize * self.fs / conv), int)
        bins = np.r_[:max_sample:bin_samples]
        v = cleared.values[list(map(xrange, bins[:-1], bins[1:]))]
        axis, = np.where(np.asanyarray(v.shape) == bin_samples)
        return pd.DataFrame(v.sum(axis))

    def refrac_window(self, ms=2.0, conv=1e3):
        """Compute the refractory window in samples.

        Parameters
        ----------
        ms : float, optional
        conv : float, optional

        Returns
        -------
        win : int
        """
        return cast(np.floor(ms / conv * self.fs), int)

    def cleared(self, threshes, ms=2.0):
        """Remove spikes from the refractory window of a channel."""
        clr = self.threshold(threshes)

        # TODO: fragile indexing here
        if clr.shape[0] < clr.shape[1]:
            clr = clr.T
        clear_refrac_out(clr.values, self.refrac_window(ms))
        return clr

    def xcorr(self, threshes, ms=2.0, binsize=1000, conv=1000, maxlags=100,
              detrend=detrend_mean, scale_type='normalize'):
        """

        Parameters
        ----------
        threshes : array_like
        ms : float, optional
        binsize : float, optional
        maxlags : int, optional
        conv : float, optional
        detrend : callable, optional
        scale_type : str

        Returns
        -------
        xc : DataFrame
            The cross correlation of all the channels in the data
        """

        binned = self.bin(threshes, ms=ms, binsize=binsize, conv=conv)
        nchannels = binned.columns.size
        left, right = ndtuples(nchannels, nchannels).T
        left, right = map(pd.Series, (left, right))
        left.name, right.name = 'ch i', 'ch j'
        sorted_indexer = Indexer.sort('channel').reset_index(drop=True)
        lshank, rshank = sorted_indexer.shank[left], sorted_indexer.shank[right]
        lshank.name, rshank.name = 'sh i', 'sh j'
        index = pd.MultiIndex.from_arrays((left, right, lshank, rshank))
        xc = xcorr(binned, maxlags=maxlags, detrend=detrend,
                   scale_type=scale_type).T
        xc.index = index
        return xc

    @classmethod
    def plot_xcorr(cls, xc, figsize=(40, 25), dpi=100, titlesize=4, labelsize=3,
                   sharex=True, sharey=True):
        # get the channel indexer
        elec_map = Indexer.channel

        # number of channels
        nchannels = elec_map.size

        # the channel index labels
        left, right, _, _ = xc.index.labels

        # indices of a lower triangular nchannels by nchannels array
        lower_inds = np.tril_indices(nchannels)

        # flatted and linearly indexed
        flat_lower_inds = np.ravel_multi_index(np.vstack(lower_inds),
                                               (nchannels, nchannels))

        # make a list of strings for titles
        title_strings = cast(np.vstack((left + 1, right + 1)), str).T.tolist()
        title_strings = np.asanyarray(list(map(' vs. '.join, title_strings)))

        # get only the ones we want
        title_strings = title_strings[flat_lower_inds]

        # create the subplots with linked axes
        fig, axs = subplots(nchannels, nchannels, sharex=sharex, sharey=sharey,
                            figsize=figsize, dpi=dpi)

        # get the axes objects that we want to show
        axs_to_show = axs.flat[flat_lower_inds]

        # set the title on the axes objects that we want to see
        titler = lambda ax, s, fs: ax.set_title(s, fontsize=fs)
        sizes = itertools.repeat(titlesize, axs_to_show.size)
        list(map(titler, axs_to_show.flat, title_strings, sizes))

        # hide the ones we don't want
        upper_inds = np.triu_indices(nchannels, 1)
        flat_upper_inds = np.ravel_multi_index(np.vstack(upper_inds),
                                               (nchannels, nchannels))
        axs_to_hide = axs.flat[flat_upper_inds]
        list(map(lambda ax: map(lambda tax: tax.set_visible(False), (ax.xaxis, ax.yaxis)),
                 axs_to_hide))
        list(map(lambda ax: ax.set_frame_on(False), axs_to_hide))
        list(map(remove_legend, axs.flat))

        min_value = xc.min().min()
        for indi, i in enumerate(elec_map):
            for indj, j in enumerate(elec_map):
                ax = axs[indi, indj]
                if indi >= indj:
                    ax.tick_params(labelsize=labelsize, left=True,
                                   right=False, top=False, bottom=True,
                                   direction='out')
                    xcij = xc.ix[i, j].T
                    ax.vlines(xcij.index, min_value, xcij)
        fig.tight_layout()
        return fig, axs

    def astype(self, dtype):
        """Return a new instance of SpikeDataFrame with a (possibly) different
        dtype.

        Parameters
        ----------
        dtype : numpy.dtype

        Returns
        -------
        obj : SpikeDataFrame
            A new SpikeDataFrame object.
        """
        return self._constructor(self._data, self.meta, index=self.index,
                                 columns=self.columns, dtype=dtype, copy=False)

    def make_new(self, data, dtype=None):
        """Make a new instance of the current object.

        Parameters
        ----------
        data : array_like
        dtype : numpy.dtype, optional

        Raises
        ------
        AssertionError

        Returns
        -------
        obj : SpikeDataFrame
            A new SpikeDataFrame object.
        """
        if dtype is None:
            assert hasattr(data, 'dtype'), 'data has no "dtype" attribute'
            dtype = data.dtype
        return self._constructor(data, self.meta, index=self.index,
                                 columns=self.columns, dtype=dtype, copy=False)

    @property
    def _constructor(self):
        """Return a constructor function.

        Returns
        -------
        construct : callable
            A function to construct a new instance of SpikeDataFrame.
        """
        def construct(*args, **kwargs):
            """Construct a new instance of the type of the current object.

            Parameters
            ----------
            args : tuple
            kwargs : dict

            Returns
            -------
            obj : type(self)
                A new object of type type(self).
            """
            args = list(args)
            if len(args) == 2:
                meta = args.pop(1)
            if 'meta' not in kwargs or kwargs['meta'] is None or meta is None:
                kwargs['meta'] = self.meta
            return SpikeDataFrame(*args, **kwargs)
        return construct
