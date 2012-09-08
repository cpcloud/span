"""
"""

import sys
import abc

import numpy as np
import pandas as pd
import pylab

import span

from xcorr import xcorr
from decorate import cached_property
from utils import summary, group_indices, flatten, cast


class SpikeDataFrameAbstractBase(pd.DataFrame, metaclass=abc.ABCMeta):
    """Abstract base class for all spike data frames.
    """
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

    @abc.abstractproperty
    def fs(self):
        """Sampling rate

        Returns
        -------
        sr : float
            The sampling rate of the recording.
        """
        pass

    @abc.abstractproperty
    def raw(self):
        pass


class SpikeDataFrameBase(SpikeDataFrameAbstractBase):
    """
    """
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
    
    def fs(self): return self.meta.fs.unique().max()
    def nchans(self): return int(self.meta.channel.max() + 1)

    def channel_indices(self): return group_indices(self.channel_group)
    def shank_indices(self): return group_indices(self.shank_group)
    def side_indices(self): return group_indices(self.side_group)
    
    def channel_group(self): return self.groupby(level=self.meta.channel.name)
    def shank_group(self): return self.groupby(level=self.meta.shank.name)
    def side_group(self): return self.groupby(level=self.meta.side.name)

    def raw(self): return self.channels.values
    
    fs = cached_property(fs)
    nchans = cached_property(nchans)

    channel_indices = cached_property(channel_indices)
    shank_indices = cached_property(shank_indices)
    side_indices = cached_property(side_indices)

    channel_group = property(channel_group)
    shank_group = property(shank_group)
    side_group = property(side_group)

    raw = property(raw)

    def channel(self, i): return self.channels[i]
    def mean(self): return self.channel_summary('mean')
    def var(self): return self.channels.var()
    def std(self): return self.channels.std()
    def mad(self): return self.channels.mad()
    def sem(self): return pd.Series(scipy.stats.sem(self.raw, axis=0))
    def median(self): return self.channels.median()
    def sum(self): return self.channel_summary('sum')
    def max(self): return self.channel_summary('max')
    def min(self): return self.channel_summary('min')
    def channel_summary(self, func): return summary(self.channel_group, func)


class SpikeDataFrame(SpikeDataFrameBase):
    def __init__(self, spikes, meta=None, *args, **kwargs):
        super(SpikeDataFrameBase, self).__init__(spikes, meta=meta, *args,
                                                 **kwargs)
        self.__xcorrs, self.__binned = None, None

    def __lt__(self, other): return self.lt(other)
    def __le__(self, other): return self.le(other)
    def __gt__(self, other): return self.gt(other)
    def __ge__(self, other): return self.ge(other)
    def __ne__(self, other): return self.ne(other)
    def __eq__(self, other): return self.eq(other)
    def __bool__(self): return self.all().all()

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
        df : pd.DataFrame
        """
        cleared = self.cleared(threshes, ms=ms).channels
        max_sample = self.channels.index[-1]
        bin_samples = cast(np.floor(binsize * self.fs / conv), int)
        bins = np.r_[:max_sample:bin_samples]
        zipped_bins = list(zip(bins[:-1], bins[1:]))
        v = cleared.values[[range(bi, bj) for bi, bj in zipped_bins]]
        axis, = np.where(np.asanyarray(v.shape) == bin_samples)
        return pd.DataFrame(v.sum(axis))

    def refrac_window(self, ms=2.0, conv=1e3):
        return cast(np.floor(ms / conv * self.fs), int)

    def cleared(self, threshes, ms=2.0):
        clr = self.threshold(threshes)
        if clr.shape[0] < clr.shape[1]:
            clr = clr.T
        span.clear_refrac.clear_refrac_out(clr.values, self.refrac_window(ms))
        return clr

    @property
    def xcorrs(self): return self.__xcorrs

    @xcorrs.setter
    def xcorrs(self, value):
        self.__xcorrs = value

    @property
    def binned(self): return self.__binned

    @binned.setter
    def binned(self, value): self.__binned = value

    def xcorr1(self, i, j, threshes=3e-5, ms=2.0, binsize=1e3, maxlags=100,
               conv=1e3, detrend=pylab.detrend_none, unbiased=False,
               normalize=False):
        """
        """
        if self.binned is None:
            self.binned = self.bin(threshes=threshes, ms=ms, binsize=binsize,
                                   conv=conv)
        return xcorr(self.binned[i], self.binned[j], maxlags=maxlags,
                     normalize=normalize, unbiased=unbiased, detrend=detrend)

    def xcorr(self, threshes, ms=2.0, binsize=1e3, conv=1e3, maxlags=100,
              detrend=pylab.detrend_none, unbiased=False, normalize=False,
              plot=False, figsize=(40, 25), dpi=80, titlesize=4, labelsize=3,
              sharex=True, sharey=True):
        """
        """
        if self.xcorrs is None:
            if self.binned is None:
                self.binned = self.bin(threshes, ms=ms, binsize=binsize,
                                       conv=conv)
            nchannels = self.binned.columns.size
            ncorrs = nchannels ** 2
            xctmp = np.empty((ncorrs, 2 * maxlags - 1))

            left = pd.Series(np.tile(np.arange(nchannels), nchannels),
                             name='Left')
            right = pd.Series(np.sort(left.values), name='Right')
            lshank, rshank = ShankMap[left], ShankMap[right]
            lshank.name, rshank.name = 'Left Shank', 'Right Shank'
            
            for i, chi in self.binned.iterkv():
                for j, chj in self.binned.iterkv():
                    args = chi,
                    if i != j:
                        args += chj,
                    c = xcorr1(*args, maxlags=maxlags)
                    xctmp[k] = c
                    k += 1

            index = pd.MultiIndex.from_arrays((left, right, lshank, rshank))
            self.xcorrs = pd.DataFrame(xctmp, index=index, columns=c.index)

        xc = self.xcorrs

        if plot:
            elec_map = ElectrodeMap
            nchannels = self.nchans
            fig, axs = pylab.subplots(nchannels, nchannels, sharex=sharex,
                                      sharey=sharey, figsize=figsize, dpi=dpi)
            for indi, i in enumerate(elec_map):
                for indj, j in enumerate(elec_map):
                    ax = axs[indi, indj]
                    if indi >= indj:
                        ax.tick_params(labelsize=labelsize, left=True,
                                       right=False, top=False, bottom=True,
                                       direction='out')
                        xcij = xc.ix[i, j].T
                        ax.vlines(xcij.index, 0, xcij)
                        ax.set_title('%i vs. %i' % (i + 1, j + 1),
                                     fontsize=titlesize)
                        ax.grid()
                        remove_legend(ax=ax)
                    else:
                        ax.set_frame_on(False)
                        for tax in (ax.xaxis, ax.yaxis):
                            tax.set_visible(False)
            fig.tight_layout()
            pylab.show()
        return xc

    def astype(self, dtype):
        """ """
        return self._constructor(self._data, self.meta, index=self.index,
                                 columns=self.columns, dtype=dtype, copy=False)

    def make_new(self, data, dtype=None):
        """Make a new instance of the current object.
        """
        if dtype is None:
            assert hasattr(data, 'dtype'), 'data has no "dtype" attribute'
            dtype = data.dtype
        return self._constructor(data, self.meta, index=self.index,
                                 columns=self.columns, dtype=dtype, copy=False)

    @property
    def _constructor(self):
        """
        """
        def construct(*args, **kwargs):
            """
            """
            args = list(args)
            if len(args) == 2:
                meta = args.pop(1)
            if 'meta' not in kwargs or kwargs['meta'] is None or meta is None:
                kwargs['meta'] = self.meta
            return type(self)(*args, **kwargs)
        return construct
