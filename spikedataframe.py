"""
"""

import pandas as pd
from .xcorr import xcorr


def group_indices(group, dtype=int):
    """
    """
    inds = pd.DataFrame(group.indicies)
    inds.columns = cast(inds.columns, dtype)
    return inds


def flatten(data):
    """Flatten a SpikeDataFrame
    """
    try:
        # FIX: `stack` method is potentially very fragile
        return data.stack().reset_index(drop=True)
    except MemoryError:
        raise MemoryError('out of memory while trying to flatten')


# def bin_data(data, bins):
#     """
#     """
#     nchannels = data.columns.size
#     counts = pd.DataFrame(np.empty((bins.size - 1, nchannels)))
#     zbins = list(zip(bins[:-1], bins[1:]))
#     for column, dcolumn in data.iterkv():
#         counts[column] = pd.Series([dcolumn.ix[bi:bj].sum()
#                                     for bi, bj in zbins], name=column)
#     return counts


def summary(func, group):
    """TODO: Make this function less ugly!"""
    # check to make sure that `func` is a string or function
    func_is_valid = any(map(isinstance, (func, func),
                                  (str, types.FunctionType)))
    assert func_is_valid, ("'func' must be a string or function: "
                           "type(func) == {0}".format(type(func)))

    # if `func` is a string
    if hasattr(group, func):
        getter = operator.attrgetter(func)
        chan_func = getter(group)
        chan_func_t = getter(chan_func().T)
        return chan_func_t()

    # else if it's a function and has the attribute `__name__`
    elif hasattr(func, '__name__') and \
            hasattr(group, func.__name__):
        return summary(func.__name__)

    # else if it's just a regular ole' function
    else:
        f = lambda x: func(SpikeDataFrame.flatten(x))

    # apply the function to the channel group
    return group.apply(f)


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
        
        # get indices of the sorted dimensions of vals and reverse so
        # highest is first
        shp = np.asanyarray(vals.shape)
        shpsort = shp.argsort()[::-1]

        # transpose vals to make a reshape into a samples x channels array
        valsr = vals.transpose(shpsort).reshape(np.prod(shp) // nch, nch)
        return pd.DataFrame(valsr, columns=inds.columns)

    @property
    def channels_slow(self):
        channels = self.channel_group.apply(self.flatten).T
        channels.columns = cast(channels.columns, int)
        return channels

    @cached_property
    def fs(self): return self.meta.fs.unique().max()

    @cached_property
    def nchans(self): return int(self.meta.channel.max() + 1)

    @cached_property
    def channel_indices(self): return self.group_indices(self.channel_group)

    @property
    def shank_indices(self): return self.group_indices(self.shank_group)

    @property
    def side_indices(self): return self.group_indices(self.side_group)

    @cached_property
    def channel_group(self): return self.groupby(level=self.meta.channel.name)

    @property
    def shank_group(self): return self.groupby(level=self.meta.shank.name)

    @property
    def side_group(self): return self.groupby(level=self.meta.side.name)

    @property
    def raw(self): return self.channels.values

    def channel(self, i): return self.channels[i]
    def mean(self): return self.summary('mean')
    def var(self): return self.channels.var()
    def std(self): return self.channels.std()
    def mad(self): return self.channels.mad()
    def sem(self): return pd.Series(scipy.stats.sem(self.raw, axis=0))
    def median(self): return self.channels.median()
    def sum(self):return self.summary('sum')
    def max(self): return self.summary('max')
    def min(self): return self.summary('min')


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

    def bin(self, threshes, ms=2.0, binsize=1e3, conv=1e3, raw_out=False):
        cleared = self.cleared(threshes, ms=ms).channels
        max_sample = self.channels.index[-1]
        bin_samples = cast(np.floor(binsize * self.fs / conv), int)
        bins = np.r_[:max_sample:bin_samples]
        
        v = cleared.values[[range(bi, bj) for bi, bj in zip(bins[:-1], bins[1:])]]
        b = pd.DataFrame(v.sum(np.argmax(v.shape)))
        if raw_out:
            return b, v
        return b

    def refrac_window(self, ms=2.0, conv_factor=1e3):
        secs = ms / conv_factor
        return cast(np.floor(secs * self.fs), int)

    def __bool__(self): return self.values.all()
    
    def threshold(self, thresh): return self > thresh

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
            nchannels = binned.columns.size
            ncorrs = nchannels ** 2
            xctmp = np.empty((ncorrs, 2 * maxlags - 1))


            left = pd.Series(np.tile(np.arange(nchannels), nchannels),
                             name='Left')
            right = pd.Series(np.sort(left.values), name='Right')
            lshank, rshank = ShankMap[left], ShankMap[right]
            lshank.name, rshank.name = 'Left Shank', 'Right Shank'
            
            for i, chi in binned.iterkv():
                for j, chj in binned.iterkv():
                    args = chi,
                    if i != j:
                        args += chj,
                    c = xcorr1(*args, maxlags=maxlags)
                    xctmp[k] = c
                    k += 1

            index = pd.MultiIndex.from_arrays((left, right, lshank, rshank))
            self.xcorrs = pd.DataFrame(xctmp, index=index, columns=.index)

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
