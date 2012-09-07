#!/usr/bin/env python
# encoding: utf-8
"""
Module for reading TDT (Tucker-Davis Technologies) Tank files
"""

import os
import sys
import abc
import glob
import mmap
import types
import re
import threading
import functools
import operator

from scipy.signal import fftconvolve as convolve

import numpy as np
import pandas as pd
import pylab
import scipy
import scipy.stats


try:
    from matplotlib.cbook import Bunch
except ImportError:
    class Bunch(dict):
        def __init__(self, **kwargs):
            super(Bunch, self).__init__(**kwargs)
            self.__dict__ = self


sys.path.append(os.path.expanduser(os.path.join('~', 'code', 'py')))

import span

sys.path.pop(-1)

TsqFields = ('size', 'type', 'name', 'channel', 'sort_code', 'timestamp',
             'file_pointer_location', 'format', 'fs')
TsqNumpyTypes = (np.int32, np.int32, np.uint32, np.uint16, np.uint16,
                 np.float64, np.int64, np.int32, np.float32)
ElectrodeMap = np.array([[1, 3, 2, 6],
                         [7, 4, 5, 8],
                         [13, 12, 10, 9],
                         [14, 16, 11, 15]]).ravel() - 1
ShankMap = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3])
MedialLateral = np.array(['medial', 'lateral'])[[0, 0, 0, 0, 0, 0, 0, 0,
                                                 1, 1, 1, 1, 1, 1, 1, 1]]
Indexer = pd.DataFrame(dict(zip(('channel', 'shank', 'side'),
                                (ElectrodeMap, ShankMap, MedialLateral))))

EventTypes = pd.Series({
    0x0: np.nan,
    0x101: 'strobe_on',
    0x102: 'strobe_off',
    0x201: 'scaler',
    0x8101: 'stream',
    0x8201: 'snip',
    0x8801: 'mark'
})


def name2num(name, base=256):
    """"Convert a string to a number

    Parameters
    ----------
    name : str
    base : int, optional

    Returns
    -------
    ret : int
    """
    return (base ** np.r_[:len(name)]).dot(np.fromiter(map(ord, name), dtype=int))


def nans(size, dtype=float):
    """Create an array of NaNs

    Parameters
    ----------
    size : tuple
    dtype : descriptor, optional

    Returns
    -------
    a : array_like
    """
    a = np.zeros(size, dtype=dtype)
    a.fill(np.nan)
    return a


def nextpow2(n):
    """Return the next power of 2 of a number.

    Parameters
    ----------
    n : array_like

    Returns
    -------
    ret : array_like
    """
    return np.ceil(np.log2(np.absolute(np.asanyarray(n))))


def zeropad(x, s=0):
    """Pad an array, `x`, with `s` zeros.

    Parameters
    ----------
    x : array_like
    s : int

    Returns
    -------
    ret : `x` padded with `s` zeros.
    """
    assert isinstance(s, (int, long)), 's must be an integer'
    assert s >= 0, 's cannot be negative'
    if not s:
        return x
    return np.lib.pad(x, s, mode='constant', constant_values=(0,))


def pad_larger(x, y):
    """Pad the larger of two arrays and the return the arrays and the size of
    the larger array.

    TODO: Generalize this function to n arguments.

    Parameters
    ----------
    x, y : array_like

    Returns
    -------
    x, y : array_like
    lsize : int
        The size of the larger of `x` and `y`.
    """
    xsize, ysize = x.size, y.size
    lsize = max(xsize, ysize)
    if xsize != ysize:
        size_diff = lsize - min(xsize, ysize)

        if xsize > ysize:
            y = zeropad(y, size_diff)
        else:
            x = zeropad(x, size_diff)

    return x, y, lsize


def iscomplex(x):
    """Test whether `x` is complex.

    Parameters
    ----------
    x : array_like

    Returns
    -------
    r : bool
    """
    return np.issubdtype(x.dtype, np.complex)


def get_fft_funcs(*arrays):
    """Get the correct fft functions for the input type.

    Parameters
    ----------
    arrays : tuple

    Returns
    -------
    r : tuple of callable, callable
    """
    if any(map(iscomplex, map(np.asanyarray, arrays))):
        r = np.fft.ifft, np.fft.fft
    else:
        r = np.fft.irfft, np.fft.rfft
    return r


def acorr(x, n):
    """Compute the autocorrelation of `x`

    Parameters
    ----------
    x : array_like
        Input array
    n : int
        Number of fft points

    Returns
    -------
    r : array_like
        The autocorrelation of `x`.
    """
    x = np.asanyarray(x)
    ifft, fft = get_fft_funcs(x)
    return ifft(np.absolute(fft(x, n)) ** 2.0, n)


def correlate(x, y, n):
    """Compute the cross correlation of `x` and `y`

    Parameters
    ----------
    x, y : array_like
    n : int

    Returns
    -------
    c : array_like
        Cross correlation of `x` and `y`
    """
    ifft, fft = get_fft_funcs(x, y)
    return ifft(fft(x, n) * fft(y, n).conj(), n)


def matrixcorrelate(x):
    """Cross-correlation of the columns in a matrix
    
    Parameters
    ----------
    x : array_like
        The matrix from which to compute the cross correlations of each column
        with the others

    Returns
    -------
    c : array_like
        The 2 * maxlags - 1 x x.shape[1] ** 2 matrix of cross-correlations
    """
    raise NotImplementedError


def xcorr(x, y=None, maxlags=None, detrend=pylab.detrend_none, normalize=False,
          unbiased=False):
    """Compute the cross correlation of `x` and `y`.

    This function computes the cross correlation of `x` and `y`. It uses the
    equivalence of the cross correlation with the negative convolution computed
    using a FFT to achieve must faster cross correlation than is possible with
    the signal processing definition.

    By default it computes the raw cross-/autocorrelation.

    Note that it is not necessary for `x` and `y` to be the same size.

    Parameters
    ----------
    x : array_like
    y : array_like, optional
        If not given or is equal to `x`, the autocorrelation is computed.
    maxlags : int, optional
        The highest lag at which to compute the cross correlation.
    detrend : callable, optional
        A callable to detrend the data.
    normalize : bool, optional
    unbiased : bool, optional

    Returns
    -------
    c : pd.Series
    """
    if y is None or np.array_equal(x, y):
        # faster than the more general version
        x = detrend(np.asanyarray(x))
        lsize = x.size
        ctmp = acorr(x, int(2 ** nextpow2(2 * lsize - 1)))
    else:
        # pad the smaller of x and y with zeros
        x, y, lsize = pad_larger(*map(detrend, map(np.asanyarray, (x, y))))

        # compute the xcorr using fft convolution
        ctmp = correlate(x, y, int(2 ** nextpow2(2 * lsize - 1)))

    # no lags are given so use the entire xcorr
    if maxlags is None:
        maxlags = lsize

    lags = np.r_[1 - maxlags:maxlags]

    # make sure the full xcorr is given (acorr is symmetric around 0)
    c = ctmp[lags]

    # normalize by the number of observations seen at each lag
    if unbiased:
        assert not normalize, 'cannot have normalization and unbiased'
        
    if normalize:
        assert not unbiased, 'cannot have normalization and unbiased'
        
    mlags = (lsize - np.absolute(lags)) if unbiased else 1.0

    # normalize by the standard deviation of x and y
    if normalize:
        # ~ an order of mag faster std if already mean centered
        if detrend == pylab.detrend_mean:
            stds = x.dot(x) if y is None else np.sqrt(x.dot(x) * y.dot(y))
        else:
            stds = x.var()
            if y is not None:
                stds = np.sqrt(stds * y.var())
    else:
        stds = 1.0

    c /= stds * mlags
    return pd.Series(c, index=lags)


def remove_legend(ax=None):
    """Remove legend for ax or the current axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    """
    if ax is None:
        ax = pylab.gca()
    ax.legend_ = None


def thunkify(f):
    """Perform `f` using a threaded thunk.

    Parameters
    ----------
    f : callable
    """
    @functools.wraps(f)
    def thunked(*args, **kwargs):
        """The thunked version of `f`

        Parameters
        ----------
        args : tuple, optional
        kwargs : dict, optional

        Returns
        -------
        thunk : callable
        """
        wait_event = threading.Event()
        result = [None]
        exc = [False, None]
        def worker():
            """The worker thread with which to run `f`."""
            try:
                result[0] = f(*args, **kwargs)
            except Exception as e:
                exc[0], exc[1] = True, sys.exc_info()
            finally:
                wait_event.set()
        def thunk():
            """The actual thunk.

            Returns
            -------
            res : type(f(*args, **kwargs))
            """
            wait_event.wait()
            if exc[0]:
                raise exc[1][0](exc[1][1]).with_traceback(exc[1][2])
            return result[0]
        threading.Thread(target=worker).start()
        return thunk
    return thunked


def repeatfunc(func, times=None, *args):
    """
    """
    if times is None:
        return itertools.starmap(func, itertools.repeat(args))
    return itertools.starmap(func, itertools.repeat(args, times))


def pairwise(iterable):
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


def cached_property(f):
    """returns a cached property that is calculated by function `f`

    Parameters
    ----------
    f : callable

    Returns
    -------
    getter : callable
    """
    @property
    @functools.wraps(f)
    def getter(self):
        try:
            x = self.__property_cache[f]
        except AttributeError:
            self.__property_cache = {}
            x = self.__property_cache[f] = f(self)
        except KeyError:
            x = self.__property_cache[f] = f(self)
        return x
    return getter


class TdtTankBase(object, metaclass=abc.ABCMeta):
    fields = TsqFields
    np_types = TsqNumpyTypes
    date_re = re.compile(r'.*(\d{6}).*')
    site_re = re.compile(r'.*s(?:ite)?(?:\s|_)*(\d+)')
    header_ext = 'tsq'
    raw_ext = 'tev'
    tsq_dtype = np.dtype(list(zip(TsqFields, TsqNumpyTypes)))
    age_re = re.compile(r'.*[pP](\d+).*')

    def __init__(self, tankname):
        super(TdtTankBase, self).__init__()
        basename = os.path.basename(tankname)

        self.tankname = tankname
        self.animal_age = int(self.age_re.match(basename).group(1))

        try:
            date = self.date_re.match(basename).group(1)
        except AttributeError:
            now = pd.datetime.now()
            month, day, year = now.month, now.day, now.year
        else:
            datetmp = os.sep.join(i + j for i, j in zip(date[::2],
                                                         date[1::2])).split(os.sep)
            month, day, year = map(int, datetmp)

        self.__date = pd.datetime(year=year + 2000, month=month, day=day).date()
        self.site = int(self.site_re.match(basename).group(1))

    @property
    def date(self): return str(self.__date)

    @abc.abstractmethod
    def _read_tev(self, event_name):
        pass

    @thunkify
    def _read_tsq(self):
        b = pd.DataFrame(np.fromfile(self.tankname + os.extsep + self.header_ext,
                                     dtype=self.tsq_dtype))
        b.channel = b.channel.astype(float) - 1
        b.channel[b.channel == -1] = np.nan
        shank = Indexer.shank[b.channel].reset_index(drop=True)
        side = Indexer.side[b.channel].reset_index(drop=True)
        return b.join(shank).join(side)

    @cached_property
    def nchans(self): return self.tsq.channel.max() + 1

    @cached_property
    def tsq(self): return self._read_tsq()()

    @cached_property
    def spikes(self): return self._read_tev('Spik')()

    @cached_property
    def spike_fs(self): return self.fs.max()


class PandasTank(TdtTankBase):
    def __init__(self, tankname):
        super(PandasTank, self).__init__(tankname)

    @thunkify
    def _read_tev(self, event_name):
        """Read a TDT Tank tev files."""
        name = name2num(event_name)
        row = name == self.tsq.name
        table = ((np.float32, 1, np.float32),
                 (np.int32, 1, np.int32),
                 (np.int16, 2, np.int16),
                 (np.int8, 4, np.int8))
        first_row = (row == 1).argmax()
        fmt = self.tsq.format[first_row]
        chans = self.tsq.channel[row]
        fp_loc = self.tsq.file_pointer_location[row]
        nsamples = (self.tsq.size[first_row] - 10) * table[fmt][1]
        dtype = np.dtype(table[fmt][2]).type
        spikes = np.empty((fp_loc.size, nsamples), dtype=dtype)
        tev_name = self.tankname + os.extsep + self.raw_ext
        with open(tev_name, 'rb') as tev:
            with mmap.mmap(tev.fileno(), 0, access=mmap.ACCESS_READ) as tev:
                for i, offset in enumerate(fp_loc):
                    spikes[i] = np.frombuffer(tev, dtype, nsamples, offset)
        shanks, side = self.tsq.shank[row], self.tsq.side[row]
        index = pd.MultiIndex.from_arrays((shanks, chans, side))
        return SpikeDataFrame(spikes, meta=self.tsq, index=index, dtype=dtype)


class SpikeDataFrameAbstractBase(pd.DataFrame, metaclass=abc.ABCMeta):
    def __init__(self, spikes, meta=None, *args, **kwargs):
        """ """
        super(SpikeDataFrameAbstractBase, self).__init__(spikes, *args, **kwargs)
        self.meta = spikes.meta if meta is None else meta

    @abc.abstractproperty
    def fs(self):
        """ """
        pass


class SpikeDataFrameBase(SpikeDataFrameAbstractBase):
    """ """
    def __init__(self, *args, **kwargs):
        super(SpikeDataFrameBase, self).__init__(*args, **kwargs)

    @cached_property
    def channels(self):
        channels = self.channel_group.apply(self.flatten).T
        channels.columns = cast(channels.columns, np.int64)
        return channels

    def channel(self, i): return self.channels[i]

    @property
    def raw(self): return self.channels.values

    def iterchannels(self):
        for channel in self.channels.iteritems(): yield channel

    __iter__ = iterchannels

    @staticmethod
    def flatten(data):
        """Flatten a SpikeDataFrame
        """
        try:
            # FIX: `stack` method is potentially very fragile
            return data.stack().reset_index(drop=True)
        except MemoryError:
            raise MemoryError('out of memory while trying to flatten')

    @cached_property
    def channel_group(self):
        """ """
        return self.groupby(level=self.meta.channel.name)

    @property
    def shank_group(self):
        """ """
        return self.groupby(level=self.meta.shank.name)

    @property
    def side_group(self):
        """ """
        return self.groupby(level=self.meta.side.name)

    @cached_property
    def fs(self): return self.meta.fs.unique().max()

    def mean(self): return self.summary('mean')
    def var(self): return self.channels.var()
    def std(self): return self.channels.std()
    def mad(self): return self.channels.mad()
    def sem(self): return pd.Series(scipy.stats.sem(self.raw, axis=0))
    def median(self): return self.channels.median()
    def sum(self): return self.summary('sum')
    def max(self): return self.summary('max')
    def min(self): return self.summary('min')

    @cached_property
    def nchans(self): return int(self.meta.channel.max() + 1)

    @staticmethod
    def bin(data, bins):
        nchannels = data.columns.size
        counts = pd.DataFrame(np.empty((bins.size - 1, nchannels)))
        zbins = list(zip(bins[:-1], bins[1:]))
        for column, dcolumn in data.iterkv():
            counts[column] = pd.Series([dcolumn.ix[bi:bj].sum()
                                        for bi, bj in zbins], name=column)
        return counts

    def summary(self, func):
        # check to make sure that `func` is a string or function
        func_is_valid = any(map(isinstance, (func, func),
                                      (str, types.FunctionType)))
        assert func_is_valid, ("'func' must be a string or function: "
                               "type(func) == {0}".format(type(func)))

        # if `func` is a string
        if hasattr(self.channel_group, func):
            getter = operator.attrgetter(func)
            chan_func = getter(self.channel_group)
            chan_func_t = getter(chan_func().T)
            return chan_func_t()

        # else if it's a function and has the attribute `__name__`
        elif hasattr(func, '__name__') and \
                hasattr(self.channel_group, func.__name__):
            return self.summary(func.__name__)

        # else if it's just a regular ole' function
        else:
            f = lambda x: func(self.flatten(x))

        # apply the function to the channel group
        return self.channel_group.apply(f)


def cast(a, to_type, *args, **kwargs):
    assert hasattr(a, 'astype'), 'invalid object for casting'

    if hasattr(a, 'dtype'):
        if a.dtype == to_type:
            return a
    try:
        return a.astype(to_type, *args, **kwargs)
    except TypeError:
        try:
            return a.astype(to_type, copy=False)
        except TypeError:
            return a.astype(to_type)


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
        bin_samples = span.utils.cast(np.floor(binsize * self.fs / conv), int)
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
               conv=1e3, detrend=pylab.detrend_none, unbiased=True,
               normalize=False):
        if self.binned is None:
            self.binned = self.bin(threshes=threshes, ms=ms, binsize=binsize,
                                   conv=conv)
        return xcorr(self.binned[i], self.binned[j], maxlags=maxlags,
                     normalize=normalize, unbiased=unbiased, detrend=detrend)
        

    def xcorr(self, threshes, ms=2.0, binsize=1e3, conv=1e3, maxlags=100,
              plot=False, figsize=(40, 25), dpi=80, titlesize=4, labelsize=3,
              sharex=True, sharey=True):
        if self.xcorrs is None:
            if self.binned is None:
                self.binned = self.bin(threshes, ms=ms, binsize=binsize,
                                       conv=conv)
            nchannels = binned.columns.size
            ncorrs = nchannels ** 2
            xctmp = np.empty((ncorrs, 2 * maxlags - 1))

            left, right, lshank, rshank = [pd.Series(np.empty(ncorrs, dtype=np.int64))
                                           for _ in range(4)]
            left.name, right.name, lshank.name, rshank.name = ('Left', 'Right',
                                                               'Left Shank',
                                                               'Right Shank')
            k = 0
            for i, chi in binned.iterkv():
                shi = ShankMap[i]
                for j, chj in binned.iterkv():
                    shj = ShankMap[j]
                    left[k], right[k], lshank[k], rshank[k] = i, j, shi, shj
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
        """ """
        if dtype is None:
            assert hasattr(data, 'dtype'), 'data has no "dtype" attribute'
            dtype = data.dtype
        return self._constructor(data, self.meta, index=self.index,
                                 columns=self.columns, dtype=dtype, copy=False)

    @property
    def _constructor(self):
        def construct(*args, **kwargs):
            args = list(args)
            if len(args) == 2:
                meta = args.pop(1)
            if 'meta' not in kwargs or kwargs['meta'] is None or meta is None:
                kwargs['meta'] = self.meta
            return type(self)(*args, **kwargs)
        return construct


class SparseSpikeDataFrame(SpikeDataFrameBase, pd.SparseDataFrame):
    def __init__(self, spikes, meta=None, *args, **kwargs):
        """ """
        super(SparseSpikeDataFrame, self).__init__(spikes, meta=meta, *args,
                                                   **kwargs)


def dirsize(d):
    """ """
    s = os.path.getsize(d)
    for item in glob.glob(os.path.join(d, '*')):
        path = os.path.join(d, item)
        if os.path.isfile(path):
            s += os.path.getsize(path)
        elif os.path.isdir(path):
            s += dirsize(path)
    return s


def get_tank_names(path=os.path.expanduser(os.path.join('~', 'xcorr_data'))):
    """ """
    globs = path
    fns = glob.glob(os.path.join(globs, '*'))
    fns = np.array([f for f in fns if os.path.isdir(f)])
    tevs = glob.glob(os.path.join(globs, '**', '*%stev' % os.extsep))
    tevsize = np.asanyarray(list(map(os.path.getsize, tevs)))
    inds = np.argsort(tevsize)
    fns = np.fliplr(fns[inds][np.newaxis]).squeeze().tolist()
    return fns


def profile_spikes():
    import cProfile as profile
    import pstats
    import tempfile

    fns = get_tank_names()

    # sort by size
    fns.sort(key=lambda x: os.path.getsize(x))

    for f in fns[-1:]:
        fn = os.path.join(f, os.path.basename(f))
        with tempfile.NamedTemporaryFile(mode='w+') as stats_file:
            stats_fn = stats_file.name
            profile.run('sp = PandasTank("%s").spikes' % fn, stats_fn)
            p = pstats.Stats(stats_fn)
        p.strip_dirs().sort_stats('time').print_stats(0.05)
    return fns


def side_sums(n=None):
    """ """
    fns = get_tank_names()
    if n is None:
        n = len(fns)
    fig = pylab.figure()
    nplot = int(np.ceil(np.sqrt(n)))
    kind = 'bar'
    fmt = r'$t=%.4f,\,\,p=%.4f$, age == P%i'
    for i, fn in enumerate(fns[:n], start=1):
        p = os.path.join(fn, os.path.basename(fn))
        ax = fig.add_subplot(nplot, nplot, i)
        try:
            pt = PandasTank(p)
            s = pt.spikes.cleared(3e-5).side_group.sum()
            v = s.values
            t, p = ttest_ind(v[0], v[1])
            s.T.sum().plot(ax=ax, kind=kind)
            ax.set_title(fmt % (t, p, pt.animal_age), fontsize='tiny')
        except ValueError:
            pass
    fig.suptitle('Side-wise Spike Counts', fontsize='large')
    fig.tight_layout()
    pylab.show()


if __name__ == '__main__':
    # fns = profile_spikes()
    fns = get_tank_names()
    fns.sort(key=lambda x: os.path.getsize(x))
    ind = -2
    fn = fns[ind]
    fn_small = os.path.join(fn, os.path.basename(fn))
    t = PandasTank(fn_small)
    sp = t.spikes
    # thr = spikes.threshold(3e-5).astype(float)
    # thr.values[thr.values == 0] = np.nan

    # xc = spikes.xcorr(3e-5, plot=True, sharey=True)
    # binned = spikes.binned(3e-5)
    # b0 = binned[2]
    # b0mean = b0.mean()
    # b0cent = b0 - b0mean

    # denom = b0.var()
    # npcorr = np.correlate(b0cent.values, b0cent.values, 'full') / denom
    # mycorr = xcorr(b0)

    # pylab.subplot(211)
    # pylab.plot(npcorr)

    # pylab.subplot(212)
    # pylab.vlines(mycorr.index, 0, mycorr.values)
    # pylab.show()
