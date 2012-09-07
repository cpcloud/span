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
import operator

import pylab
import scipy.stats

import numpy as np
import pandas as pd


sys.path.append(os.path.expanduser(os.path.join('~', 'code', 'py')))

import span

sys.path.pop(-1)

TsqFields = ('size', 'type', 'name', 'channel', 'sort_code', 'timestamp',
             'file_pointer_location', 'format', 'fs')
TsqNumpyTypes = (np.int32, np.int32, np.uint32, np.uint16, np.uint16,
                 np.float64, np.int64, np.int32, np.float32)

ElectrodeMap = pd.Series(np.array([[1, 3, 2, 6],
                                   [7, 4, 5, 8],
                                   [13, 12, 10, 9],
                                   [14, 16, 11, 15]]).ravel() - 1, name='Electrode Map')

NShanks = 4
NSides = NShanks * 2
ShankMap = pd.Series(np.outer(range(NShanks),
                              np.ones(NShanks)).astype(int).ravel(),
                     name='Shank Map')
MedialLateral = pd.Series(np.asanyarray(('medial', 'lateral'))[np.hstack((np.zeros(NSides),
                                                                          np.ones(NSides)))],
                          name='Side Map')
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
}, name='Event Types')


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
        # return pd.DataFrame(spikes, index=index)


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
    def channel_indices(self):
        inds = pd.DataFrame(self.channel_group.indices)
        inds.columns = cast(inds.columns, int)
        return inds

    @cached_property
    def shank_indices(self):
        inds = pd.DataFrame(self.shank_group.indices)
        inds.columns = cast(inds.columns, int)
        return inds

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
        valsr = vals.transpose(shpsort).reshape((np.prod(shp) // nch, nch))
        return pd.DataFrame(valsr, columns=inds.columns)

    @cached_property
    def channels2(self):
        channels = self.channel_group.apply(self.flatten).T
        channels.columns = cast(channels.columns, int)
        return channels

    def channel(self, i):
        """Get an indivdual channel
        """
        return self.ix[self.channel_indicies[i]].stack()

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


def get_tank_names(path=os.path.expanduser(os.path.join('~', 'xcorr_data'))):
    """Get the names of the tank on the current machine.
    """
    globs = path
    fns = glob.glob(os.path.join(globs, '*'))
    fns = np.array([f for f in fns if os.path.isdir(f)])
    tevs = glob.glob(os.path.join(globs, '**', '*%stev' % os.extsep))
    tevsize = np.asanyarray(list(map(os.path.getsize, tevs)))
    inds = np.argsort(tevsize)
    fns = np.fliplr(fns[inds][np.newaxis]).squeeze().tolist()
    return fns


def profile_spikes(pct_stats=0.05, sortby='time'):
    """Profile the reading of TEV files.
    """
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
        p.strip_dirs().sort_stats(sortby).print_stats(pct_stats)
    return fns


if __name__ == '__main__':
    # fns = profile_spikes()
    fns = get_tank_names()
    fns.sort(key=lambda x: os.path.getsize(x))
    ind = -2
    fn = fns[ind]
    fn_small = os.path.join(fn, os.path.basename(fn))
    t = PandasTank(fn_small)
    sp = t.spikes
    raw = sp.raw
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
