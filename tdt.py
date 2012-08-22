#!/usr/bin/env python
# encoding: utf-8
"""
"""
import os
import sys
import abc
import mmap
import contextlib
import types
import re
import threading
import traceback
import functools

from itertools import izip, imap

import numpy as np
import pandas as pd
import scipy.stats as stats

import span

try:
    from matplotlib.cbook import Bunch
except ImportError:
    class Bunch(dict):
        def __init__(self, **kwargs):
            super(Bunch, self).__init__(**kwargs)
            self.__dict__ = self

        def __getstate__(self):
            return self

        def __setstate__(self, state):
            self.update(state)
            self.__dict__ = self


TsqFields = ('size', 'type', 'name', 'chan', 'sortcode', 'timestamp', 'fp_loc',
             'format', 'fs')
TsqNumpyTypes = (np.int32, np.int32, np.uint32, np.uint16, np.uint16,
                 np.float64, np.int64, np.int32, np.float32)
ElectrodeMap = np.array([[1, 3, 2, 6],
                         [7, 4, 5, 8],
                         [13, 12, 10, 9],
                         [14, 16, 11, 15]])

ShankMap = np.array([0, 0, 0, 1, 1, 0, 1, 1, 2, 2, 3, 2, 2, 3, 3, 3])


def name2num(name, base=256):
    """"Convert a string to a number"""
    return (base ** np.arange(len(name))).dot(tuple(imap(ord, name)))


def nans(size, dtype=np.float64):
    """Create an array of NaNs"""
    a = np.zeros(size, dtype=dtype)
    a.fill(np.nan)
    return a


def thunkify(f):
    """ """
    @functools.wraps(f)
    def thunked(*args, **kwargs):
        """ """
        wait_event = threading.Event()
        result = [None]
        exc = [False, None]
        def worker():
            """ """
            try:
                result[0] = f(*args, **kwargs)
            except Exception as e:
                exc[0], exc[1] = True, sys.exc_info()
                print 'Lazy thunk has thrown any exception:\n%s' % traceback.format_exc()
            finally:
                wait_event.set()
        def thunk():
            """ """
            wait_event.wait()
            if exc[0]:
                raise exc[1][0], exc[1][1], exc[1][2]
            return result[0]
        threading.Thread(target=worker).start()
        return thunk
    return thunked


def cached_property(f):
    """returns a cached property that is calculated by function f"""
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


class TdtTankBase(object):
    __metaclass__ = abc.ABCMeta
    fields = TsqFields
    np_types = TsqNumpyTypes
    date_re = re.compile(r'.*(\d{6}).*')
    header_ext = 'tsq'
    raw_ext = 'tev'
    tsq_dtype = np.dtype(zip(TsqFields, TsqNumpyTypes))

    def __init__(self, tankname):
        super(TdtTankBase, self).__init__()
        self.tankname = tankname

        try:
            date = self.date_re.match(os.path.basename(self.tankname)).group(1)
        except AttributeError:
            now = pd.datetime.now()
            month, day, year = now.month, now.day, now.year
        else:
            datetmp = os.sep.join(i + j for i, j in izip(date[::2],
                                                         date[1::2])).split(os.sep)
            month, day, year = imap(int, datetmp)
        self.date = str(pd.datetime(year, month, day).date())

    @abc.abstractmethod
    def _read_tev(self, event_name):
        pass

    @thunkify
    def _read_tsq(self):
        raw_tsq = np.fromfile(self.tankname + os.extsep + self.header_ext,
                              dtype=self.tsq_dtype)
        b = pd.DataFrame(raw_tsq)
        bchan = b.chan - 1
        try:
            bchan = bchan.astype(np.float64, copy=False)
        except TypeError:
            bchan = bchan.astype(np.float64)
        bchan[bchan == -1] = np.nan
        shank_series = pd.Series(ShankMap, name='shank')
        return b.join(shank_series[bchan].reset_index(drop=True))

    @cached_property
    def nchans(self):
        return self.tsq.chan.max()

    @cached_property
    def tsq(self):
        return self._read_tsq()()

    @cached_property
    def spikes(self):
        return self._read_tev('Spik')()

    @cached_property
    def lfps(self):
        return self._read_tev('LFPs')()

    @cached_property
    def fs(self):
        return self.tsq.fs.unique()

    @property
    def spike_fs(self):
        return self.fs.max()

    @property
    def lfp_fs(self):
        return self.fs.min()

    @cached_property
    def times(self):
        return np.arange(self.channel(0).values.size) * 1e6 / self.spike_fs


class PandasTank(TdtTankBase):
    def __init__(self, tankname):
        super(PandasTank, self).__init__(tankname)

    @thunkify
    def _read_tev(self, event_name):
        """ """
        name = name2num(event_name)
        row = name == self.tsq.name
        table = ((np.float32, 1, np.float32),
                 (np.int32, 1, np.int32),
                 (np.int16, 2, np.int16),
                 (np.int8, 4, np.int8))
        first_row = np.argmax(row == 1)
        fmt = self.tsq.format[first_row]
        chans = self.tsq.chan[row] - 1
        fp_loc = self.tsq.fp_loc[row]
        nsamples = (self.tsq.size[first_row] - 10) * table[fmt][1]
        dt = np.dtype(table[fmt][2])
        dtype = dt.type
        spikes = np.empty((fp_loc.size, nsamples), dtype=dtype)
        tev_name = '%s%s%s' % (self.tankname, os.extsep, self.raw_ext) 
        with open(tev_name, 'rb') as tev:
            with contextlib.closing(mmap.mmap(tev.fileno(), 0,
                                              access=mmap.ACCESS_READ)) as tev:
                for i, offset in enumerate(fp_loc):
                    spikes[i] = np.frombuffer(tev, dtype, nsamples, offset)
        shanks = self.tsq.shank[row]
        index = pd.MultiIndex.from_arrays((shanks, chans))
        return pd.DataFrame(spikes, index=index, dtype=dtype)

    @cached_property
    def channels(self): return self.changroup.apply(self.flatten).T

    @cached_property
    def shanks(self): return NotImplemented

    @property
    def values(self): return self.channels.values
    def iterchannels(self):
        for _, v in self.channels.iteritems(): yield v

    __iter__ = iterchannels

    def mean(self): return self.summary('mean')
    def var(self): return self.channels.var()
    def std(self): return self.channels.std()
    def sem(self): return pd.Series(stats.sem(self.values, axis=0))
    def median(self): return self.summary('median')
    def sum(self): return self.summary('sum')
    def max(self): return self.summary('max')
    def min(self): return self.summary('min')
    
    def bin_data(self, binsize, spike_times):
        if binsize != 1:
            return span.thresh.thresh.bin_data(spike_times,
                                               np.r_[:spike_times.max():binsize])
        else:
            return self.cleared
    
    def channel(self, i): return self.flatten(self.changroup.get_group(i))
    def threshold(self, thresh): return self.spikes > thresh

    def cleared(self, threshes, ms):
        threshed = self.threshold(threshes)
        window = span.thresh.spike_window(ms, self.spike_fs)
        tv = threshed.values
        try:
            tv = tv.astype(np.uint8, copy=False)
        except TypeError:
            tv = tv.astype(np.uint8)
        return span.clear_refrac.clearref(tv, window)

    def summary(self, func):
        assert any(imap(isinstance, (func, func), 
                        (basestring, types.FunctionType)))
        if hasattr(self.changroup, func):
            chan_func = getattr(self.changroup, func)
            chan_func_t = getattr(chan_func().T, func)
            return chan_func_t()
        elif hasattr(self.changroup, func.__name__):
            return self.summary(func.__name__)
        else:
            f = lambda x: func(self.flatten(x))
        
        return self.changroup.apply(f)

    def flatten(self, data):
        try:
            return data.stack().reset_index(drop=True)
        except MemoryError:
            raise MemoryError('out of memory')    
    
    @cached_property
    def changroup(self): return self.spikes.groupby(level=self.tsq.chan.name)

    @cached_property
    def shankgroup(self): return self.spikes.groupby(level=self.tsq.shank.name)


def dirsize(d):
    s = os.path.getsize(d)
    for item in glob.glob(os.path.join(d, '*')):
        path = os.path.join(d, item)
        if os.path.isfile(path):
            s += os.path.getsize(path)
        elif os.path.isdir(path):
            s += dirsize(path)
    return s
    

if __name__ == '__main__':
    import cProfile as profile
    import pstats
    import tempfile
    import glob
    fn = '/home/phillip/xcorr_data/Spont_Spikes_091210_p17rat_s4_657umV/Spont_Spikes_091210_p17rat_s4_657umV'
    globs = os.path.expanduser(os.path.join('~', 'xcorr_data'))
    fns = glob.glob(os.path.join(globs, '*'))
    fns = np.array([f for f in fns if os.path.isdir(f)])
    tevs = glob.glob(os.path.join(globs, '**', '*%stev' % os.extsep))
    tevsize = np.asanyarray(list(imap(os.path.getsize, tevs)))
    inds = np.argsort(tevsize)
    fns = np.fliplr(fns[inds][np.newaxis]).squeeze().tolist()
    for f in fns[-1:]:
        fn = os.path.join(f, os.path.basename(f))
        with tempfile.NamedTemporaryFile(mode='w+') as stats_file:
            stats_fn = stats_file.name
            profile.run('s = PandasTank("%s"); sp = s.spikes' % fn, stats_fn)
            p = pstats.Stats(stats_fn)
        p.strip_dirs().sort_stats('time').print_stats(0.05)
