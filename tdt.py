#!/usr/bin/env python
# encoding: utf-8
"""
"""
import os
import sys
import abc
import glob
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
import pylab

from scipy.stats import sem, ttest_ind

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


TsqFields = ('size', 'type', 'name', 'chan', 'sortcode', 'timestamp', 'fp_loc',
             'format', 'fs')
TsqNumpyTypes = (np.int32, np.int32, np.uint32, np.uint16, np.uint16,
                 np.float64, np.int64, np.int32, np.float32)
ElectrodeMap = np.array([[1, 3, 2, 6],
                         [7, 4, 5, 8],
                         [13, 12, 10, 9],
                         [14, 16, 11, 15]])

EventTypes = pd.Series({
    0x0: 'unknown',
    0x101: 'strobe_on',
    0x102: 'strobe_off',
    0x201: 'scaler',
    0x8101: 'stream',
    0x8201: 'snip',
    0x8801: 'mark'
})

ShankMap = np.array([0, 0, 0, 1, 1, 0, 1, 1, 2, 2, 3, 2, 2, 3, 3, 3])
_MedialLateral = np.array([0, 0, 0, 0,
                          0, 0, 0, 0,
                          1, 1, 1, 1,
                          1, 1, 1, 1])
MedialLateral = np.zeros(_MedialLateral.shape, dtype='S7')
MedialLateral[_MedialLateral == 0] = 'med'
MedialLateral[_MedialLateral == 1] = 'lat'

ChannelIndexer = pd.DataFrame({'channel': ElectrodeMap.ravel() - 1,
                               'shank': ShankMap,
                               'medlat': MedialLateral})


def name2num(name, base=256):
    """"Convert a string to a number"""
    return (base ** np.arange(len(name))).dot(np.array(tuple(imap(ord, name))))


def nans(size, dtype=np.float64):
    """Create an array of NaNs"""
    a = np.zeros(size, dtype=dtype)
    a.fill(np.nan)
    return a


def thunkify(f):
    """Perform `f` using a threaded thunk."""
    @functools.wraps(f)
    def thunked(*args, **kwargs):
        """The thunked version of `f`"""
        wait_event = threading.Event()
        result = [None]
        exc = [False, None]
        def worker():
            """The worker thread with which to run `f`"""
            try:
                result[0] = f(*args, **kwargs)
            except Exception as e:
                exc[0], exc[1] = True, sys.exc_info()
                print 'Lazy thunk has thrown any exception:\n%s' % traceback.format_exc()
            finally:
                wait_event.set()
        def thunk():
            """The actual thunk."""
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
    site_re = re.compile(r'.*s(?:ite)?(?:\s|_)*(\d+)')
    header_ext = 'tsq'
    raw_ext = 'tev'
    tsq_dtype = np.dtype(zip(TsqFields, TsqNumpyTypes))
    age_re = re.compile(r'.*[pP](\d+).*')

    def __init__(self, tankname):
        super(TdtTankBase, self).__init__()
        self.tankname = tankname
        self.animal_age = int(self.age_re.match(os.path.basename(tankname)).group(1))

        try:
            date = self.date_re.match(os.path.basename(self.tankname)).group(1)
        except AttributeError:
            now = pd.datetime.now()
            month, day, year = now.month, now.day, now.year
        else:
            datetmp = os.sep.join(i + j for i, j in izip(date[::2],
                                                         date[1::2])).split(os.sep)
            month, day, year = imap(int, datetmp)
        self._date = pd.datetime(year=year + 2000, month=month, day=day).date()
        site_match = self.site_re.match(os.path.basename(tankname))
        self.site = int(site_match.group(1))

    @property
    def date(self):
        return str(self._date)

    @abc.abstractmethod
    def _read_tev(self, event_name):
        pass

    @thunkify
    def _read_tsq(self):
        raw_tsq = np.fromfile(self.tankname + os.extsep + self.header_ext,
                              dtype=self.tsq_dtype)
        b = pd.DataFrame(raw_tsq)
        b.chan = b.chan.astype(np.float64)
        b.chan[b.chan - 1 == -1] = np.nan
        shank = pd.Series(ShankMap, name='shank')
        shank = shank[b.chan].reset_index(drop=True)
        side = pd.Series(MedialLateral, name='side')
        side = side[b.chan].reset_index(drop=True)
        return b.join(shank).join(side)

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
    def spike_fs(self):
        return self.fs.max()

    
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
        first_row = (row == 1).argmax()
        fmt = self.tsq.format[first_row]
        chans = self.tsq.chan[row] - 1
        fp_loc = self.tsq.fp_loc[row]
        nsamples = (self.tsq.size[first_row] - 10) * table[fmt][1]
        dtype = np.dtype(table[fmt][2]).type
        spikes = np.empty((fp_loc.size, nsamples), dtype=dtype)
        tev_name = self.tankname + os.extsep + self.raw_ext
        with open(tev_name, 'rb') as tev:
            with contextlib.closing(mmap.mmap(tev.fileno(), 0,
                                              access=mmap.ACCESS_READ)) as tev:
                for i, offset in enumerate(fp_loc):
                    spikes[i] = np.frombuffer(tev, dtype, nsamples, offset)
        shanks = self.tsq.shank[row]
        side = self.tsq.side[row]
        index = pd.MultiIndex.from_arrays((shanks, chans, side))
        return SpikeDataFrame(spikes, meta=self.tsq, index=index, dtype=dtype)
    

class SpikeDataFrameAbstractBase(pd.DataFrame):
    __metaclass__ = abc.ABCMeta

    def __init__(self, spikes, meta=None, *args, **kwargs):
        """ """
        super(SpikeDataFrameBase, self).__init__(*args, **kwargs)
        try:
            self.meta = spikes.meta
        except AttributeError:
            self.meta = meta
            if self.meta is None:
                raise ValueError('"meta" argument cannot be none')

    @abc.abstractproperty
    def fs(self):
        pass


class SpikeDataFrameBase(SpikeDataFrameAbstractBase):
    """ """
    def __init__(self, spikes, meta=None, *args, **kwargs):
        super(SpikeDataFrameBase, self).__init__(spikes, meta=meta, *args, **kwargs)
            
    @cached_property
    def channels(self):
        channels = self.channel_group.apply(self.flatten).T
        channels.columns = cast(channels.columns, np.int64)
        return channels

    def channel(self, i): return self.flatten(self.channel_group.get_group(i))

    @property
    def raw(self): return self.channels.values

    def iterchannels(self):
        for _, channel in self.channels.iteritems(): yield channel

    __iter__ = iterchannels

    def flatten(self, data):
        """ """
        try:
            # TODO: `stack` method is potentially very fragile
            return data.stack().reset_index(drop=True)
        except MemoryError:
            raise MemoryError('out of memory')

    @cached_property
    def channel_group(self):
        return self.groupby(level=self.meta.chan.name)

    @cached_property
    def shank_group(self):
        return self.groupby(level=self.meta.shank.name)

    @cached_property
    def side_group(self):
        return self.groupby(level=self.meta.side.name)

    @cached_property
    def fs(self):
        return self.meta.fs.unique().max()

    def mean(self): return self.summary('mean')
    def var(self): return self.channels.var()
    def std(self): return self.channels.std()
    def mad(self): return self.channels.mad()
    def sem(self): return pd.Series(sem(self.raw, axis=0))
    def median(self): return self.summary('median')
    def sum(self): return self.summary('sum')
    def max(self): return self.summary('max')
    def min(self): return self.summary('min')
    def bin(self, bins):
        ncolumns = channels.columns.size
        counts = empty((bins.size - 1, ncolumns))
        zipped_bins = zip(bins[:-1], bins[1:])
        for column in xrange(ncolumns):
            cold = data[column]
            counts[:, column] = asanyarray([chd[bi:bj].sum()
                                            for bi, bj in zipped_bins])
        return pd.DataFrame(counts)

    def summary(self, func):
        # get the correct string base class for the version of python
        stringbase = basestring if sys.version_info.major < 3 else str

        # check to make sure that `func` is a string or function
        func_is_valid = any(imap(isinstance, (func, func), (stringbase,
                                                            types.FunctionType)))
        assert func_is_valid, "'func' must be a string or function"

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

    def bin_times(self, threshes, ms=2, bin_size=1000):
        spike_times = self.spike_times(threshes, ms)
        bins = np.r_[:spike_times[-1]:bin_size]
        return self.bin(spike_times, bins)

    def refrac_window(self, ms=2, conv_factor=1e3):
        secs = ms / conv_factor
        samples = np.floor(secs * self.fs)
        try:
            samples = samples.astype(int, copy=False)
        except TypeError:
            samples = samples.astype(int)
        return samples

    def spike_times(self, threshes, ms=2, conv_factor=1e3):
        cleared = self.cleared(threshes, ms=ms)
        
    def cleared(self, threshes, ms=2):
        clr = self.threshold(threshes)
        window = self.refrac_window(ms)
        if clr.shape[0] < clr.shape[1]:
            clr = clr.T
        span.clear_refrac.clearref_out(clr.values.view(np.uint8), window)
        return clr

    def astype(self, dtype):
        return self._constructor(self._data, self.meta, index=self.index,
                                 columns=self.columns, dtype=dtype, copy=False)

    def make_new(self, data, dtype=None):
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

    def plot_thresh_range(self, low, high=None, n=50, refrac_period=2):
        """Compute the total number of spikes across all channels for an evenly
        spaced range of threshold values. Also plot the resulting curve:
        number of spikes vs. threshold as well as the discrete difference of the
        spike counts, to look for discontinuities.

        Parameters
        ----------
        low : int
        high : int, optional
        n : int, optional
        refrac_period int, optional
        """
        if high is None:
            low, high = 0, low

        value_range = np.linspace(low, high, n)
        total_spikes = np.zeros(value_range.shape, dtype=np.int64)
        for i, value in enumerate(value_range):
            total_spikes[i] = self.cleared(value, refrac_period).sum().sum()
        fig = pl.figure()

        ax1 = fig.add_subplot(211)
        ax1.plot(value_range, total_spikes)
        ax1.set_xlabel('Threshold')
        ax1.set_ylabel('Spike Count')
        ax1.set_title('Spike Count vs. Threshold')
        ax1.axis('tight')

        ax2 = fig.add_subplot(212)
        ax2.plot(value_range[1:], np.diff(total_spikes))
        ax2.set_xlabel('Threshold')
        ax2.set_ylabel(r'$\mathrm{d}/\mathrm{d}x$ Spike Count')
        ax2.set_title('$\mathrm{d}/\mathrm{d}x$ Spike Count vs. Threshold')
        ax2.axis('tight')

    def threshold(self, thresh):
        return self.gt(thresh)


class SparseSpikeDataFrame(SpikeDataFrameBase):
    def __init__(self, spikes, meta=None, *args, **kwargs):
        super(SpikeDataFrameBase, self).__init__(spikes, meta=meta, *args,
                                                 **kwargs)

        
def dirsize(d):
    s = os.path.getsize(d)
    for item in glob.glob(os.path.join(d, '*')):
        path = os.path.join(d, item)
        if os.path.isfile(path):
            s += os.path.getsize(path)
        elif os.path.isdir(path):
            s += dirsize(path)
    return s


def get_tank_names(path=os.path.expanduser(os.path.join('~', 'xcorr_data'))):
    globs = path
    fns = glob.glob(os.path.join(globs, '*'))
    fns = np.array([f for f in fns if os.path.isdir(f)])
    tevs = glob.glob(os.path.join(globs, '**', '*%stev' % os.extsep))
    tevsize = np.asanyarray(list(imap(os.path.getsize, tevs)))
    inds = np.argsort(tevsize)
    fns = np.fliplr(fns[inds][np.newaxis]).squeeze().tolist()
    return fns


def profile_spikes():
    import cProfile as profile
    import pstats
    import tempfile

    fns = get_tank_names()

    for f in fns[-1:]:
        fn = os.path.join(f, os.path.basename(f))
        with tempfile.NamedTemporaryFile(mode='w+') as stats_file:
            stats_fn = stats_file.name
            profile.run('sp = PandasTank("%s").spikes' % fn, stats_fn)
            p = pstats.Stats(stats_fn)
            try:
                del sp
            except NameError:
                pass
        p.strip_dirs().sort_stats('time').print_stats(0.05)
    return fns


def side_sums(n=None):
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
    fns = profile_spikes()
    # n = None
    # try:
    #     n = int(sys.argv[1])
    # except:
    #     pass
    # else:
    #     if n <= 0:
    #         raise ValueError('invalid value for n: %i, must be > 0' % n)
    # side_sums(n)
