import functools

import numpy as np
from numpy.testing import *
from numpy.testing.decorators import slow

from nose.tools import nottest, assert_raises
from nose import SkipTest

from pandas import Series, DataFrame, Int64Index, DatetimeIndex
from pandas.util.testing import rands
import pandas as pd


from numpy.random import uniform as randrange, randint

import span
from span.tdt import SpikeDataFrame
from span.tdt.spikeglobals import Indexer


def assert_all_dtypes(df, dtype, msg='dtypes not all the same'):
    assert all(dt == dtype for dt in df.dtypes), msg


def skip(test):
    @functools.wraps(test)
    def wrapper():
        if mock:
            return test()

        raise SkipTest


def create_stsq(size=None, typ='stream',
                name=span.utils.name2num('Spik'), nchannels=16, sort_code=0,
                fmt=np.float32, fs=4882.8125,
                samples_per_channel=None, strobe=None):
    if size is None:
        size = randint(2 ** 3, 2 ** 4)

    if samples_per_channel is None:
        samples_per_channel = randint(2 ** 5, 2 ** 6)

    names = ('size', 'type', 'name', 'channel', 'sort_code', 'timestamp',
             'fp_loc', 'format', 'fs', 'shank')
    nsamples = samples_per_channel * nchannels
    index = Int64Index(np.arange(nsamples))

    size = Series(size, index=index, name='size')
    typ = Series(typ, index=index, name='type')
    name = Series(name, index=index, name='name')
    channel = Series(np.tile(np.arange(nchannels),
                             (samples_per_channel, 1))[:, ::-1].ravel(),
                     index=index, name='channel')
    sort_code = Series(sort_code, index=index, name='sort_code')
    fp_loc = Series(index, name='fp_loc')
    fmt = Series([fmt] * index.size, index=index, name='format')

    start = randrange(100e7, 128e7)
    ts = start + (size[0] / fs) * np.r_[:samples_per_channel]
    ts = np.tile(ts, (nchannels, 1)).T.ravel()
    timestamp = Series(ts, index=index, name='timestamp')

    srt = Indexer.sort('channel').reset_index(drop=True)
    shank = srt.shank[channel].reset_index(drop=True)

    data = (size, typ, name, channel, sort_code, timestamp, fp_loc, fmt, fs,
            shank)

    return DataFrame(dict(zip(names, data)))


def create_spike_df(size=None, typ='stream', name=span.utils.name2num('Spik'),
                    nchannels=16, sort_code=0, fmt=np.float32, fs=4882.8125,
                    samples_per_channel=None):
    from span.tdt.spikeglobals import ChannelIndex as columns
    tsq = create_stsq(size, typ, name, nchannels, sort_code, fmt, fs,
                      samples_per_channel)

    tsq.timestamp = span.utils.fromtimestamp(tsq.timestamp)
    tsq = tsq.reset_index(drop=True)
    groups = DataFrame(tsq.groupby('channel').groups)
    spikes = np.random.randn(groups.shape[0] * tsq.size.get_value(0),
                             nchannels) / 1e5
    ns = int(1e9 / tsq.fs.get_value(0))
    dtstart = np.datetime64(tsq.timestamp.get_value(0))
    dt = dtstart + np.arange(spikes.shape[0]) * np.timedelta64(ns, 'ns')
    index = DatetimeIndex(dt, freq=ns * pd.datetools.Nano(), name='time',
                          tz='US/Eastern')
    columns = columns.swaplevel(1, 0)
    return SpikeDataFrame(DataFrame(spikes, index, columns).sort_index(axis=1))


def knownfailure(test):
    """Let nose know that we know a test fails."""

    @functools.wraps(test)
    def inner(*args, **kwargs):
        try:
            test(*args, **kwargs)
        except Exception:
            raise SkipTest
        else:
            raise AssertionError('Failure expected')

    return inner
