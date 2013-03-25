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


def assert_all_dtypes(df, dtype, msg='dtypes not all the same'):
    assert all(dt == dtype for dt in df.dtypes), msg


def skip(test):
    @functools.wraps(test)
    def wrapper():
        if mock:
            return test()

        raise SkipTest


def create_stsq(size=None, typ='stream',
                name=span.utils.name2num('Spik'), nchannels=16, nshanks=4,
                sort_code=0, fmt=np.float32, fs=1000, samples_per_channel=None,
                strobe=None):
    if size is None:
        size = 8

    if samples_per_channel is None:
        samples_per_channel = 32

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
    shank = Series(np.repeat(np.arange(nshanks), nchannels // nshanks),
                   name='shank').ix[channel]
    sort_code = Series(sort_code, index=index, name='sort_code')
    fp_loc = Series(index, name='fp_loc')
    fmt = Series([fmt] * index.size, index=index, name='format')

    start = randrange(100e7, 128e7)
    ts = start + (size[0] / fs) * np.r_[:samples_per_channel]
    ts = np.tile(ts, (nchannels, 1)).T.ravel()
    timestamp = Series(ts, index=index, name='timestamp')

    fs = np.repeat(fs, timestamp.size)

    data = (size, typ, name, channel, sort_code, timestamp, fp_loc, fmt, fs,
            shank)

    return DataFrame(dict(zip(names, data)))


def create_elec_map(channels_per_shank, nshanks):
    nchannels = channels_per_shank * nshanks
    wthn = randint(20, 100)
    btwn = randint(20, 100)
    return ElectrodeMap(np.arange(nchannels).reshape(-1, nshanks), wthn, btwn)


def create_random_elec_map():
    channels_per_shank = randint(4, 9)
    nshanks = randint(4, 9)
    return create_elec_map(channels_per_shank, nshanks)


def create_spike_df(size=None, typ='stream', name=span.utils.name2num('Spik'),
                    nchannels=16, nshanks=4, sort_code=0, fmt=np.float32,
                    fs=4882.8125, samples_per_channel=None):
    tsq = create_stsq(size, typ, name, nchannels, nshanks, sort_code, fmt, fs,
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
    return SpikeDataFrame(DataFrame(spikes, index).sort_index(axis=1))


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
