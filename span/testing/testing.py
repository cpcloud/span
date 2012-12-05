import functools
import nose
import copy

import numpy as np
from numpy import int64, float64
from numpy.testing import assert_allclose, assert_array_equal
from numpy.testing.decorators import slow

from nose.tools import nottest
from nose import SkipTest

from pandas import Series, DataFrame, Int64Index
from pandas.util.testing import *
_rands = copy.deepcopy(rands)
del rands

from numpy.random import uniform as randrange, randint

import span
from span.tdt.spikeglobals import Indexer
from span.tdt.tank import _reshape_spikes
# from span.utils import cartesian

def assert_all_dtypes(df, dtype, msg='dtypes not all the same'):
    assert all(dt == dtype for dt in df.dtypes), msg


def skip(test):
    @functools.wraps(test)
    def wrapper():
        if mock:
            return test()

        raise nose.SkipTest


def create_stsq(size=None, typ='stream',
                name=span.utils.name2num('Spik'), nchannels=16, sort_code=0,
                fmt=np.float32, fs=4882.8125,
                samples_per_channel=None):
    if size is None:
        size = randint(2 ** 5, 2 ** 6)

    if samples_per_channel is None:
        samples_per_channel = randint(2 ** 6, 2 ** 7)

    names = ('size', 'type', 'name', 'channel', 'sort_code', 'timestamp',
             'fp_loc', 'format', 'fs', 'shank', 'side')
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
    side = srt.side[channel].reset_index(drop=True)

    data = [size, typ, name, channel, sort_code, timestamp, fp_loc, fmt, fs,
            shank, side]

    return DataFrame(dict(zip(names, data)))


def create_spike_df(size=None, typ='stream', name=span.utils.name2num('Spik'),
                    nchannels=16, sort_code=0, fmt=np.float32, fs=4882.8125,
                    samples_per_channel=None):
    tsq = create_stsq(size, typ, name, nchannels, sort_code, fmt, fs,
                      samples_per_channel)
    spikes = np.empty((tsq.fp_loc.size, tsq.size[0]), tsq.format[0])
    tsq.timestamp = span.utils.fromtimestamp(tsq.timestamp)
    tsq = tsq.reset_index(drop=True)
    groups = DataFrame(tsq.groupby('channel').groups).values
    df = _reshape_spikes(spikes, groups, tsq, tsq.fs.unique().item(),
                         tsq.channel.dropna().nunique(), tsq.timestamp[0])
    return df()


def rands(size, shape):
    r = []

    n = np.prod(shape)

    for i in xrange(n):
        r.append(_rands(size))

    return np.asarray(r).reshape(shape)
