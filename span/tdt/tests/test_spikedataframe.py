from unittest import TestCase
import itertools as itools

import numpy as np
from numpy.random import rand, randint
from numpy.testing.decorators import slow

import pandas as pd
from pandas import MultiIndex
from pandas.util.testing import assert_frame_equal


from span.tdt.spikedataframe import SpikeDataFrame
from span.utils import detrend_mean, detrend_linear, detrend_none
from span.testing import (assert_all_dtypes, create_spike_df,
                          assert_array_equal, assert_raises)


class TestSpikeDataFrame(object):
    def setUp(self):
        self.spikes = create_spike_df()
        self.raw = self.spikes.values
        self.threshes = 2e-5, 3e-5
        self.mses = None, 2, 3
        self.binsizes = 0, 1, randint(10, 12)

    def tearDown(self):
        del self.spikes

    def test_init_noindex_nocolumns(self):
        spikes = SpikeDataFrame(self.raw)
        assert_array_equal(spikes.values, self.raw)

    def test_init_noindex_columns(self):
        from span.tdt.spikeglobals import ChannelIndex as columns
        spikes = SpikeDataFrame(self.raw, columns=columns)
        assert_array_equal(spikes.values, self.raw)
        assert_array_equal(spikes.columns.values, columns.values)

    def test_init_with_index_nocolumns(self):
        spikes = SpikeDataFrame(self.raw, index=self.spikes.index)
        assert_array_equal(spikes.values, self.raw)
        assert_array_equal(spikes.index.values, self.spikes.index.values)

    def test_init_with_index_with_columns(self):
        from span.tdt.spikeglobals import ChannelIndex as columns
        cols, _ = columns.swaplevel(1, 0).sortlevel('shank')
        spikes = SpikeDataFrame(self.raw, index=self.spikes.index,
                                columns=cols)
        assert_frame_equal(spikes, self.spikes)

    def test_fs(self):
        fs = self.spikes.fs
        assert isinstance(fs, float)

    def test_nchannels(self):
        nchans = self.spikes.nchannels
        cols = self.spikes.columns
        values = cols.levels[cols.names.index('channel')].values
        assert nchans == values.max() + 1

    def test_nsamples(self):
        nsamps = self.spikes.nsamples
        assert nsamps == self.spikes.shape[0]

        fs = self.spikes.fs
        expected = round((self.spikes.nsamples / fs) * fs)
        assert nsamps == expected

    def test_threshold_std_array(self):
        sp = self.spikes
        std = sp.std()
        thresh = 2.0
        thr = sp.threshold(thresh * std)
        assert_all_dtypes(thr, np.bool_)
        assert thr.shape == sp.shape

    def test_threshold_std_scalar(self):
        sp = self.spikes
        std = sp.std()
        thr = sp.threshold(std.values[randint(std.size)])
        assert_all_dtypes(thr, np.bool_)
        assert thr.shape == sp.shape

    def test_threshold_array(self):
        sp = self.spikes
        std = rand(sp.shape[1])
        thr = sp.threshold(2.0 * std)
        assert_all_dtypes(thr, np.bool_)
        assert thr.shape == sp.shape

    def test_threshold_scalar(self):
        sp = self.spikes
        std = rand()
        thr = sp.threshold(2.0 * std)
        assert_all_dtypes(thr, np.bool_)
        assert thr.shape == sp.shape

    def test_clear_refrac(self):
        thr = self.spikes.threshold(3 * self.spikes.std())

        for ms in self.mses:
            cleared = self.spikes.clear_refrac(thr, ms)
            assert_all_dtypes(cleared, np.bool_)
            assert cleared.values.sum() <= thr.values.sum()

    def test_constructor(self):
        s = self.spikes
        ind = self.spikes.index
        cols = self.spikes.columns
        s_new = self.spikes._constructor(s.values, index=ind, columns=cols)
        assert isinstance(s_new, pd.DataFrame)
        assert isinstance(s_new, SpikeDataFrame)
        assert isinstance(s, type(s_new))

    @slow
    def test_xcorr(self):
        maxlags = None, 2
        detrends = None, detrend_none, detrend_mean, detrend_linear
        scale_types = None, 'none', 'normalize', 'unbiased', 'biased'
        sortlevels = 'shank i', 'channel i', 'shank j', 'channel j'
        sortlevels += tuple(xrange(len(sortlevels)))
        nan_autos = True, False
        lag_names = 'a',

        args = itools.product(maxlags, detrends, scale_types, sortlevels,
                              nan_autos, lag_names)

        for maxlag, detrend, scale_type, level, nan_auto, lag_name in args:
            yield (self.xcorr_builder, maxlag, detrend, scale_type, level,
                   nan_auto, lag_name)

    def xcorr_builder(self, maxlag, detrend, scale_type, level, nan_auto,
                      lag_name):
        thr = self.spikes.threshold(self.spikes.std())
        clr = self.spikes.clear_refrac(thr)
        binned = clr.resample('L', how='sum')
        xc = self.spikes.xcorr(binned, maxlag, detrend, scale_type, level,
                               nan_auto, lag_name)
        assert isinstance(xc, pd.DataFrame)

        weird_levels = 'asdfalsdj', 2342, object()

        for level in weird_levels:
            assert_raises(Exception, self.spikes.xcorr, binned, maxlag,
                          detrend, scale_type, level, nan_auto, lag_name)
        assert_raises(AssertionError, self.spikes.xcorr, binned,
                      binned.shape[0] + 10, detrend, scale_type, level,
                      nan_auto, lag_name)

    def test_jitter(self):
        jittered = self.spikes.jitter()
        assert not np.array_equal(jittered, self.spikes)
