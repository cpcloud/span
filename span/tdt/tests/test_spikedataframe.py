import itertools as itools

import numpy as np
from numpy.random import rand, randint
from numpy.testing.decorators import slow

import pandas as pd
from pandas.util.testing import assert_frame_equal


from span.tdt.spikedataframe import SpikeDataFrame
from span.utils import detrend_mean, detrend_linear, detrend_none
from span.testing import (assert_all_dtypes, create_spike_df,
                          assert_array_equal, assert_raises,
                          assert_frame_equal)


class TestSpikeDataFrame(object):
    def setUp(self):
        self.spik = create_spike_df()
        self.raw = self.spik.values
        self.threshes = 2e-5, 3e-5
        self.mses = None, 2, 3
        self.binsizes = 0, 1, randint(10, 12)

    def tearDown(self):
        del self.spik

    def test_init_noindex_nocolumns(self):
        spikes = SpikeDataFrame(self.raw)
        assert_array_equal(spikes.values, self.raw)

    def test_init_noindex_columns(self):
        spikes = SpikeDataFrame(self.raw, columns=self.spik.columns)
        assert_array_equal(spikes.values, self.raw)
        assert_array_equal(spikes.columns.values, self.spik.columns.values)

    def test_init_with_index_nocolumns(self):
        spikes = SpikeDataFrame(self.raw, index=self.spik.index)
        assert_array_equal(spikes.values, self.raw)
        assert_array_equal(spikes.index.values, self.spik.index.values)

    def test_init_with_index_with_columns(self):
        spikes = SpikeDataFrame(self.raw, self.spik.index, self.spik.columns)
        assert_array_equal(spikes.values, self.raw)
        assert_array_equal(spikes.columns.values, self.spik.columns.values)
        assert_array_equal(spikes.index.values, self.spik.index.values)

    def test_init_with_other_frame(self):
        spikes = SpikeDataFrame(self.spik)
        assert_frame_equal(spikes, self.spik)

    def test_fs(self):
        fs = self.spik.fs
        assert isinstance(fs, float)

    def test_nchannels(self):
        nchannels = self.spik.nchannels
        assert nchannels == len(self.spik.columns)

    def test_nsamples(self):
        nsamps = self.spik.nsamples
        assert nsamps == self.spik.shape[0]

        fs = self.spik.fs
        expected = round((self.spik.nsamples / fs) * fs)
        assert nsamps == expected

    def test_threshold_std_array(self):
        sp = self.spik
        std = sp.std()
        thresh = 2.0
        thr = sp.threshold(thresh * std)
        assert_all_dtypes(thr, np.bool_)
        assert thr.shape == sp.shape

    def test_threshold_std_scalar(self):
        sp = self.spik
        std = sp.std()
        thr = sp.threshold(std.values[randint(std.size)])
        assert_all_dtypes(thr, np.bool_)
        assert thr.shape == sp.shape

    def test_threshold_array(self):
        sp = self.spik
        std = rand(sp.shape[1])
        thr = sp.threshold(2.0 * std)
        assert_all_dtypes(thr, np.bool_)
        assert thr.shape == sp.shape

    def test_threshold_scalar(self):
        sp = self.spik
        std = rand()
        thr = sp.threshold(2.0 * std)
        assert_all_dtypes(thr, np.bool_)
        assert thr.shape == sp.shape

    def test_clear_refrac(self):
        thr = self.spik.threshold(3 * self.spik.std())

        for ms in self.mses:
            cleared = thr.clear_refrac(ms)
            assert_all_dtypes(cleared, np.bool_)
            assert cleared.values.sum() <= thr.values.sum()

    def test_constructor(self):
        s = self.spik
        ind = self.spik.index
        cols = self.spik.columns
        s_new = self.spik._constructor(s.values, index=ind, columns=cols)
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

        args = itools.product(maxlags, detrends, scale_types, sortlevels,
                              nan_autos)

        for maxlag, detrend, scale_type, level, nan_auto in args:
            yield (self.xcorr_builder, maxlag, detrend, scale_type, level,
                   nan_auto)

    def xcorr_builder(self, maxlag, detrend, scale_type, level, nan_auto):
        thr = self.spik.threshold(self.spik.std())
        thr.clear_refrac(inplace=True)
        clr = thr
        binned = clr.resample('L', how='sum')
        xc = self.spik.xcorr(binned, maxlag, detrend, scale_type, level,
                             nan_auto)
        assert isinstance(xc, pd.DataFrame)

        weird_levels = 'asdfalsdj', 2342, object()

        for level in weird_levels:
            assert_raises(Exception, self.spik.xcorr, binned, maxlag,
                          detrend, scale_type, level, nan_auto)
        assert_raises(AssertionError, self.spik.xcorr, binned,
                      binned.shape[0] + 10, detrend, scale_type, level,
                      nan_auto)

    def test_basic_jitter(self):
        jittered = self.spik.basic_jitter()
        assert not np.array_equal(jittered, self.spik)
