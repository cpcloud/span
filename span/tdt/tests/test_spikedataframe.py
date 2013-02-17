from unittest import TestCase
import itertools as itools

import numpy as np
from numpy.random import rand, randint
from numpy.testing.decorators import slow

import pandas as pd
from pandas import Series, MultiIndex
from pandas.util.testing import assert_frame_equal


from span.tdt.spikedataframe import SpikeDataFrame, _create_xcorr_inds
from span.utils import detrend_mean, detrend_linear
from span.testing import assert_all_dtypes, create_spike_df, assert_array_equal


class TestSpikeDataFrameBase(TestCase):
    def test_sem(self):
        df = self.spikes

        for axis, ddof in itools.product((0, 1), (0, 1)):
            s = df.sem(axis=axis, ddof=ddof)
            self.assertIsInstance(s, Series)
            self.assertEqual(s.values.size, df.shape[1 - axis])

    def setUp(self):
        self.spikes = create_spike_df()
        self.rows, self.columns = self.spikes.shape
        self.raw = self.spikes.values.copy()

    def tearDown(self):
        del self.raw, self.spikes

    def test_constructor(self):
        x = self.spikes._constructor(self.spikes.values)
        self.assertIsInstance(x, SpikeDataFrame)

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
        self.assertIsInstance(fs, float)

    def test_nchannels(self):
        nchans = self.spikes.nchannels
        cols = self.spikes.columns
        values = cols.levels[cols.names.index('channel')].values
        self.assertEqual(nchans, values.max() + 1)

    def test_nsamples(self):
        nsamps = self.spikes.nsamples
        self.assertEqual(nsamps, self.spikes.shape[0])

        fs = self.spikes.fs
        expected = round((self.spikes.nsamples / fs) * fs)
        self.assertEqual(nsamps, expected)


class TestSpikeDataFrame(TestCase):
    def setUp(self):
        self.spikes = create_spike_df()
        self.threshes = 2e-5, 3e-5
        self.mses = None, 2, 3
        self.binsizes = 0, 1, randint(10, 12)

    def tearDown(self):
        del self.spikes

    def test_threshold_std_array(self):
        sp = self.spikes
        std = sp.std()
        thresh = 2.0
        thr = sp.threshold(thresh * std)
        assert_all_dtypes(thr, np.bool_)
        self.assertTupleEqual(thr.shape, sp.shape)

    def test_threshold_std_scalar(self):
        sp = self.spikes
        std = sp.std()
        thr = sp.threshold(std.values[randint(std.size)])
        assert_all_dtypes(thr, np.bool_)
        self.assertTupleEqual(thr.shape, sp.shape)

    def test_threshold_array(self):
        sp = self.spikes
        std = rand(sp.shape[1])
        thr = sp.threshold(2.0 * std)
        assert_all_dtypes(thr, np.bool_)
        self.assertTupleEqual(thr.shape, sp.shape)

    def test_threshold_scalar(self):
        sp = self.spikes
        std = rand()
        thr = sp.threshold(2.0 * std)
        assert_all_dtypes(thr, np.bool_)
        self.assertTupleEqual(thr.shape, sp.shape)

    def test_clear_refrac(self):
        thr = self.spikes.threshold(3 * self.spikes.std())

        for ms in self.mses:
            cleared = self.spikes.clear_refrac(thr, ms)
            assert_all_dtypes(cleared, np.bool_)
            self.assertLessEqual(cleared.values.sum(), thr.values.sum())

    def test_constructor(self):
        s = self.spikes
        ind = self.spikes.index
        cols = self.spikes.columns
        s_new = self.spikes._constructor(s.values, index=ind, columns=cols)
        self.assertIsInstance(s_new, pd.DataFrame)
        self.assertIsInstance(s_new, SpikeDataFrame)
        self.assertIsInstance(s, type(s_new))

    @slow
    def test_xcorr(self):
        thr = self.spikes.threshold(self.spikes.std())
        clr = self.spikes.clear_refrac(thr)
        binned = clr.resample('L', how='sum')

        maxlags = None, 2, binned.shape[0] + 1
        detrends = detrend_mean, detrend_linear
        scale_types = 'normalize', 'unbiased', 'biased'
        sortlevels = 'shank i', 'channel i', 'shank j', 'channel j'
        sortlevels += tuple(xrange(len(sortlevels)))
        nan_autos = True, False
        lag_names = 'a',

        args = itools.product(maxlags, detrends, scale_types, sortlevels,
                              nan_autos, lag_names)

        weird_levels = 'asdfalsdj', 2342
        for maxlag, detrend, scale_type, level, nan_auto, lag_name in args:
            if maxlag > binned.shape[0]:
                self.assertRaises(AssertionError, self.spikes.xcorr, binned,
                                  maxlag, detrend, scale_type, level,
                                  nan_auto, lag_name)
            else:
                xc = self.spikes.xcorr(binned, maxlag, detrend, scale_type,
                                       level, nan_auto, lag_name)
                self.assertIsInstance(xc, pd.DataFrame)
                self.assertRaises(TypeError, self.spikes.xcorr, binned, maxlag,
                                  detrend, scale_type, object(), nan_auto,
                                  lag_name)

                for level in weird_levels:
                    self.assertRaises(AssertionError, self.spikes.xcorr,
                                      binned, maxlag, detrend, scale_type,
                                      level, nan_auto, lag_name)


class TestCreateXCorrInds(TestCase):
    def setUp(self):
        self.spikes = create_spike_df()

    def tearDown(self):
        del self.spikes

    def test_create_xcorr_inds(self):
        thr = self.spikes.threshold(self.spikes.std())
        clr = self.spikes.clear_refrac(thr)
        binned = clr.resample('L', how='sum')
        inds = _create_xcorr_inds(binned.columns)
        self.assertIsInstance(inds, MultiIndex)
