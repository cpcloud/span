from unittest import TestCase
import numbers
import itertools as itools

import numpy as np
from numpy.random import rand, randint
from numpy.testing.decorators import slow

import pandas as pd
from pandas import Series
from pandas.util.testing import assert_frame_equal


from span.tdt.spikedataframe import SpikeDataFrame, _create_xcorr_inds
from span.utils import detrend_mean, detrend_linear
from span.testing import assert_all_dtypes, create_spike_df, assert_array_equal
from span.testing import skip


class TestSpikeDataFrameBase(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = create_spike_df()
        tsq = cls.df.meta
        cls.meta = tsq

        nsamples, _ = tsq.shape[0], tsq.size.unique().max()
        size = nsamples * tsq.shape[1], tsq.channel.nunique()
        cls.rows, cls.columns = size

    @classmethod
    def tearDownClass(cls):
        del cls.columns, cls.rows, cls.meta, cls.df

    def test_sem(self):
        df = self.df

        for axis, ddof in itools.product((0, 1), (0, 1)):
            s = df.sem(axis=axis, ddof=ddof)
            self.assertIsInstance(s, Series)
            self.assertEqual(s.values.size, df.shape[1 - axis])

    def setUp(self):
        self.spikes = self.df.copy()
        self.raw = self.spikes.values.copy()

    def tearDown(self):
        del self.raw, self.spikes

    def test_constructor(self):
        x = self.spikes._constructor(self.spikes.values)
        self.assertIsInstance(x, SpikeDataFrame)
        self.assertIsNotNone(x.meta)

    def test_init_noindex_nocolumns(self):
        spikes = SpikeDataFrame(self.raw, self.meta)
        self.assertRaises(AssertionError, SpikeDataFrame, self.raw, None)
        self.assertRaises(TypeError, SpikeDataFrame, self.raw)
        assert_array_equal(spikes.values, self.raw)

    def test_init_noindex_columns(self):
        from span.tdt.spikeglobals import ChannelIndex as columns
        spikes = SpikeDataFrame(self.raw, self.meta, columns=columns)
        self.assertRaises(AssertionError, SpikeDataFrame, self.raw, None)
        self.assertRaises(TypeError, SpikeDataFrame, self.raw)
        assert_array_equal(spikes.values, self.raw)
        assert_array_equal(spikes.columns.values, columns.values)

    def test_init_with_index_nocolumns(self):
        spikes = SpikeDataFrame(self.raw, self.meta, index=self.spikes.index)
        assert_array_equal(spikes.values, self.raw)
        assert_array_equal(spikes.index.values, self.spikes.index.values)

    def test_init_with_index_with_columns(self):
        from span.tdt.spikeglobals import ChannelIndex as columns
        cols, _ = columns.swaplevel(1, 0).sortlevel('shank')
        spikes = SpikeDataFrame(self.raw, self.meta, index=self.spikes.index,
                                columns=cols)
        assert_frame_equal(spikes, self.df)

    def test_fs(self):
        fs = self.spikes.fs
        self.assertEqual(fs, self.meta.fs.max())
        assert_array_equal([fs] * self.meta.fs.size, self.meta.fs)
        self.assertIsInstance(fs, float)

    def test_nchans(self):
        nchans = self.spikes.nchans
        cols = self.spikes.columns
        values = cols.levels[cols.names.index('channel')].values
        self.assertEqual(nchans, values.max() + 1)

    def test_chunk_size(self):
        cs = self.spikes.chunk_size
        self.assertIsInstance(cs, numbers.Integral)

    def test_sort_code(self):
        sc = self.spikes.sort_code
        self.assertIsInstance(sc, numbers.Integral)

    def test_dtype(self):
        dtype = self.spikes.dtype
        self.assertIsInstance(dtype, np.dtype)

    def test_tdt_type(self):
        tt = self.spikes.tdt_type
        self.assertIsInstance(tt, basestring)

    def test_nshanks(self):
        nsh = self.spikes.nshanks
        self.assertIsInstance(nsh, numbers.Integral)


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

    def test_bin(self):
        sp = self.spikes
        thr = sp.threshold(randint(1, 4) * sp.std())
        clr = sp.clear_refrac(thr)
        binsizes = -1, 0, 1, 10
        reject_counts = -1, 0, 100
        dropnas = True, False
        args = itools.product(binsizes, reject_counts, dropnas)

        for binsize, reject_count, dropna in args:
            if binsize > 0 and reject_count >= 0:
                binned = sp.bin(clr, binsize, reject_count, dropna)
                self.assertIsInstance(binned, SpikeDataFrame)
            else:
                self.assertRaises(AssertionError, sp.bin, clr, binsize,
                                  reject_count, dropna)

    def test_fr(self):
        sp = self.spikes
        thr = sp.threshold(3 * sp.std())
        clr = sp.clear_refrac(thr)
        binned = sp.bin(clr, binsize=4)
        axes = 0, 1
        sems = True, False
        args = itools.product(sp.columns.names,
                              xrange(len(sp.columns.names)),
                              axes,
                              sems)

        for level_name, level_number, axis, sem in args:
            for t in (level_name, level_number):
                if axis:
                    if sem:
                        mfr, sm = sp.fr(binned, t, axis, sem)
                        self.assertTupleEqual(mfr.shape, sm.shape)
                    else:
                        mfr = sp.fr(binned, t, axis, sem)
                else:
                    if t:
                        self.assertRaises(ValueError, sp.fr, binned, t, axis,
                                          sem)
                    else:
                        mfr = sp.fr(binned, t, axis, sem)

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
        binned = self.spikes.bin(clr, binsize=10)

        maxlags = None, 2, binned.shape[0] + 1
        detrends = detrend_mean, detrend_linear
        scale_types = 'normalize', 'unbiased', 'biased'
        sortlevels = 'channel i', 'channel j', 'shank i', 'shank j'
        sortlevels += tuple(range(len(sortlevels)))
        nan_autos = True, False
        lag_names = 'a',

        args = itools.product(maxlags, detrends, scale_types, sortlevels,
                              nan_autos, lag_names)

        for maxlag, detrend, scale_type, level, nan_auto, lag_name in args:
            if maxlag > binned.shape[0]:
                self.assertRaises(AssertionError, self.spikes.xcorr, binned,
                                  maxlag, detrend, scale_type, level,
                                  nan_auto, lag_name)
            else:
                xc = self.spikes.xcorr(binned, maxlag, detrend, scale_type,
                                       level, nan_auto, lag_name)
                self.assertIsInstance(xc, pd.DataFrame)

            self.assertRaises(AssertionError, self.spikes.xcorr, binned,
                              maxlag, detrend, scale_type, 'asdfalsdj',
                              nan_auto, lag_name)
            self.assertRaises(AssertionError, self.spikes.xcorr, binned,
                              maxlag, detrend, scale_type, 2342,
                              nan_auto, lag_name)


class TestCreateXCorrInds(TestCase):
    def test_create_xcorr_inds(self):
        nchannels = xrange(2, 33, 2)
        names = 'shank i', 'shank j', 'channel i', 'channel j'
        for nchannel in nchannels:
            inds = _create_xcorr_inds(nchannel)
            self.assertIsInstance(inds, pd.MultiIndex)
            self.assertEqual(tuple(inds.names), names)
            self.assertEqual(len(inds), nchannel ** 2)
