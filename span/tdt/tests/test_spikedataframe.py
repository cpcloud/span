
import os
import unittest
import itertools

# from nose.tools import nottest

import numpy as np
from numpy.random import randn, rand, randint
from numpy.testing.decorators import slow

import pandas as pd
from pandas import Series
from pandas.util.testing import assert_frame_equal

from span.tdt.tank import PandasTank
from span.tdt.spikedataframe import (SpikeDataFrameBase, SpikeDataFrame,
                                     SpikeGroupedDataFrame)
from span.utils import (detrend_none, detrend_mean, detrend_linear,
                        fromtimestamp)
from span.testing import (assert_all_dtypes, randrange, create_spike_df,
                          assert_allclose, assert_array_equal)


class TestSpikeGrouper(unittest.TestCase):
    pass


class TestChannelGetter(unittest.TestCase):
    pass


class TestShankGetter(unittest.TestCase):
    pass


class TestSideGetter(unittest.TestCase):
    pass


class TestSpikeGroupedDataFrame(unittest.TestCase):
    def setUp(self):
        self.x = randn(randint(10, 20), randint(10, 20))

    def tearDown(self):
        del self.x

    def test_constructor(self):
        df0 = SpikeGroupedDataFrame(self.x)

        df1 = df0._constructor(df0)
        df2 = df0._constructor(df0.values)
        df3 = df0._constructor(self.x)

        for df in (df1, df2, df3):
            assert_frame_equal(df0, df)

    def test_sem(self):
        df = SpikeGroupedDataFrame(self.x)

        for axis, ddof in itertools.product((0, 1), (0, 1)):
            s = df.sem(axis=axis, ddof=ddof)
            self.assertIsInstance(s, Series)
            self.assertEqual(s.values.size, df.shape[1 - axis])


class TestSpikeDataFrameBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = create_spike_df()
        tsq = cls.df.meta
        cls.meta = tsq

        nsamples, _ = tsq.shape[0], tsq.size.unique().max()
        size = nsamples * tsq.shape[1], tsq.channel.nunique()

        cls.rows, cls.columns = size
        cls.raw = randrange(-5e-6, 5e-6, size=size)

    @classmethod
    def tearDownClass(cls):
        del cls.raw, cls.columns, cls.rows, cls.meta, cls.df

    def setUp(self):
        self.spikes = self.df.copy()

    def tearDown(self):
        del self.spikes

    def test_init_noindex_nocolumns(self):
        spikes = SpikeDataFrame(self.raw, self.meta)
        self.assertRaises(AssertionError, SpikeDataFrame, self.raw, None)
        self.assertRaises(TypeError, SpikeDataFrame, self.raw)
        assert_frame_equal(spikes, self.df)

    def test_init_noindex_columns(self):
        from span.tdt.spikeglobals import ChannelIndex as columns
        spikes = SpikeDataFrame(self.raw, self.meta, columns=columns)
        self.assertRaises(AssertionError, SpikeDataFrame, self.raw, None)
        self.assertRaises(TypeError, SpikeDataFrame, self.raw)
        assert_frame_equal(spikes, self.df)

    def test_init_with_index_nocolumns(self):
        spikes = SpikeDataFrame(self.raw, self.meta, index=self.index)
        assert_frame_equal(spikes, self.df)

    def test_init_with_index_with_columns(self):
        from span.tdt.spikeglobals import ChannelIndex as columns
        spikes = SpikeDataFrame(self.raw, self.meta, index=self.index,
                                columns=columns)
        assert_frame_equal(spikes, self.df)

    def test_fs(self):
        fs = self.spikes.fs

        self.assertEqual(fs, self.meta.fs.max())
        assert_array_equal(fs, self.meta.fs)

        if hasattr(fs, 'dtype'):
            self.assert_(np.issubdtype(fs.dtype, np.floating))
        else:
            self.assertIsInstance(fs, float)

    def test_nchans(self):
        nchans = self.spikes.nchans
        cols = self.spikes.columns
        values = cols.levels[cols.names.index('channel')].values
        self.assertEqual(nchans, values.max() + 1)


class TestSpikeDataFrame(unittest.TestCase):
    def setUp(self):
        self.spikes = create_spike_df()
        self.threshes = 2e-5, 3e-5
        self.mses = 2.0, 3.0
        self.binsizes = 0, 1, np.random.randint(10, 1001)

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
        thr = sp.threshold(2.0 * std[np.random.randint(std.size)])
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

    @slow
    def test_bin(self):
        sp = self.spikes
        thr = sp.threshold(np.random.randint(1, 4) * sp.std())
        clr = sp.clear_refrac(thr)
        binsizes = -1, 0, 1, 1000
        reject_counts = -1, 0, 100
        dropnas = True, False
        args = itertools.product(binsizes, reject_counts, dropnas)

        for binsize, reject_count, dropna in args:
            if binsize > 0 and reject_count > 0:
                if reject_count > 0:
                    binned = sp.bin(clr, binsize, reject_count, dropna)
            else:
                self.assertRaises(AssertionError, sp.bin, clr, binsize,
                                  reject_count, dropna)

    @slow
    def test_fr(self):
        sp = self.spikes
        thr = sp.threshold(3 * sp.std())
        clr = sp.clear_refrac(thr)
        binned = sp.bin(clr, binsize=1000)
        args = itertools.product(sp.columns.names,
                                 xrange(len(sp.columns.names)), (0, 1),
                                 (True, False))

        for level_name, level_number, axis, sem in args:
            for t in (level_name, level_number):
                if axis:
                    if sem:
                        mfr, sm = sp.fr(binned, t, axis, sem)
                        self.assertEqual(mfr.shape, sm.shape)
                    else:
                        mfr = sp.fr(binned, t, axis, sem)

                else:
                    self.assertRaises(ValueError, sp.fr, binned, t, axis, sem)

    @slow
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

    # @nottest
    @slow
    def test_xcorr(self):
        thr = self.spikes.threshold(3 * self.spikes.std())
        clr = self.spikes.clear_refrac(thr, ms=2)
        binned = self.spikes.bin(clr, binsize=1000)

        maxlags = None, 1, 100
        detrends = detrend_mean, detrend_none, detrend_linear
        scale_types = None, 'none', 'normalize', 'unbiased', 'biased'
        sortlevel_s = ('channel i', 'channel j', 'shank i', 'shank j',
                       'side i', 'side j')
        sortlevel_i = xrange(len(sortlevel_s))
        dropnas = True, False
        nan_autos = True, False

        args = itertools.product(maxlags, detrends, scale_types, sortlevel_s,
                                 sortlevel_i, dropnas, nan_autos)

        for (maxlag, detrend, scale_type, level_s, level_i, dropna,
             nan_auto) in args:
            for level in (level_s, level_i):
                xc = self.spikes.xcorr(binned, maxlag, detrend, scale_type,
                                       level, dropna, nan_auto)


class TestLfpDataFrame(unittest.TestCase):
    pass


class TestCreateXCorrInds(unittest.TestCase):
    pass
