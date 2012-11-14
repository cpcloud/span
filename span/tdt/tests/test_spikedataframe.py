from future_builtins import map

import os
import unittest
import itertools

from nose.tools import nottest

import numpy as np
from numpy.random import randn
from numpy.testing.decorators import slow

import pandas as pd
from pandas import Series
from pandas.util.testing import assert_frame_equal, assert_series_equal

from span.tdt.tank import PandasTank
from span.tdt.spikedataframe import (SpikeDataFrameBase, SpikeDataFrame,
                                     SpikeGroupedDataFrame)
from span.utils import detrend_none, detrend_mean, detrend_linear
from span.testing import assert_all_dtypes, randrange


class TestSpikeGrouper(unittest.TestCase):
    pass


class TestChannelGetter(unittest.TestCase):
    pass


class TestShankGetter(unittest.TestCase):
    pass


class TestSideGetter(unittest.TestCase):
    pass


class TestSpikeGroupedDataFrame(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.x = randn(10, 12)

    @classmethod
    def tearDownClass(cls):
        del cls.x

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


# @nottest
class TestSpikeDataFrameBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tankname = os.path.join(os.path.expanduser('~'), 'Data', 'xcorr_data',
                                'Spont_Spikes_091210_p17rat_s4_657umV')
        tn = os.path.join(tankname, os.path.basename(tankname))
        cls.tank = PandasTank(tn)
        tsq = cls.tank.tsq
        cls.meta = tsq

        nsamples, _ = tsq.shape[0], tsq.size.unique().max()
        size = nsamples * tsq.shape[1], tsq.channel.nunique()

        cls.rows, cls.columns = size
        cls.raw = randrange(-5e-6, 5e-6, size=size)

        cls.threshes = 2e-5, 3e-5
        cls.mses = 2.0, 3.0
        cls.binsizes = 0, 1, np.random.randint(10, 1001)

    def clean_up_tsq(self):
        pass

    @classmethod
    def tearDownClass(cls):
        del cls.binsizes, cls.mses, cls.threshes, cls.meta, cls.raw
        del cls.rows, cls.columns, cls.tank

    def setUp(self):
        self.spikes = SpikeDataFrame(self.raw, self.meta)
        self.clean_up_tsq()

    def tearDown(self):
        del self.spikes

    def test_init_noindex(self):
        spikes = SpikeDataFrame(self.raw, self.meta)
        self.assertRaises(AssertionError, SpikeDataFrame, self.raw, None)
        self.assertRaises(TypeError, SpikeDataFrame, self.raw)

    def test_init_with_index(self):
        assert False

    def test_fs(self):
        fs = self.spikes.fs

        self.assertEqual(fs, self.meta.fs.max())
        self.assertIn(fs, self.meta.fs)

        if hasattr(fs, 'dtype'):
            self.assert_(np.issubdtype(fs.dtype, np.floating))
        else:
            self.assertIsInstance(fs, float)

    def test_nchans(self):
        nchans = self.spikes.nchans
        cols = self.spikes.columns
        values = cols.levels[cols.names.index('channel')].values
        self.assertEqual(nchans, values.max() + 1)

    def test_indices_and_groups(self):
        inds = 'channel', 'shank', 'side'
        for ind in inds:
            self.assert_(hasattr(self.spikes, ind + '_indices'))
            self.assert_(hasattr(self.spikes, ind + '_group'))
            indices = getattr(self.spikes, ind + '_indices')
            grp = getattr(self.spikes, ind + '_group')
            self.assertIsInstance(indices, (pd.DataFrame, pd.Series))
            self.assertIsInstance(grp, (pd.core.groupby.DataFrameGroupBy,
                                        pd.core.groupby.SeriesGroupBy))


class TestSpikeDataFrame(TestSpikeDataFrameBase):
    def test_threshold(self):
        shp = self.spikes.shape
        for thresh in self.threshes:
            threshed = self.spikes.threshold(thresh)
            assert_all_dtypes(threshed, np.bool_)
            self.assertTupleEqual(threshed.shape, shp)

    @slow
    def test_bin(self):
        max_sample = self.spikes.index[-1]
        for arg_set in itertools.product(self.threshes, self.mses,
                                         self.binsizes):
            thresh, ms, binsize = arg_set
            binned = self.spikes.bin(thresh, ms, binsize)
            bin_samples = int(np.floor(binsize * self.spikes.fs / 1e3))
            self.assertTupleEqual(binned.shape, (max_sample / bin_samples,
                                                 self.spikes.shape[1]))
            assert_all_dtypes(binned, np.int64)

    @slow
    def test_fr(self):
        index = self.spikes.index
        levels = index.names
        axes = 0, 1, None
        arg_sets = itertools.product(self.threshes, levels, axes, self.binsizes,
                                     self.mses)
        raises = {0: ValueError, None: KeyError}
        for arg_set in arg_sets:
            thresh, level, axis, binsize, ms = arg_set
            try:
                self.assertRaises(raises[axis], self.spikes.fr, thresh, level,
                                  axis, binsize, ms)
            except KeyError:
                fr, sem = self.spikes.fr(thresh, level, axis, binsize, ms)
                sz = index.levels[levels.index(level)].size
                self.assertEqual(fr.size, sz)
                self.assertEqual(sem.size, sz)

    def test_refrac_window(self):
        args = np.arange(10)
        r = list(map(type, [self.spikes.refrac_window(arg) for arg in args]))
        self.assertListEqual(r, list(itertools.repeat(int, len(r))))

    @slow
    def test_clear_refrac(self):
        for arg_set in itertools.product(self.threshes, self.mses):
            thresh, ms = arg_set
            cleared = self.spikes.clear_refrac(thresh, ms)
            assert_all_dtypes(cleared, np.bool_)

    def test__constructor(self):
        s = self.spikes
        ind = self.spikes.index
        cols = self.spikes.columns
        s_new = self.spikes._constructor(s.values, index=ind, columns=cols)
        self.assertIsInstance(s_new, pd.DataFrame)
        self.assertIsInstance(s_new, SpikeDataFrame)
        self.assertIsInstance(s, type(s_new))

    @slow
    def test_xcorr(self):
        maxlags = 50, 100, None
        detrends = detrend_mean, detrend_none, detrend_linear
        scales = None, 'normalize', 'unbiased', 'biased'
        reject_counts = 0, 1, 10, 100, 1000
        arg_sets = itertools.product(self.threshes, self.mses, self.binsizes,
                                     maxlags, detrends, scales, reject_counts)
        for arg_set in arg_sets:
            thresh, ms, binsize, maxlag, detrend, scale, reject_count = arg_set
            xc, _ = self.spikes.xcorr(thresh, ms, binsize, maxlag, detrend,
                                      scale, reject_count)
            self.assertTupleEqual(xc.values.shape, (self.spikes.nchans ** 2,
                                                    2 * maxlag - 1))
            assert_all_dtypes(xc, np.float64)

        self.assertRaises(AssertionError, self.spikes.xcorr, maxlags=int(1e10))


@slow
class TestLfpDataFrame(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tankname = os.path.join(os.path.expanduser('~'), 'Data', 'xcorr_data',
                                'Spont_Spikes_091210_p17rat_s4_657umV')
        tn = os.path.join(tankname, os.path.basename(tankname))
        cls.tank = PandasTank(tn)
        cls.rows, cls.columns = cls.tank.lfps.shape
        cls.lfps = cls.tank.lfps
        cls.meta = cls.lfps.meta
        cls.threshes = 2e-5,
        cls.mses = 2.0, 3.0
        cls.binsizes = 0, 1, 1000

    def test_fs(self):
        fs = self.lfps.fs

    @classmethod
    def tearDownClass(cls):
        del cls.binsizes, cls.mses, cls.threshes, cls.meta, cls.lfps
        del cls.rows, cls.columns, cls.tank


class TestCreateXCorrInds(unittest.TestCase):
    pass
