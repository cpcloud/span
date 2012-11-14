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
from span.utils import detrend_none, detrend_mean, detrend_linear, fromtimestamp
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


class TestSpikeDataFrameBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tankname = os.path.join(os.path.expanduser('~'), 'Data', 'xcorr_data',
                                'Spont_Spikes_091210_p17rat_s4_657umV')
        tn = os.path.join(tankname, os.path.basename(tankname))
        cls.tank = PandasTank(tn)
        tsq = cls.tank.stsq
        cls.meta = tsq.reset_index(drop=True)
        cls.meta.timestamp = fromtimestamp(cls.meta.timestamp)

        nsamples, _ = tsq.shape[0], tsq.size.unique().max()
        size = nsamples * tsq.shape[1], tsq.channel.nunique()

        cls.rows, cls.columns = size
        cls.raw = randrange(-5e-6, 5e-6, size=size)

        cls.threshes = 2e-5, 3e-5
        cls.mses = 2.0, 3.0
        cls.binsizes = 0, 1, np.random.randint(10, 1001)

    @classmethod
    def tearDownClass(cls):
        del cls.binsizes, cls.mses, cls.threshes, cls.meta, cls.raw
        del cls.rows, cls.columns, cls.tank

    def setUp(self):
        from span.tdt.spikeglobals import ChannelIndex as columns
        self.index = self.create_date_range()
        self.spikes = SpikeDataFrame(self.raw, self.meta, index=self.index,
                                     columns=columns)

    def tearDown(self):
        del self.spikes, self.index

    def create_date_range(self):
        us_per_sample = (round(1e6 / self.meta.fs.unique().max()) *
                         pd.datetools.Micro())
        ts = self.meta.timestamp[0]
        index = pd.date_range(ts, periods=self.rows, freq=us_per_sample,
                              name='time', tz='US/Eastern')

    def test_init_noindex_nocolumns(self):
        spikes = SpikeDataFrame(self.raw, self.meta)
        self.assertRaises(AssertionError, SpikeDataFrame, self.raw, None)
        self.assertRaises(TypeError, SpikeDataFrame, self.raw)

    def test_init_noindex_columns(self):
        from span.tdt.spikeglobals import ChannelIndex as columns
        spikes = SpikeDataFrame(self.raw, self.meta, columns=columns)
        self.assertRaises(AssertionError, SpikeDataFrame, self.raw, None)
        self.assertRaises(TypeError, SpikeDataFrame, self.raw)

    def test_init_with_index_nocolumns(self):
        spikes = SpikeDataFrame(self.raw, self.meta, index=self.index)

    def test_init_with_index_with_columns(self):
        from span.tdt.spikeglobals import ChannelIndex as columns
        spikes = SpikeDataFrame(self.raw, self.meta, index=self.index,
                                columns=columns)

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


class TestSpikeDataFrame(TestSpikeDataFrameBase):
    def test_threshold(self):
        assert False

    @slow
    def test_bin(self):
        assert False

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

    @slow
    def test_xcorr(self):
        assert False


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
