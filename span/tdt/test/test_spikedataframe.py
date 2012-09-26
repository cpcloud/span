import os
import unittest
import itertools
import operator

import numpy as np
from numpy.testing import assert_array_equal, assert_allclose

import pandas as pd

from span.tdt.tank import PandasTank
from span.utils import detrend_none, detrend_mean, detrend_linear


def assert_all_dtypes(df, dtype, msg='dtypes not all the same'):
    assert all(dt == dtype for dt in df.dtypes), msg


class TestSpikeDataFrameAbstractBase(unittest.TestCase):
    def test_init(self):
        self.assertRaises(TypeError, SpikeDataFrameAbstractBase,
                          pd.DataFrame(np.random.randn(10, 13)),
                          pd.DataFrame(np.random.rand(1080, 17)))


class TestSpikeDataFrameBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tankname = os.path.join(os.path.expanduser('~'), 'Data', 'xcorr_data',
                                'Spont_Spikes_091210_p17rat_s4_657umV')
        tn = os.path.join(tankname, os.path.basename(tankname))
        tank = PandasTank(tn)
        cls.spikes = tank.spikes
        cls.meta = cls.spikes.meta
        cls.threshes = 1e-5, 2e-5, 3e-5, 4e-5
        cls.mses = 2.0, 3.0, 4.0, 5.0
        cls.binsizes = 1, 10, 100, 1000

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
        cols = self.spikes.channels.columns
        values = cols.levels[cols.names.index('channel')].values
        self.assertEqual(nchans, values.max() + 1)

    def test__channels(self):
        chn = self.spikes._channels
        self.assert_(callable(chn))

    def test_channels(self):
        chn = self.spikes.channels
        cols = self.spikes.channels.columns
        values = cols.levels[cols.names.index('channel')]
        self.assertRaises(ValueError, operator.add, values.max(), 1)
        self.assertEqual(self.spikes.nchans, values.values.max() + 1)

    def test_indices_and_groups(self):
        inds = 'channel', 'shank', 'side'
        for ind in inds:
            assert hasattr(self.spikes, ind + '_indices')
            assert hasattr(self.spikes, ind + '_group')
            indices = getattr(self.spikes, ind + '_indices')
            grp = getattr(self.spikes, ind + '_group')
            self.assertIsInstance(indices, (pd.DataFrame, pd.Series))
            self.assertIsInstance(grp, (pd.DataFrame, pd.Series))

    def test_values(self):
        vals = self.spikes.raw
        self.assertTupleEqual(vals.shape, self.spikes.channels.shape)
        self.assertEqual(self.spikes.channels.values.dtype, vals.dtype)


class TestSpikeDataFrame(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tankname = os.path.join(os.path.expanduser('~'), 'Data', 'xcorr_data',
                                'Spont_Spikes_091210_p17rat_s4_657umV')
        tn = os.path.join(tankname, os.path.basename(tankname))
        tank = PandasTank(tn)
        cls.spikes = tank.spikes
        cls.meta = cls.spikes.meta
        cls.threshes = 1e-5, 2e-5, 3e-5, 4e-5
        cls.mses = 2.0, 3.0, 4.0, 5.0
        cls.binsizes = 1, 10, 100, 1000

    def test_threshold(self):
        shp = self.spikes.channels.shape
        for thresh in self.threshes:
            threshed = self.spikes.threshold(thresh)
            assert_all_dtypes(threshed, np.bool_)
            self.assertTupleEqual(threshed.shape, shp)

    def test_bin(self):
        max_sample = self.spikes.channels.index[-1]
        for arg_set in itertools.product(self.threshes, self.mses,
                                         self.binsizes):
            thresh, ms, binsize = arg_set
            binned = self.spikes.bin(thresh, ms, binsize)
            bin_samples = int(np.floor(binsize * self.spikes.fs / 1e3))
            self.assertTupleEqual(binned.shape, (max_sample / bin_samples,
                                                 self.spikes.channels.shape[1]))
            assert_all_dtypes(binned, np.int64)

    def test_refrac_window(self):
        args = np.arange(100)
        r = [self.spikes.refrac_window(arg) for arg in args]
        assert_all_dtypes(r, int)

    def test_cleared(self):
        for arg_set in itertools.product(self.threshes, self.mses):
            thresh, ms = arg_set
            cleared = self.spikes.cleared(thresh, ms)
            assert_all_dtypes(cleared, np.bool_)

    def test__constructor(self):
        s = self.spikes
        ind = self.spikes.index
        cols = self.spikes.columns
        s_new = self.spikes._constructor(s.values, index=ind, columns=cols)
        self.assertIsInstance(s, type(s_new))
        self.assertIsInstance(s, pd.DataFrame)
        self.assertIsInstance(s_new, pd.DataFrame)

    def test_xcorr(self):
        maxlags = 50, 100, 150, 200, 250, 300, None
        detrends = detrend_mean, detrend_none, detrend_linear
        scales = None, 'none', 'normalize', 'unbiased', 'biased'
        arg_sets = itertools.product(self.threshes, self.mses, self.binsizes,
                                     maxlags, detrends, scales)
        for arg_set in arg_sets:
            thresh, ms, binsize, maxlag, detrend, scale = arg_set
            xc = self.spikes.xcorr(thresh, ms, binsize, maxlag, detrend, scale)
            self.assertTupleEqual(xc.values.shape, (self.spikes.nchans ** 2,
                                                    2 * maxlag - 1))
            assert_all_dtypes(xc, np.float64)


class TestDetrend(unittest.TestCase):
    def test_detrend_none(self):
        x = np.random.randn(10, 11, 12)
        dtx = detrend_none(x)
        assert_array_equal(x, dtx)

    def test_detrend_mean(self):
        x = np.random.randn(10, 9, 8, 7)
        dtx = detrend_mean(x)
        assert_array_equal(x, x - x.mean())
        assert_allclose(dtx.mean(), 0)

    def test_detrend_mean_dataframe(self):
        x = pd.DataFrame(np.random.randn(10, 13))
        dtx = detrend_mean(x)
        m = dtx.mean()
        assert_allclose(m.values.squeeze(), np.zeros(m.shape))

    def test_detrend_linear(self):
        x = np.random.randn(10, 13)
        dtx = detrend_linear(x)
        assert_allclose(dtx.mean(), 0)
        assert_allclose(dtx.std(), 1)

    def test_detrend_linear_dataframe(self):
        x = pd.DataFrame(np.random.randn(10, 13))
        dtx = detrend_linear(x)
        m = dtx.mean()
        s = dtx.std()
        assert_allclose(m.values.squeeze(), np.zeros(m.shape))
        assert_allclose(s.values.squeeze(), np.ones(s.shape))