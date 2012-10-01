from future_builtins import map

import os
import unittest
import itertools
import operator

from nose.tools import nottest

import numpy as np
from numpy.testing import assert_array_equal
from numpy.testing.decorators import slow

import pandas as pd

from span.tdt.tank import PandasTank
from span.tdt.spikedataframe import SpikeDataFrameAbstractBase
from span.utils import detrend_none, detrend_mean, detrend_linear


def assert_all_dtypes(df, dtype, msg='dtypes not all the same'):
    assert all(dt == dtype for dt in df.dtypes), msg


def get_unified_dytpe(df, dtype):
    dtypes = df.dtypes
    expected = pd.Series(list(itertools.repeat(dtype, dtypes.size)))
    return dtype if np.array_equal(dtypes, expected) else None


@nottest
class TestSpikeDataFrameAbstractBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tankname = os.path.join(os.path.expanduser('~'), 'Data', 'xcorr_data',
                                'Spont_Spikes_091210_p17rat_s4_657umV')
        tn = os.path.join(tankname, os.path.basename(tankname))
        tank = PandasTank(tn)
        rows, _ = tank.spikes.shape
        cls.spikes = tank.spikes
        cls.meta = cls.spikes.meta
        cls.threshes = 2e-5, 3e-5
        cls.mses = 2.0, 3.0
        cls.binsizes = 1, 10, 100

    def test_init(self):
        self.assertRaises(TypeError, SpikeDataFrameAbstractBase,
                          pd.DataFrame(np.random.randn(10, 13)),
                          pd.DataFrame(np.random.rand(108, 17)))


@nottest
@slow
class TestSpikeDataFrameBase(TestSpikeDataFrameAbstractBase):
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

    @slow
    def test_channels(self):
        chn = self.spikes.channels
        cols = chn.columns
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
            self.assertIsInstance(grp, (pd.core.groupby.DataFrameGroupBy,
                                        pd.core.groupby.SeriesGroupBy))


@nottest
@slow
class TestSpikeDataFrame(TestSpikeDataFrameAbstractBase):
    def test_threshold(self):
        shp = self.spikes.channels.shape
        for thresh in self.threshes:
            threshed = self.spikes.threshold(thresh)
            assert_all_dtypes(threshed, np.bool_)
            self.assertTupleEqual(threshed.shape, shp)

    @slow
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
        self.assertIsInstance(s_new, pd.DataFrame)
        self.assertIsInstance(s, type(s_new))

    @slow
    def test_xcorr(self):
        maxlags = 200, 300, None
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


@slow
class TestLfpDataFrame(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tankname = os.path.join(os.path.expanduser('~'), 'Data', 'xcorr_data',
                                'Spont_Spikes_091210_p17rat_s4_657umV')
        tn = os.path.join(tankname, os.path.basename(tankname))
        tank = PandasTank(tn)
        rows, _ = tank.lfps.shape
        cls.lfps = tank.lfps
        cls.meta = cls.lfps.meta
        cls.threshes = 2e-5, 3e-5
        cls.mses = 2.0, 3.0
        cls.binsizes = 1, 10, 100

    def test_fs(self):
        fs = self.lfps.fs
