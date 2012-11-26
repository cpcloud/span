import unittest
import itertools as it

import numpy as np

from span.tdt import distance_map, ElectrodeMap
from span.testing import create_spike_df


class TestDistanceMap(unittest.TestCase):
    def test_distance_map(self):
        nshanks = np.r_[:9]
        elecs = nshanks.copy()

        between = np.r_[25:150:25]
        within = between.copy()

        metrics = 'wminkowski',
        ps = 1, 2, np.inf

        arg_sets = it.product(nshanks, elecs, within, between, metrics, ps)

        for arg_set in arg_sets:
            nsh, nelec, wthn, btwn, metr, p = arg_set

            if not (nsh and nelec):
                self.assertRaises(AssertionError, distance_map, nsh, nelec,
                                  wthn, btwn, metr, p)
            else:
                if nelec == 1:
                    nsh = 1

                if nsh == 1:
                    btwn = 0

                if nsh > 1 and nelec > 1:
                    dm = distance_map(nsh, nelec, btwn, wthn, metr, p)
                    self.assertEqual(dm.size, (nsh * nelec) ** 2)


class TestElectrodeMap(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.b = np.arange(0, 101, 25)
        cls.w = cls.b.copy()
        cls.orders = None, 'lm', 'ml'
        cls.base_indices = 0, 1
        cls.nelecs = np.arange(64) + 1

    @classmethod
    def tearDownClass(cls):
        del cls.nelecs, cls.base_indices, cls.orders, cls.w, cls.b

    def test_1d_map(self):
        b, w, nelecs = self.b, np.zeros_like(self.b), self.nelecs
        arg_sets = it.product(b, w, nelecs, self.orders, self.base_indices)

        for bb, ww, n, order, bi in arg_sets:
            a = np.random.randint(1, n + 1, size=n)

            if order is not None and a.ndim == 2:
                em = ElectrodeMap(a, order, bi)

                if ww and bb:
                    dm = em.distance_map(1, ww, bb)
                    self.assertIsNotNone(dm)
                else:
                    self.assertRaises(AssertionError, em.distance_map, 1, ww,
                                      bb)

    def test_2d_map(self):
        assert False

    def test_distance_map_1d(self):
        assert False

    def test_distance_map_2d(self):
        assert False

    def test_show(self):
        assert False


class TestDistanceMapWithCrossCorrelation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sp = create_spike_df()
        thr = sp.threshold(4 * sp.std())
        clr = sp.clear_refrac(thr)
        binned = sp.bin(clr, binsize=10)
        cls.xc = sp.xcorr(binned, maxlags=10)

        rawmap = np.array([1, 3, 2, 6,
                           7, 4, 8, 5,
                           13, 10, 12, 9,
                           14, 16, 11, 15]).reshape(4, 4)
        cls.elecmap = ElectrodeMap(rawmap, order='lm')
        cls.dm = cls.elecmap.distance_map(50, 125)

    def test_set_index(self):
        xcc = self.xc.T.set_index(self.dm, append=True).T
        lag0_tmp = xcc.ix[0].dropna().sortlevel(level=6)
        lag0 = lag0_tmp.reset_index(level=range(6), drop=True)
        self.assertIsNotNone(lag0)
