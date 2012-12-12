import unittest
import numbers
import itertools as itools

import numpy as np
from numpy.random import randint

from span.tdt import distance_map, ElectrodeMap
from span.testing import create_spike_df


class TestDistanceMap(unittest.TestCase):
    def test_distance_map(self):
        nshanks = np.r_[:9]
        elecs = nshanks.copy()

        between = np.array([25, 50])
        within = between.copy()

        metrics = 'wminkowski',
        ps = 1, 2, np.inf

        arg_sets = itools.product(nshanks, elecs, within, between, metrics, ps)

        def _tester(nsh, nelec, wthn, btwn, metr, p):
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

        for arg_set in arg_sets:
            _tester(*arg_set)


class TestElectrodeMap(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.b = np.arange(0, 101, 25)
        cls.w = cls.b.copy()
        cls.nelecs = np.arange(64) + 1

    @classmethod
    def tearDownClass(cls):
        del cls.nelecs, cls.w, cls.b

    def test_nchans(self):
        b, w, nelecs = self.b, np.zeros_like(self.b), self.nelecs
        arg_sets = itools.product(b, w, nelecs)

        for bb, ww, nelec, in arg_sets:
            nshanks = randint(1, 9)
            a = randint(1, nelec + 1, size=(nelec, nshanks))
            em = ElectrodeMap(a)
            self.assert_(hasattr(em, 'nchans'))
            self.assertIsInstance(em.nchans, numbers.Integral)

    def test_1d_map(self):
        b, w, nelecs = self.b, np.zeros_like(self.b), self.nelecs
        arg_sets = itools.product(b, w, nelecs)

        for bb, ww, n in arg_sets:
            a = randint(1, n + 1, size=n)

            em = ElectrodeMap(a)

            if ww and bb:
                dm = em.distance_map(1, ww, bb)
                self.assertIsNotNone(dm)
            else:
                self.assertRaises(AssertionError, em.distance_map, 1, ww, bb)

    def test_2d_map(self):
        assert False

    def test_distance_map_1d(self):
        assert False

    def test_distance_map_2d(self):
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
        cls.elecmap = ElectrodeMap(rawmap)
        cls.dm = cls.elecmap.distance_map(50, 125)

    def test_set_index(self):
        xcc = self.xc.T.set_index(self.dm, append=True).T
        lag0_tmp = xcc.ix[0].dropna().sortlevel(level=6)
        lag0 = lag0_tmp.reset_index(level=range(6), drop=True)
        self.assertIsNotNone(lag0)
