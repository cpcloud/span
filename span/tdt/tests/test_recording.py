import unittest
import itertools as itools

import numpy as np

from span.tdt import distance_map, ElectrodeMap
from span.testing import create_spike_df, slow


@slow
class TestDistanceMap(unittest.TestCase):
    def test_distance_map(self):
        nshanks = np.r_[:9]
        elecs = nshanks.copy()

        between = np.r_[25:150:25]
        within = between.copy()

        metrics = 'wminkowski',
        ps = 1, 2, np.inf

        arg_sets = itools.product(nshanks, elecs, within, between, metrics, ps)

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
    def setUp(self):
        self.between_shank = np.arange(25, 101, 25)
        self.within_shank = self.between_shank.copy()
        self.base_indices = 0, 1
        self.nelecs = np.arange(64) + 1

    def tearDown(self):
        del self.nelecs, self.base_indices, self.within_shank
        del self.between_shank

    def test_constructor(self):
        self.assertRaises(AssertionError, ElectrodeMap, [1, 2], 'ml')
        em = ElectrodeMap([[1, 2], [3, 4], [5, 6]], 'ml')

    def test_1d_map(self):
        args = itools.product(self.nelecs, (None,), self.base_indices)
        for nelec, order, base_index in args:
            elecs = np.random.randint(nelec, size=nelec)
            em = ElectrodeMap(elecs, order, base_index)

    def test_2d_map(self):
        orders = None, 'ml', 'lm'
        args = itools.product(self.nelecs, orders, self.base_indices)

        for nelec, order, base_index in args:
            nshanks = np.random.randint(2, 10)
            elecs = np.random.randint(nelec, size=(nelec, nshanks))

            if nshanks % 2 == 0:
                em = ElectrodeMap(elecs, order, base_index)
            else:
                self.assertRaises(AssertionError, ElectrodeMap, elecs, order,
                                  base_index)

    def test_distance_map_1d(self):
        assert False

    def test_distance_map_2d(self):
        assert False

    def test_nshanks(self):
        def _1d():
            args = itools.product(self.nelecs, (None,), self.base_indices)
            for nelec, order, base_index in args:
                elecs = np.random.randint(nelec, size=nelec)
                em = ElectrodeMap(elecs, order, base_index)
                nc = em.nchans

        def _2d():
            orders = None, 'ml', 'lm'
            args = itools.product(self.nelecs, orders, self.base_indices)

            for nelec, order, base_index in args:
                nshanks = np.random.randint(2, 10)
                elecs = np.random.randint(nelec, size=(nelec, nshanks))
                em = ElectrodeMap(elecs, order, base_index)
                ns = em.nshanks

        _1d()
        _2d()

    def test_nchans(self):
        def _1d():
            args = itools.product(self.nelecs, (None,), self.base_indices)
            for nelec, order, base_index in args:
                elecs = np.random.randint(nelec, size=nelec)
                em = ElectrodeMap(elecs, order, base_index)
                nc = em.nchans

        def _2d():
            orders = None, 'ml', 'lm'
            args = itools.product(self.nelecs, orders, self.base_indices)

            for nelec, order, base_index in args:
                nshanks = np.random.randint(2, 10)
                elecs = np.random.randint(nelec, size=(nelec, nshanks))
                em = ElectrodeMap(elecs, order, base_index)
                ns = em.nchans

        _1d()
        _2d()

    def test_one_based(self):
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
