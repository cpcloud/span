from future_builtins import map, zip

import unittest
import itertools as it

import numpy as np
import pandas as pd

from span.tdt import distance_map, ElectrodeMap


class TestDistanceMap(unittest.TestCase):
    def test_distance_map(self):
        nshanks = np.arange(9)
        elecs = np.arange(11)
        between = np.arange(start=0, stop=100, step=25)
        within = between.copy()
        metrics = 'wminkowski',
        ps = np.arange(10) + 1

        arg_sets = it.product(nshanks, elecs, within, between, metrics, ps)

        for arg_set in arg_sets:
            nsh, nelec, wthn, btwn, metr, p = arg_set

            if not nsh or not nelec:
                self.assertRaises(AssertionError, distance_map, nsh, nelec, wthn,
                                  btwn, metr, p)
            else:
                if nelec == 1:
                    nsh = 1

                if nsh == 1:
                    btwn = 0
                    
                dm = distance_map(nsh, nelec, btwn, wthn, metr, p)
                self.assertEqual(dm.size, (nsh * nelec) ** 2)
                

class TestElectrodeMap(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.b = np.arange(0, 100, 25)
        cls.w = cls.b.copy()
        cls.orders = None, 'lm', 'ml'
        cls.base_indices = 0, 1
        cls.nelecs = np.arange(64) + 1

    def test_1d_map(self):
        b, w, nelecs = self.b, np.zeros_like(self.b), self.nelecs
        arg_sets = it.product(b, w, nelecs, self.orders, self.base_indices)
        
        for arg_set in arg_sets:
            bb, ww, n, order, bi = arg_set
            em = ElectrodeMap(np.random.randint(n, size=n), n, order, bi)
            dm = em.distance_map(1, ww, bb)

    def test_2d_map(self):
        assert False

    def test_distance_map_1d(self):
        assert False

    def test_distance_map_2d(self):
        assert False

    def test_show(self):
        assert False
