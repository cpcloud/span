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
        between = np.arange(start=0, stop=400, step=25)
        within = between.copy()
        metrics = 'wminkowski',
        ps = np.arange(10) + 1
        arg_sets = it.product(nshanks, elecs, between, within, metrics, ps)
        for arg_set in arg_sets:
            nsh, nelec, btwn, wthn, metr, p = arg_set
            if not nsh or not nelec:
                self.assertRaises(AssertionError, distance_map, nsh, nelec, btwn,
                                  wthn, metr, p)
            dm = distance_map(nsh, nelec, btwn, wthn, metr, p)


class TestElectrodeMap(unittest.TestCase):
    def test_1d_map(self):
        assert False

    def test_2d_map(self):
        assert False

    def test_distance_map_1d(self):
        assert False

    def test_distance_map_2d(self):
        assert False

    def test_show(self):
        assert False
