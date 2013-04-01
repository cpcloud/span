import unittest
import nose

from numpy.random import permutation, randint
from numpy import ndarray

from span.tdt.recording import ElectrodeMap, distance_map
import pandas as pd
from pandas import Series


class TestDistanceMap(unittest.TestCase):
    def test_distance_map(self):
        nshanks = randint(1, 10)
        electrodes_per_shank = randint(1, 10)
        within_shank = randint(20, 100)
        metric = 'wminkowski'
        p = 2.0

        if nshanks == 1:
            between_shank = 0
            dm = distance_map(nshanks, electrodes_per_shank, within_shank,
                              between_shank, metric, p)
            self.assertRaises(
                AssertionError, distance_map,
                nshanks, electrodes_per_shank, within_shank,
                1, metric, p)
        else:
            between_shank = randint(20, 100)
            dm = distance_map(nshanks, electrodes_per_shank, within_shank,
                              between_shank, metric, p)
        self.assertIsInstance(dm, ndarray)


@nose.tools.nottest
class TestElectrodeMap(unittest.TestCase):
    def tearDown(self):
        del self.p, self.metric, self.map_, self.between_shank
        del self.within_shank, self.newshape, self.channels_per_shank
        del self.nshank, self.nchannel

    def test___bytes__(self):
        electrode_map = ElectrodeMap(self.map_, self.within_shank,
                                     self.between_shank)
        b = electrode_map.__bytes__()
        self.assertIsInstance(b, bytes)
        self.assertEqual(bytes(electrode_map), b)

    def test___init__(self):
        # these should succeed
        self.em = ElectrodeMap(self.map_)
        self.em = ElectrodeMap(self.map_, 50, 125)
        self.em = ElectrodeMap(self.map_.ravel(), 60)

        # these should fail
        self.assertRaises(AssertionError, ElectrodeMap, self.map_.ravel(), 60,
                          125)
        self.assertRaises(AssertionError, ElectrodeMap, self.map_, None, 125)
        self.assertRaises(AssertionError, ElectrodeMap, self.map_, 60, None)
        self.assertRaises(AssertionError, ElectrodeMap, self.map_.ravel(),
                          None, 125)

    def test___unicode__(self):
        electrode_map = ElectrodeMap(self.map_, self.within_shank,
                                     self.between_shank)
        u = electrode_map.__unicode__()
        self.assertIsInstance(u, unicode)
        self.assertEqual(unicode(electrode_map), u)

    def test_channel(self):
        electrode_map = ElectrodeMap(self.map_, self.within_shank,
                                     self.between_shank)
        self.assertIsInstance(electrode_map.channel, ndarray)

    def test_distance_map(self):
        electrode_map = ElectrodeMap(self.map_, self.within_shank,
                                     self.between_shank)
        dm = electrode_map.distance_map(self.metric, self.p)
        self.assertIsInstance(dm, Series)

    def test_nchannel(self):
        electrode_map = ElectrodeMap(self.map_, self.within_shank,
                                     self.between_shank)
        expected = self.nchannel
        self.assertEqual(expected, electrode_map.nchannel)

    def test_nshank(self):
        electrode_map = ElectrodeMap(self.map_, self.within_shank,
                                     self.between_shank)
        expected = self.nshank
        self.assertEqual(expected, electrode_map.nshank)

    def test_original(self):
        electrode_map = ElectrodeMap(self.map_, self.within_shank,
                                     self.between_shank)
        orig = electrode_map.original
        self.assertIsInstance(orig, pd.DataFrame)

    def test_raw(self):
        electrode_map = ElectrodeMap(self.map_, self.within_shank,
                                     self.between_shank)
        r = electrode_map.raw
        self.assertIsInstance(r, pd.DataFrame)

    def test_shank(self):
        electrode_map = ElectrodeMap(self.map_, self.within_shank,
                                     self.between_shank)
        self.assertIsInstance(electrode_map.shank, ndarray)


@nose.tools.istest
class TestElectrodeMap1D(TestElectrodeMap):
    def setUp(self):
        self.nchannel, self.nshank = 16, 1
        self.channels_per_shank = self.nchannel // self.nshank
        self.newshape = self.channels_per_shank, self.nshank
        self.within_shank = randint(20, 101)
        self.between_shank = 0
        self.map_ = permutation(self.nchannel).reshape(self.newshape)
        self.metric = 'wminkowski'
        self.p = 2.0


@nose.tools.istest
class TestElectrodeMap2D(TestElectrodeMap):
    def setUp(self):
        self.nchannel, self.nshank = 16, 4
        self.channels_per_shank = self.nchannel // self.nshank
        self.newshape = self.channels_per_shank, self.nshank
        self.within_shank = randint(20, 101)
        self.between_shank = randint(20, 101)
        self.map_ = permutation(self.nchannel).reshape(self.newshape)
        self.metric = 'wminkowski'
        self.p = 2.0


if __name__ == '__main__':
    unittest.main()
