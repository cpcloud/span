import os
import unittest
import string
import datetime

import pandas as pd

from span.tdt.tank import TdtTankBase, PandasTank
from span.tdt import SpikeDataFrame


class TestGetFirstMatch(unittest.TestCase):
    def test_get_first_match(self):
        assert False


class TestMatchInt(unittest.TestCase):
    def test_match_int(self):
        assert False


class TestTdtTankBase(unittest.TestCase):
    def test___init__(self):
        self.assertRaises(TypeError, TdtTankBase, pd.util.testing.rands(10))


class TestPandasTank(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tankname = os.path.join(os.path.expanduser('~'), 'Data', 'xcorr_data',
                                'Spont_Spikes_091210_p17rat_s4_657umV')
        tn = os.path.join(tankname, os.path.basename(tankname))
        cls.tank = PandasTank(tn)

    @classmethod
    def tearDownClass(cls):
        del cls.tank

    def test_age(self):
        assert hasattr(self.tank, 'age')
        self.assertIsInstance(self.tank.age, (type(None), int))

    def test_site(self):
        assert hasattr(self.tank, 'site')
        self.assertIsInstance(self.tank.site, (type(None), int))

    def test_read_tev(self):
        names = 'Spik', 'LFPs'
        for name in names:
            tev = self.tank._read_tev(name)()
            self.assertIsNotNone(tev)
            self.assertIsInstance(tev, SpikeDataFrame)

    def test_read_tsq(self):
        names = 'Spik', 'LFPs'
        for name in names:
            tsq, _ = self.tank._read_tsq(name)()
            self.assertIsNotNone(tsq)
            self.assertIsInstance(tsq, pd.DataFrame)

    def test_tsq(self):
        names = 'Spik', 'LFPs'
        for name in names:
            self.assertIsNotNone(self.tank.tsq(name))

    def test_stsq(self):
        self.assertIsNotNone(self.tank.stsq)

    def test_ltsq(self):
        self.assertIsNotNone(self.tank.ltsq)

    def test_tev(self):
        names = 'Spik', 'LFPs'
        for name in names:
            self.assertIsNotNone(self.tank.tev(name))

    def test_spikes(self):
        self.assertIsNotNone(self.tank.spikes)

    def test_lfps(self):
        self.assertIsNotNone(self.tank.lfps)
