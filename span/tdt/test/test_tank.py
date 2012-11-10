import os
import unittest
import string
import datetime

import pandas as pd

from span.tdt.tank import TdtTankBase, PandasTank
from span.tdt import SpikeDataFrame


class TestNonZeroExistingFile(unittest.TestCase):
    def test_success(self):
        assert False

    def test_fail(self):
        assert False


class TestGetFirstMatch(unittest.TestCase):
    def test_success(self):
        assert False


class TestMatchInt(unittest.TestCase):
    def test_success(self):
        assert False

    def test_fail(self):
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

    def test__parse_date(self):
        d = self.tank._parse_date('')
        now = pd.datetime.now()
        self.assertEqual(d.month, now.month)
        self.assertEqual(d.day, now.day)
        self.assertEqual(d.year, now.year + 2000)

        # date-like string case
        s = self.tank.name
        d = self.tank._parse_date(s)
        self.assertIsInstance(d, datetime.date)

    def test_age(self):
        assert hasattr(self.tank, 'age')
        self.assertIsInstance(self.tank.age, (type(None), int))

    def test_site(self):
        assert hasattr(self.tank, 'site')
        self.assertIsinstance(self.tank.site, (type(None), int))

    def test__read_tev(self):
        tev = self.tank._read_tev('Spik')()
        self.assertIsNotNone(tev)
        self.assertIsInstance(tev, SpikeDataFrame)

        tev = self.tank._read_tev('LFPs')()
        self.assertIsNotNone(tev)
        self.assertIsInstance(tev, SpikeDataFrame)

    def test__read_tsq(self):
        tsq = self.tank._read_tsq()
        self.assertIsNotNone(tsq)
        self.assertIsInstance(tsq, pd.DataFrame)

    def test_tsq(self):
        self.assertIsNotNone(self.tank.tsq)

    def test_tev(self):
        self.assertIsNotNone(self.tank.tev('Spik'))
        self.assertIsNotNone(self.tank.tev('LFPs'))

    def test_spikes(self):
        self.assertIsNotNone(self.tank.spikes)

    def test_lfps(self):
        self.assertIsNotNone(self.tank.lfps)
