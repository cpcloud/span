import os
import re

import unittest

import pandas as pd

from span.tdt.tank import TdtTankBase, PandasTank
from span.tdt import SpikeDataFrame

from span.tdt.tank import _get_first_match, _match_int


class TestGetFirstMatch(unittest.TestCase):
    def test_get_first_match_succeed(self):
        pattern = r'(\d).*'
        string = r'1, 2, 3'
        fm = _get_first_match(pattern, string)
        self.assert_(fm)

    def test_get_first_match_fail(self):
        pattern = r'(\d).*'
        string = r'asdf'
        self.assertRaises(AttributeError, _get_first_match, pattern, string)


class TestMatchInt(unittest.TestCase):
    def test_match_int_succeed(self):
        pattern = r'(\d).*'
        string = r'1, 2, 3'
        r = _match_int(pattern, string)
        self.assertIsInstance(r, int)

    def test_match_int_succeed_and_exceptions(self):
        pattern = r'(\d).*'
        string = r'1, 2, 3'
        r = _match_int(pattern, string)
        self.assertIsInstance(r, int)

        r, e = _match_int(pattern, string, get_exc=True)
        self.assertIsInstance(r, int)
        self.assertIsNone(e)

    def test_match_int_fail_none(self):
        pattern = r'(\d).*'
        string = r'asdf'
        r = _match_int(pattern, string)
        self.assertIsNone(r)

    def test_match_int_fail_none_and_exceptions(self):
        pattern = r'(\d).*'
        string = r'asdf'
        r, e = _match_int(pattern, string, get_exc=True)
        self.assertIsNone(r)
        self.assertIsInstance(e, Exception)


class TestTdtTankBase(unittest.TestCase):
    def test_init(self):
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

    def setUp(self):
        self.names = 'Spik', 'LFPs'

    def tearDown(self):
        del self.names

    def test_repr(self):
        r = repr(self.tank)

    def test_read_tev(self):
        for name in self.names:
            tev = self.tank._read_tev(name)()
            self.assertIsNotNone(tev)
            self.assertIsInstance(tev, SpikeDataFrame)

    def test_read_tsq(self):
        for name in self.names:
            tsq, _ = self.tank._read_tsq(name)()
            self.assertIsNotNone(tsq)
            self.assertIsInstance(tsq, pd.DataFrame)

    def test_tsq(self):
        for name in self.names:
            self.assertIsNotNone(self.tank.tsq(name))

    def test_stsq(self):
        self.assertIsNotNone(self.tank.stsq)

    def test_ltsq(self):
        self.assertIsNotNone(self.tank.ltsq)

    def test_tev(self):
        for name in self.names:
            self.assertIsNotNone(self.tank.tev(name))

    def test_spikes(self):
        self.assertIsNotNone(self.tank.spikes)

    def test_lfps(self):
        self.assertIsNotNone(self.tank.lfps)
