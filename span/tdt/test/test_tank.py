import os
import unittest
import random
import string
import datetime

import pandas as pd

from span.tdt.tank import TdtTankBase, PandasTank
from span.tdt import SpikeDataFrame


class TestTdtTankBase(unittest.TestCase):
    def test___init__(self):
        self.assertRaises(TypeError, TdtTankBase, '')


class TestPandasTank(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tankname = os.path.join(os.path.expanduser('~'), 'Data', 'xcorr_data',
                                'Spont_Spikes_091210_p17rat_s4_657umV')
        tn = os.path.join(tankname, os.path.basename(tankname))
        cls.tank = PandasTank(tn)

    def test__parse_date(self):
        d = self.tank._parse_date('')
        now = pd.datetime.now()
        assert d.month == now.month
        assert d.day == now.day
        assert d.year == now.year + 2000

        # date-like string case
        s = self.tank.tankname
        d = self.tank._parse_date(s)

    def test_date(self):
        assert hasattr(self.tank, 'date')
        assert isinstance(self.tank.date, (basestring, datetime.date))

    def test__read_tev(self):
        tev = self.tank._read_tev('Spik')()
        assert tev is not None
        assert isinstance(tev, SpikeDataFrame)

        tev = self.tank._read_tev('LFPs')()
        assert tev is not None
        assert isinstance(tev, SpikeDataFrame)

    def test__read_tsq(self):
        tsq = self.tank._read_tsq()
        assert tsq is not None
        assert isinstance(tsq, pd.DataFrame)

    def test_tsq(self):
        assert self.tank.tsq is not None

    def test_tev(self):
        assert self.tank.tev('Spik') is not None
        assert self.tank.tev('LFPs') is not None
        # assert self.tank.tev('Tick') is not None

    def test_spikes(self):
        assert self.tank.spikes is not None
