import os
import types
import numbers
import unittest
import datetime
import warnings
import glob

import numpy as np
import pandas as pd

from six.moves import zip

from span.tdt.tank import (TdtTankBase, PandasTank, _read_tev,
                           _read_tev_parallel, _python_read_tev_serial)
from span.tdt import SpikeDataFrame
from span.tdt.tank import _get_first_match, _match_int
from span.testing import slow


class TestReadTev(object):
    def setUp(self):
        path = os.getenv('SPAN_DATA_PATH')

        assert os.path.isdir(path)

        gettev = glob.glob(os.path.join(path, '*%stev' % os.extsep))

        if len(gettev) == 1:
            self.path, = gettev
        else:
            self.path = gettev[0]

        self.filename, _ = os.path.splitext(self.path)
        self.tank = PandasTank(self.filename)
        self.names = 'Spik', 'LFPs'

    def tearDown(self):
        del self.names, self.tank, self.filename, self.path

    def _reader_builder(self, reader):
        for name in self.names:
            tsq, _ = self.tank.tsq(name)

            with warnings.catch_warnings():
                warnings.simplefilter('ignore', FutureWarning)
                tsq.reset_index(drop=True, inplace=True)
            fp_locs = tsq.fp_loc

            assert np.dtype(np.int64) == fp_locs.dtype

            chunk_size = tsq.size.unique().max()
            grouped = pd.DataFrame(tsq.groupby('channel').groups)
            grouped_locs = fp_locs.values.take(grouped.values)

            spikes = np.empty((grouped_locs.shape[0] * chunk_size,
                               grouped_locs.shape[1]), np.float32)

            reader(self.path, grouped_locs, chunk_size, spikes)

            # mean should be at least on the order of millivolts if not less
            mag = np.log10(np.abs(spikes).mean())
            assert mag <= -3.0

    def test_read_tev(self):
        for reader in {_read_tev, _read_tev_parallel, _python_read_tev_serial}:
            yield self._reader_builder, reader


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

    def test_properties(self):
        names = ('fs', 'name', 'age', 'site', 'date', 'time', 'datetime',
                 'duration')
        typs = ((numbers.Real, np.floating), basestring,
                (numbers.Integral, np.integer),
                (numbers.Integral, np.integer), datetime.date, datetime.time,
                pd.datetime, np.timedelta64)

        for name, typ in zip(names, typs):
            self.assert_(hasattr(self.tank, name))
            self.assertIsInstance(getattr(self.tank, name),
                                  (types.NoneType, typ))

    def setUp(self):
        self.names = 'Spik', 'LFPs'

    def tearDown(self):
        del self.names

    def test_repr(self):
        r = repr(self.tank)
        self.assert_(r)

    @slow
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

    @slow
    def test_tev(self):
        for name in self.names:
            self.assertIsNotNone(self.tank.tev(name))

    @slow
    def test_spikes(self):
        self.assertIsNotNone(self.tank.spikes)

    @slow
    def test_lfps(self):
        self.assertIsNotNone(self.tank.lfps)
