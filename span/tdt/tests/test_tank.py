import os
import types
import numbers
import unittest
import datetime
from glob import glob

import numpy as np
import pandas as pd

from span.tdt.tank import TdtTankBase, PandasTank, _read_tev, _read_tev_python
from span.tdt import SpikeDataFrame
from span.tdt.tank import _get_first_match, _match_int
from span.testing import slow, assert_array_almost_equal


@slow
class TestReadTev(unittest.TestCase):
    def setUp(self):
        home = os.path.expanduser('~')
        path = os.path.join(home, 'Data', 'xcorr_data',
                            'Spont_Spikes_091210_p17rat_s4_657umV')
        self.path = glob(os.path.join(path, '*%stev' % os.extsep))[0]
        self.tank = PandasTank(self.path[:-4])
        self.names = 'Spik', 'LFPs'

    def tearDown(self):
        del self.names, self.tank, self.path

    def test_read_tev_equal(self):
        for name in self.names:
            tsq, _ = self.tank.tsq(name)
            fp_locs = tsq.fp_loc

            self.assertEqual(np.dtype(np.int64), fp_locs.dtype)

            nsamples, chunk_size = fp_locs.size, tsq.size.unique().max()

            del tsq

            spikes = np.empty((nsamples, chunk_size), np.float32)
            spikes_cython = spikes.copy()

            _read_tev_python(self.path, chunk_size, fp_locs, spikes)
            _read_tev(self.path, chunk_size, fp_locs, spikes_cython)

            assert_array_almost_equal(spikes, spikes_cython)

            # mean should be at least on the order of millivolts if not less
            mag = np.log10(np.abs(spikes).mean())
            self.assertLessEqual(mag, -3.0)

            del spikes, mag, fp_locs

    def test_read_tev_python(self):
        for name in self.names:
            tsq, _ = self.tank.tsq(name)
            fp_locs = tsq.fp_loc

            self.assertEqual(np.dtype(np.int64), fp_locs.dtype)

            nsamples, chunk_size = fp_locs.size, tsq.size.unique().max()

            del tsq

            spikes = np.empty((nsamples, chunk_size), np.float32)

            _read_tev_python(self.path, chunk_size, fp_locs, spikes)

            # mean should be at least on the order of millivolts if not less
            mag = np.log10(np.abs(spikes).mean())
            self.assertLessEqual(mag, -3.0)

            del spikes, mag, fp_locs

    def test_read_tev_cython(self):
        for name in self.names:
            tsq, _ = self.tank.tsq(name)
            fp_locs = tsq.fp_loc

            self.assertEqual(np.dtype(np.int64), fp_locs.dtype)

            nsamples, chunk_size = fp_locs.size, tsq.size.unique().max()

            del tsq

            spikes = np.empty((nsamples, chunk_size), np.float32)

            _read_tev(self.path, chunk_size, fp_locs, spikes)

            # mean should be at least on the order of millivolts if not less
            mag = np.log10(np.abs(spikes).mean())
            self.assertLessEqual(mag, -3.0)

            del spikes, mag, fp_locs


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
        typs = (numbers.Real, basestring, numbers.Integral,
                numbers.Integral, datetime.date, datetime.time,
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
