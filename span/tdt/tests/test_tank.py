import os
import types
import numbers
import unittest
import datetime
import warnings

import numpy as np
import pandas as pd

from six.moves import zip
import six

from span.tdt.tank import (TdtTank, PandasTank, _python_read_tev_raw,
                           _create_ns_datetime_index, _reshape_spikes,
                           _raw_reader)
from span.tdt import SpikeDataFrame
from span.testing import slow, create_stsq
from span.utils import OrderedDict
from span import ElectrodeMap, NeuroNexusMap


class TestReadTev(object):
    def setUp(self):
        span_data_path = os.environ['SPAN_DATA_PATH']
        assert os.path.isdir(span_data_path)
        self.filename = os.path.join(span_data_path,
                                     'Spont_Spikes_091210_p17rat_s4_657umV')

        self.tank = PandasTank(self.filename, ElectrodeMap(NeuroNexusMap,
                                                           50, 125))
        self.names = 'Spik', 'LFPs'

    def tearDown(self):
        del self.names, self.tank, self.filename

    def _reader_builder(self, reader):
        for name in self.names:
            tsq, _ = self.tank.tsq(name)

            with warnings.catch_warnings():
                warnings.simplefilter('ignore', FutureWarning)
                tsq.reset_index(drop=True, inplace=True)

            fp_locs = tsq.fp_loc.astype(int)

            assert np.issubdtype(fp_locs.dtype, np.integer)

            chunk_size = tsq.size.unique().max()

            spikes = np.empty((tsq.shape[0], chunk_size), np.float32)

            reader(self.filename + os.extsep + 'tev', fp_locs, chunk_size,
                   spikes)

            # mean should be at least on the order of millivolts if not less
            mag = np.log10(np.abs(spikes).mean())
            assert mag <= -3.0

    def test_read_tev(self):
        for reader in {_python_read_tev_raw, _raw_reader}:
            yield self._reader_builder, reader


def test_create_ns_datetime_index():
    start, fs, nsamples = datetime.datetime.now().date(), 103.342, 10
    index = _create_ns_datetime_index(start, fs, nsamples)
    assert isinstance(index, pd.DatetimeIndex)
    assert int(1e9 / fs) == index.freq.n
    assert index.size == nsamples


def test_reshape_spikes():
    meta = create_stsq(size=64, samples_per_channel=17)
    nblocks, block_size = meta.shape[0], meta.size[0]
    df = pd.DataFrame(np.empty((nblocks, block_size)))
    items = df.groupby(meta.channel.values).indices.items()
    items.sort()
    group_inds = np.column_stack(OrderedDict(items).itervalues())
    nchannels = group_inds.shape[1]
    nsamples = nblocks * block_size // nchannels
    reshaped = _reshape_spikes(df.values, group_inds)

    a, b = reshaped.shape, (nsamples, nchannels)
    assert a == b


class TestPandasTank(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tankname = os.path.join(os.environ['SPAN_DATA_PATH'],
                                'Spont_Spikes_091210_p17rat_s4_657umV')
        cls.tank = PandasTank(tankname)

    @classmethod
    def tearDownClass(cls):
        del cls.tank

    def test_properties(self):
        names = ('fs', 'name', 'age', 'site', 'date', 'time', 'datetime',
                 'duration')
        typs = ((numbers.Real, np.floating), six.string_types,
                (numbers.Integral, np.integer),
                (numbers.Integral, np.integer), datetime.date, datetime.time,
                pd.datetime, np.timedelta64)

        for name, typ in zip(names, typs):
            self.assert_(hasattr(self.tank, name))

            for event_name in self.names:
                attr = getattr(self.tank, name)

                try:
                    tester = attr[event_name]
                except (TypeError, IndexError):
                    tester = attr

                self.assertIsInstance(tester, (types.NoneType, typ))

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
            tsq, _ = self.tank._get_tsq_event(name)()
            self.assertIsNotNone(tsq)
            self.assertIsInstance(tsq, pd.DataFrame)

    def test_tsq(self):
        for name in self.names:
            self.assertIsNotNone(self.tank.tsq(name))

    @slow
    def test_tev(self):
        for name in self.names:
            self.assertIsNotNone(self.tank._tev(name))

    @slow
    def test_events(self):
        for name in self.names:
            self.assertIsInstance(getattr(self.tank, name.lower()),
                                  SpikeDataFrame)
