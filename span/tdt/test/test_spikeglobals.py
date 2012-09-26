import itertools
import operator

import numpy as np
from numpy.testing import assert_array_equal
import pandas as pd

from span.tdt.spikeglobals import (
    DistanceMap, ElectrodeMap, ElectrodesPerShank, EventTypes, Indexer,
    MedialLateral, NShanks, NSides, ShankMap, TsqFields, TsqNumpyTypes)

def test_TsqFields():
    assert isinstance(TsqFields, (list, tuple)), 'TsqFields must be a list or tuple of numpy.dtype instances'

def test_TsqNumpyTypes():
    insts = map(isinstance, TsqNumpyTypes, itertools.repeat(type,
                                                            len(TsqNumpyTypes)))
    assert all(insts), 'invalid class of TsqNumpyTypes'
    itemsizes = map(operator.attrgetter('itemsize'), map(np.dtype, TsqNumpyTypes))
    expected = [4, 4, 4, 2, 2, 8, 8, 4, 4]
    assert len(itemsizes) == len(expected)
    eq_itemsizes = map(operator.eq, itemsizes, expected)
    assert all(eq_itemsizes), 'invalid itemsizes for dtypes'


def test_ElectrodeMap():
    assert isinstance(ElectrodeMap, pd.Series)
    assert ElectrodeMap.size == NShanks * ElectrodesPerShank
    assert ElectrodeMap.name == 'Electrode Map'


def test_NShanks():
    assert NShanks == 4


def test_ElectrodesPerShank():
    assert ElectrodesPerShank == 4


def test_NSides():
    assert NSides == NShanks * 2


def test_ShankMap():
    assert isinstance(ShankMap, pd.Series)
    assert ShankMap.values.size == NShanks ** 2
    assert ShankMap.name == 'Shank Map'
    assert ShankMap.dtype == np.int64


def test_MedialLateral():
    assert isinstance(MedialLateral, pd.Series)
    uniq = MedialLateral.unique()
    assert uniq.size == 2
    assert uniq.dtype in (np.dtype('O'), np.dtype(str))
    assert MedialLateral.values.size == NShanks * ElectrodesPerShank


def test_Indexer():
    for el, d in zip(('channel', 'shank', 'side'),
                     (ElectrodeMap, ShankMap, MedialLateral)):
        assert_array_equal(getattr(Indexer, el), d)


def test_EventTypes():
    assert isinstance(EventTypes, pd.Series)
    assert EventTypes.name == 'Event Types'
    assert np.isnan(EventTypes[0])


def test_DistanceMap():
    assert isinstance(DistanceMap, pd.DataFrame)
    assert_array_equal(DistanceMap.columns, Indexer.channel)
    assert_array_equal(DistanceMap.columns, DistanceMap.index)
