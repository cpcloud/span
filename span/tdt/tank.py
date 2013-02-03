#!/usr/bin/env python

# tank.py ---

# Copyright (C) 2012 Copyright (C) 2012 Phillip Cloud <cpcloud@gmail.com>

# Author: Phillip Cloud <cpcloud@gmail.com>

# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.


"""
Examples
--------
>>> import span
>>> path = 'some/path/to/a/tank/file'
>>> tank = span.tdt.PandasTank(path)
"""

from future_builtins import zip

import os
import abc
import re
import numbers
import itertools

import numpy as np
from numpy import nan as NA
from numpy import (float32 as f4, int32 as i4, uint32 as u4, uint16 as u2,
                   float64 as f8, int64 as i8)
from pandas import Series, DataFrame, DatetimeIndex
import pandas as pd

from numba import autojit, NumbaError

from span.tdt.spikeglobals import Indexer, EventTypes, DataTypes
from span.tdt.spikedataframe import SpikeDataFrame
from span.tdt._read_tev import (_read_tev_parallel as __read_tev_parallel,
                                _read_tev_serial as __read_tev_serial)


from span.utils import (name2num, thunkify, cached_property, fromtimestamp,
                        assert_nonzero_existing_file)


_TsqFields = ('size', 'type', 'name', 'channel', 'sort_code', 'timestamp',
              'fp_loc', 'format', 'fs')

_TsqNumpyTypes = i4, i4, u4, u2, u2, f8, i8, i4, f4

_TsqTypeDict = dict(zip(_TsqFields, _TsqNumpyTypes))


def _get_first_match(pattern, string):
    """Helper function for getting the first match of a pattern within a
    string.

    Parameters
    ----------
    pattern, string : str

    Returns
    -------
    rgrp1 : str
    """
    try:
        r = pattern.match(string)
    except AttributeError:
        r = re.match(pattern, string)

    return r.group(1)


@autojit
def _read_tev_numba(filename, grouped, block_size, spikes):
    nblocks, nchannels = grouped.shape

    dt = spikes.dtype

    f = open(filename, 'rb')

    for c in xrange(nchannels):
        for b in xrange(nblocks):
            f.seek(grouped[b, c])
            low = b * block_size
            high = (b + 1) * block_size
            spikes[low:high, c] = np.fromfile(f, dt, block_size)

    f.close()


def _read_tev_parallel(filename, grouped, block_size, spikes):
    """Read a TDT tev file into a numpy array. Slightly faster than
    the pure Python version.

    Parameters
    ----------
    filename : char *
        Name of the TDT file to load.

    block_size : int
        The number of samples per chunk of data.

    fp_locs : integral[:]
        The array of locations of each chunk in the TEV file.

    spikes : floating[:, :]
        Output array
    """
    assert filename, 'filename (1st argument) cannot be empty'
    assert block_size > 0, '"block_size" must be greater than 0'
    assert isinstance(filename, basestring), 'filename must be a string'
    assert isinstance(block_size, numbers.Integral)
    assert isinstance(spikes, np.ndarray)

    __read_tev_parallel(filename, grouped, block_size, spikes)


def _read_tev_serial(filename, grouped, block_size, spikes):
    """Read a TDT tev file into a numpy array. Slightly faster than
    the pure Python version.

    Parameters
    ----------
    filename : char *
        Name of the TDT file to load.

    nsamples : i8
        The number of samples per chunk of data.

    fp_locs : i8[:]
        The array of locations of each chunk in the TEV file.

    spikes : floating[:, :]
        Output array
    """
    assert filename, 'filename (1st argument) cannot be empty'
    assert isinstance(filename, basestring), 'filename must be a string'
    assert isinstance(block_size, numbers.Integral)
    assert isinstance(spikes, np.ndarray)

    __read_tev_serial(filename, grouped, block_size, spikes)


def _read_tev_python(filename, grouped, block_size, spikes):
    dt = spikes.dtype
    nblocks, nchannels = grouped.shape

    with open(filename, 'rb') as f:
        for c, b in itertools.product(xrange(nchannels), xrange(nblocks)):
            f.seek(grouped[b, c])
            v = np.fromfile(f, dt, block_size)
            spikes[b * block_size:(b + 1) * block_size, c] = v


def _read_tev(*args, **kwargs):
    try:
        _read_tev_numba(*args, **kwargs)
    except (NameError, NumbaError):
        _read_tev_parallel(*args, **kwargs)


def _match_int(pattern, string, get_exc=False, excs=(AttributeError,
                                                     ValueError, TypeError)):
    """Convert a string matched from a regex to an integer or return None.

    Parameters
    ----------
    pattern : str or compiled regex
        Regular expression to use to match elements of `string`

    string : str
        The string to match

    get_exc : bool, optional, default False
        Whether to return the exceptions caught

    excs : tuple of Exceptions, optional, default (AttributeError, ValueError,
                                                     TypeError)
        The exceptions to catch when trying to match an integer in the regular
        expression `pattern`

    Returns
    -------
    r : {int, None, tuple of int, tuple of None and Exception}
    """
    try:
        r = int(_get_first_match(pattern, string))
        e = None
    except excs as e:
        r = None

    if get_exc:
        r = r, e

    return r


class TdtTankAbstractBase(object):
    """Interface to tank reading methods.

    Attributes
    ----------
    tsq (pandas.DataFrame) : Recording metadata
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self):
        super(TdtTankAbstractBase, self).__init__()

    @abc.abstractmethod
    def _read_tev(self, event_name):
        pass  # pragma: no cover

    @thunkify
    def _read_tsq(self, event_name):
        """Read the metadata (TSQ) file of a TDT Tank.

        Returns
        -------
        b : pandas.DataFrame
            Recording metadata
        """
        # create the path name
        tsq_name = self.path + os.extsep + self._header_ext

        # read in the raw data as a numpy rec array and convert to
        # DataFrame
        tsq = DataFrame.from_records(np.fromfile(tsq_name, dtype=self.tsq_dtype))

        # zero based indexing
        tsq.channel -= 1.0

        # -1s are invalid
        tsq.channel[tsq.channel == -1.0] = NA

        tsq.type = EventTypes[tsq.type].reset_index(drop=True)
        tsq.format = DataTypes[tsq.format].reset_index(drop=True)

        tsq.timestamp[tsq.logical_not(tsq.timestamp)] = NA
        tsq.fs[np.logical_not(tsq.fs)] = NA

        # fragile subtraction (i.e., what if TDT changes this value?)
        tsq.size -= 10

        # create some new indices based on the electrode array
        srt = Indexer.sort('channel').reset_index(drop=True)
        shank = srt.shank[tsq.channel].reset_index(drop=True)

        tsq['shank'] = shank

        # convert the event_name to a number
        name = name2num(event_name)

        # get the row of the metadata where its value equals the name-number
        row = tsq.name == name

        # make sure there's at least one event
        assert row.any(), 'no event named %s in tank: %s' % (event_name,
                                                             self.path)

        # get all the metadata for those events
        tsq = tsq[row]

        # convert to integer where possible
        tsq.channel = tsq.channel.astype(int)
        tsq.shank = tsq.shank.astype(int)

        return tsq, row

    def tsq(self, event_name):
        getter = self._read_tsq(event_name)
        d, row = getter()
        return d, row

    @cached_property
    def stsq(self):
        tsq, _ = self.tsq('Spik')
        return tsq.reset_index(drop=True)

    @cached_property
    def ltsq(self):
        tsq, _ = self.tsq('LFPs')
        return tsq.reset_index(drop=True)

    def tev(self, event_name):
        """Return the data from a particular event.

        Parameters
        ----------
        event_name : str
            The name of the event whose data you'd like to retrieve.

        Returns
        -------
        tev : SpikeDataFrame
            The raw data from the TEV file
        """
        return self._read_tev(event_name)()


class TdtTankBase(TdtTankAbstractBase):
    """Base class encapsulating methods for reading a TDT Tank.

    Parameters
    ----------
    path : str
        The path to the tank file sans extension.

    Attributes
    ----------
    fields
    np_types
    tsq_dtype

    path (``str``) : Full path of the tank sans extensions
    name (``str``) : basename of self.path
    age (``int``) : The postnatal day age of the animal
    site (``int``) : The site number of the recording, can be ``None``
    datetime (``datetime.datetime``) : Date and time of the recording
    time (``datetime.time``) : Time of the recording
    date (``datetime.date``) : Date of the recording
    fs (``float``) : sampling rate
    start (``Timestamp``) : Start time of the recording
    end (``Timestamp``) : End time of the recording
    duration (``timedelta64[us]``) : Duration of the recording
    """

    fields = _TsqFields
    np_types = _TsqNumpyTypes
    tsq_dtype = np.dtype(list(zip(fields, np_types)))
    types = Series(list(map(np.dtype, np_types)), index=fields)

    _site_re = re.compile(r'(?:.*s(?:ite)?(?:|_)?(\d+))?')
    _age_re = re.compile(r'.*[pP](\d+).*')

    _header_ext = 'tsq'
    _raw_ext = 'tev'

    def __init__(self, path):
        super(TdtTankBase, self).__init__()

        tank_with_ext = path + os.extsep
        tev_path = tank_with_ext + self._raw_ext
        tsq_path = tank_with_ext + self._header_ext

        assert_nonzero_existing_file(tev_path)
        assert_nonzero_existing_file(tsq_path)

        self.path = path
        self.name = os.path.basename(path)
        self.age = _match_int(self._age_re, self.name)
        self.site = _match_int(self._site_re, self.name)
        istart = self.stsq.timestamp.index[0]
        iend = self.stsq.timestamp.index[-1]
        tstart = pd.datetime.fromtimestamp(self.stsq.timestamp[istart])
        self.__datetime = pd.Timestamp(tstart)
        self.time = self.__datetime.time()
        self.date = self.__datetime.date()
        self.fs = self.stsq.reset_index(drop=True).fs[0]
        self.start = self.__datetime
        tend = pd.datetime.fromtimestamp(self.stsq.timestamp[iend])
        self.end = pd.Timestamp(tend)
        self.duration = np.timedelta64(self.end - self.start)

    def __repr__(self):
        objr = repr(self.__class__)
        params = dict(age=self.age, name=self.name, site=self.site, obj=objr,
                      fs=self.fs, datetime=str(self.datetime),
                      duration=self.duration / np.timedelta64(1, 'm'))
        # max_str_len = max(map(len, params.values()))
        # names = params.keys()
        # values = map(lambda x: x.rjust(max_str_len), params.values())

        fmt = ('{obj}\nname:     {name}\ndatetime: {datetime}\nage:      '
               'P{age}\nsite:     {site}\nfs:       {fs}\n'
               'duration: {duration:.2f} min')
        return fmt.format(**params)

    @property
    def datetime(self):
        return self.__datetime.to_pydatetime()

    @cached_property
    def spikes(self):
        return self.tev('Spik')

    @cached_property
    def lfps(self):
        return self.tev('LFPs')


class PandasTank(TdtTankBase):
    """Implements the abstract methods from TdtTankBase.

    Parameters
    ----------
    path : str
        Name of the tank files sans extension.

    See Also
    --------
    TdtTankBase
        Base class implementing TSQ reading.
    """
    def __init__(self, path):
        super(PandasTank, self).__init__(path)

    @thunkify
    def _read_tev(self, event_name, group='channel'):
        """Read an event from a TDT Tank tev file.

        Parameters
        ----------
        event_name : str

        Returns
        -------
        d : SpikeDataFrame

        Raises
        ------
        AssertionError
            If there is no event with the name `event_name`.

        See Also
        --------
        span.tdt.SpikeDataFrame
        """
        from span.tdt.spikeglobals import ChannelIndex as columns

        meta, row = self.tsq(event_name)

        # first row of event type
        first_row = np.argmax(row)

        # data type of this event
        dtype = meta.format[first_row]

        # number of samples per chunk
        block_size = np.int64(meta.size[first_row])
        nchannels = meta.channel.dropna().nunique()
        nsamples = meta.shape[0] * block_size // nchannels

        # raw ndarray for data
        spikes = np.empty((nsamples, nchannels), dtype=dtype)

        # tev filename
        tev_name = self.path + os.extsep + self._raw_ext

        meta.reset_index(drop=True, inplace=True)

        grouped = DataFrame(meta.groupby('channel').groups)
        grouped_locs = meta.fp_loc.values.take(grouped.values)

        # read in the TEV data to spikes
        _read_tev(tev_name, grouped_locs, block_size, spikes)

        # convert timestamps to datetime objects
        meta.timestamp = fromtimestamp(meta.timestamp)

        meta = meta.reset_index(drop=True)

        # datetime hack
        ns = int(1e9 / meta.fs.get_value(0))
        dtstart = np.datetime64(self.datetime)
        dt = dtstart + np.arange(nsamples) * np.timedelta64(ns, 'ns')
        index = DatetimeIndex(dt, freq=ns * pd.datetools.Nano(), name='time',
                              tz='US/Eastern')
        cols, _ = columns.swaplevel(1, 0).sortlevel('shank')
        return SpikeDataFrame(spikes, index=index, columns=cols, dtype=float)
