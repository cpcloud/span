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

import os
import abc
import re
import collections
import numbers
import warnings

import numpy as np
from numpy import nan as NA
from pandas import DataFrame, DatetimeIndex, Series
import pandas as pd
from pytz import UnknownTimeZoneError

from span.tdt.spikeglobals import Indexer, EventTypes, RawDataTypes
from span.tdt.spikedataframe import SpikeDataFrame
from span.tdt._read_tev import _read_tev_raw

from span.utils import (thunkify, cached_property, fromtimestamp,
                        assert_nonzero_existing_file, ispower2, OrderedDict,
                        num2name)


def _python_read_tev_raw(filename, fp_locs, block_size, spikes):
    with open(filename, 'rb') as f:
        for i, loc in enumerate(fp_locs):
            f.seek(loc)
            spikes[i] = np.fromfile(f, spikes.dtype, block_size)


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
    def _raw_tsq(self):
        # create the path name
        tsq_name = self.path + os.extsep + self._header_ext

        # read in the raw data as a numpy rec array and convert to
        # DataFrame
        tsq = DataFrame(np.fromfile(tsq_name, self.dtype))
        inds = tsq.strobe <= np.finfo(np.float64).eps
        tsq.strobe[inds] = NA

        # zero based indexing
        tsq.channel -= 1.0

        # -1s are invalid
        tsq.channel[tsq.channel == -1.0] = NA

        tsq.type = EventTypes[tsq.type].values
        tsq.format = RawDataTypes[tsq.format].values

        tsq.timestamp[np.logical_not(tsq.timestamp)] = NA
        tsq.fs[np.logical_not(tsq.fs)] = NA

        # trim the fat
        tsq.size.ix[2:] -= self.dtype.itemsize / tsq.size.dtype.itemsize

        # create some new indices based on the electrode array
        srt = Indexer.sort('channel')
        srt.reset_index(drop=True, inplace=True)
        shank = srt.shank[tsq.channel].values

        tsq['shank'] = shank

        not_null_strobe = tsq.strobe.notnull()

        for key in ('channel', 'shank', 'sort_code', 'fp_loc'):
            try:
                tsq[key][not_null_strobe] = NA
            except ValueError:
                tsq[key] = tsq[key].astype(float)
                tsq[key][not_null_strobe] = NA

        return tsq

    @thunkify
    def _get_tsq_event(self, event_name):
        """Read the metadata (TSQ) file of a TDT Tank.

        Returns
        -------
        b : pandas.DataFrame
            Recording metadata
        """
        tsq = self.raw

        # make sure there's at least one event
        p = self.path

        # get the row of the metadata where its value equals the name-number
        row = tsq.name.isin([event_name])
        assert row.any(), 'no event named %s in tank: %s' % (event_name, p)

        # get all the metadata for those events
        tsq = tsq[row]

        # convert to integer where possible
        try:
            tsq.channel = tsq.channel.astype(int)
            tsq.shank = tsq.shank.astype(int)
        except ValueError:
            pass

        return tsq, row.argmax()

    def tsq(self, event_name):
        return self._get_tsq_event(event_name)()

    @cached_property
    def raw(self):
        return self._raw_tsq()()

    def _tev(self, event_name):
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
    dtype

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
    _names = ('size', 'type', 'name', 'channel', 'sort_code', 'timestamp',
              'fp_loc', 'strobe', 'format', 'fs')
    _formats = 'i4', 'i4', 'u4', 'u2', 'u2', 'f8', 'i8', 'f8', 'i4', 'f4'
    _offsets = 0, 4, 8, 12, 14, 16, 24, 24, 32, 36

    dtype = np.dtype({'names': _names, 'formats': _formats,
                      'offsets': _offsets}, align=True)

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

        def _first_int_group(regex, name):
            try:
                return int(regex.search(name).group(1))
            except (TypeError, ValueError):  # pragma: no cover
                return None

        self.age = _first_int_group(self._age_re, self.name)
        self.site = _first_int_group(self._site_re, self.name)

        not_na_ts = self.raw.timestamp.dropna()
        tstart = pd.datetime.fromtimestamp(not_na_ts.head(1).item())
        tend = pd.datetime.fromtimestamp(not_na_ts.tail(1).item())

        self.__datetime = pd.Timestamp(tstart)
        self.time = self.__datetime.time()
        self.date = self.__datetime.date()

        self.start = self.__datetime
        self.end = pd.Timestamp(tend)

        self.duration = np.timedelta64(self.end - self.start)
        unames = self.raw.name.unique()
        raw_names = map(lambda x: NA if not x else x, map(num2name, unames))
        self.names = Series(list(raw_names), index=unames)

        self.raw.name = self.names[self.raw.name].reset_index(drop=True)
        self._name_mapper = dict(zip(self.names.str.lower().values,
                                     self.names.values))

        def _try_get_na(x):
            try:
                return x.item()
            except ValueError:
                return NA

        fs_nona = self.raw.fs.dropna()
        name_nona = self.raw.name.dropna()
        diter = ((name,  _try_get_na(fs_nona[name_nona == name].head(1)))
                 for name in self.names.dropna().values)
        self.fs = Series(dict(diter))

    def __repr__(self):
        objr = repr(self.__class__)
        params = dict(age=self.age, name=self.name, site=self.site, obj=objr,
                      fs=self.fs.to_dict(), datetime=str(self.datetime),
                      duration=self.duration / np.timedelta64(1, 'm'))
        fmt = ('{obj}\nname:     {name}\ndatetime: {datetime}\nage:      '
               'P{age}\nsite:     {site}\nfs:       {fs}\n'
               'duration: {duration:.2f} min')
        return fmt.format(**params)

    @property
    def values(self):
        return self.raw.values

    @property
    def datetime(self):
        return self.__datetime.to_pydatetime()

    def __getattr__(self, name):
        # try:
        mapper = super(TdtTankBase, self).__getattribute__('_name_mapper')

        # check to see if something similar was given
        lowered_name = name.lower()

        if lowered_name != name and lowered_name in mapper:
            raise AttributeError('Tried to retrieve the attribute '
                                 '\'%s\', did you mean \'%s\'?'
                                 % (name, lowered_name))

        return self._tev(mapper[name])
        # except (AssertionError, KeyError):
        # except AssertionError:
            # return super(TdtTankBase, self).__getattribute__(name)


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
        ValueError
            If there are duplicate file pointer locations

        See Also
        --------
        span.tdt.SpikeDataFrame
        """
        from span.tdt.spikeglobals import ChannelIndex as columns

        meta, first_row = self.tsq(event_name)

        # data type of this event
        dtype = meta.format[first_row]

        # number of samples per chunk
        block_size = meta.size[first_row]
        nchannels = meta.channel.dropna().nunique()
        nblocks = meta.shape[0]
        nsamples = nblocks * block_size // nchannels

        # raw ndarray for data
        spikes = DataFrame(np.empty((nblocks, block_size), dtype=dtype))

        tev_name = self.path + os.extsep + self._raw_ext
        meta.reset_index(drop=True, inplace=True)

        # convert timestamps to datetime objects (vectorized)
        meta.timestamp = fromtimestamp(meta.timestamp)

        index = _create_ns_datetime_index(self.datetime, self.fs[event_name],
                                          nsamples)
        return _read_tev(tev_name, meta.fp_loc.astype(int).values, block_size,
                         meta.channel.values, meta.shank.values,
                         spikes, index, columns.reorder_levels((1, 0)))


def _create_ns_datetime_index(start, fs, nsamples):
    """Create a DatetimeIndex in nanoseconds

    Parameters
    ----------
    start : datetime
    fs : Series
    nsamples : int

    returns
    -------
    index : DatetimeIndex
    """
    ns = int(1e9 / fs)
    dtstart = np.datetime64(start)
    dt = dtstart + np.arange(nsamples) * np.timedelta64(ns, 'ns')

    try:
        return DatetimeIndex(dt, freq=ns * pd.datetools.Nano(), name='time',
                             tz='US/Eastern')
    except UnknownTimeZoneError:  # pragma: no cover
        warnings.warn('time zone not found, you might need to reinstall '
                      'pytz or matplotlib or both', RuntimeWarning)
        return DatetimeIndex(dt, freq=ns * pd.datetools.Nano(), name='time')


def _reshape_spikes(df, group_inds):
    reshaped = df.take(group_inds, axis=-1)
    shp = reshaped.shape
    shpsrt = np.argsort(reshaped.shape)[::-1]
    nchannels = shp[shpsrt[-1]]
    newshp = reshaped.size // nchannels, nchannels
    return reshaped.transpose(shpsrt).reshape(newshp)


def _read_tev(filename, fp_locs, block_size, channel, shank, spikes,
              index, columns):
    assert isinstance(filename, basestring), 'filename must be a string'
    assert isinstance(fp_locs, (np.ndarray, collections.Sequence)), \
        'fp_locs must be a sequence'
    assert isinstance(block_size, (numbers.Integral, np.integer)), \
        'block_size must be an integer'
    assert ispower2(block_size), 'block_size must be a power of 2'
    assert isinstance(channel, (np.ndarray, collections.Sequence)), \
        'channel must be a sequence'
    assert isinstance(shank, (np.ndarray, collections.Sequence)), \
        'shank must be a sequence'
    assert len(channel) == len(shank), 'len(channel) != len(shank)'
    assert isinstance(spikes, DataFrame), 'spikes must be a DataFrame'
    assert spikes.shape[1] == block_size, \
        'number of columns of spikes must equal block_size'
    assert isinstance(index, pd.Index), 'index must be an instance of Index'
    assert isinstance(columns, pd.Index), \
        'columns must be an instance of Index'
    return _read_tev_impl(filename, fp_locs, block_size, channel, shank,
                          spikes, index, columns)


_raw_reader = _read_tev_raw


def _read_tev_impl(filename, fp_locs, block_size, channel, shank, spikes,
                   index, columns):
    _raw_reader(filename, fp_locs, block_size, spikes.values)

    items = spikes.groupby(channel).indices.items()
    items.sort()

    group_inds = np.column_stack(OrderedDict(items).itervalues())
    reshaped = _reshape_spikes(spikes.values, group_inds)
    d = DataFrame(reshaped, index, columns)
    d.sort_index(axis=1, inplace=True)
    return SpikeDataFrame(d, dtype=float)


if __name__ == '__main__':
    span_data_path = os.environ['SPAN_DATA_PATH']
    f = os.path.join(span_data_path, 'Spont_Spikes_091210_p17rat_s4_657umV')
    tank = PandasTank(f)
    sp = tank.Spik
