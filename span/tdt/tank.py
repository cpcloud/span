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
import itertools
import collections
import numbers

import numpy as np
from numpy import nan as NA
from pandas import DataFrame, DatetimeIndex
import pandas as pd
from six.moves import xrange

from span.tdt.spikeglobals import Indexer, EventTypes, DataTypes
from span.tdt.spikedataframe import SpikeDataFrame
from span.tdt._read_tev import _read_tev_raw

from span.utils import (name2num, thunkify, cached_property, fromtimestamp,
                        assert_nonzero_existing_file, ispower2)


def _python_read_tev_serial(filename, grouped, block_size, spikes):
    dt = spikes.dtype
    nblocks, nchannels = grouped.shape
    iprod = itertools.product
    izip = itertools.izip

    with open(filename, 'rb') as f:
        for (c, b), gbc in izip(iprod(xrange(nchannels), xrange(nblocks)),
                                grouped.flat):
            f.seek(gbc)
            ks = slice(b * block_size, (b + 1) * block_size)
            spikes[ks, c] = np.fromfile(f, dt, block_size)


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
        tsq = DataFrame.from_records(np.fromfile(tsq_name, self.dtype))
        tsq.strobe[tsq.strobe <= np.finfo(np.float64).eps] = NA

        # zero based indexing
        tsq.channel -= 1.0

        # -1s are invalid
        tsq.channel[tsq.channel == -1.0] = NA

        tsq.type = EventTypes[tsq.type].values
        tsq.format = DataTypes[tsq.format].values

        tsq.timestamp[np.logical_not(tsq.timestamp)] = NA
        tsq.fs[np.logical_not(tsq.fs)] = NA

        # trim the fat
        tsq.size.ix[2:] -= self.dtype.itemsize / tsq.size.dtype.itemsize

        # create some new indices based on the electrode array
        srt = Indexer.sort('channel')
        srt.reset_index(drop=True, inplace=True)
        shank = srt.shank[tsq.channel].values

        tsq['shank'] = shank

        # convert the event_name to a number
        name = name2num(event_name)

        # get the row of the metadata where its value equals the name-number
        row = tsq.name == name

        # make sure there's at least one event
        p = self.path
        assert row.any(), 'no event named %s in tank: %s' % (event_name, p)
        self.raw = tsq

        # get all the metadata for those events
        tsq = tsq[row]

        # convert to integer where possible
        tsq.channel = tsq.channel.astype(int)
        tsq.shank = tsq.shank.astype(int)

        return tsq, row.argmax()

    def tsq(self, event_name):
        return self._read_tsq(event_name)()

    @cached_property
    def stsq(self):
        tsq, _ = self.tsq('Spik')
        tsq.reset_index(drop=True, inplace=True)
        return tsq

    @cached_property
    def ltsq(self):
        tsq, _ = self.tsq('LFPs')
        tsq.reset_index(drop=True, inplace=True)
        return tsq

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

        try:
            self.age = int(self._age_re.search(self.name).group(1))
        except:
            self.age = None

        try:
            self.site = int(self._site_re.search(self.name).group(1))
        except:
            self.site = None

        istart = self.stsq.timestamp.index[0]
        iend = self.stsq.timestamp.index[-1]
        tstart = pd.datetime.fromtimestamp(self.stsq.timestamp[istart])

        self.__datetime = pd.Timestamp(tstart)
        self.time = self.__datetime.time()
        self.date = self.__datetime.date()
        self.fs = self.stsq.fs[istart]
        self.start = self.__datetime

        tend = pd.datetime.fromtimestamp(self.stsq.timestamp[iend])

        self.end = pd.Timestamp(tend)
        self.duration = np.timedelta64(self.end - self.start)

    def __repr__(self):
        objr = repr(self.__class__)
        params = dict(age=self.age, name=self.name, site=self.site, obj=objr,
                      fs=self.fs, datetime=str(self.datetime),
                      duration=self.duration / np.timedelta64(1, 'm'))
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
        spikes = np.empty((nblocks, block_size), dtype=dtype)

        # tev filename
        tev_name = self.path + os.extsep + self._raw_ext

        meta.reset_index(drop=True, inplace=True)

        # inds = meta.groupby('channel').indices
        # grouped = DataFrame(inds)
        # grouped.sort_index(axis=1, inplace=True)
        # grouped_locs = meta.fp_loc.values.take(grouped.values)

        # convert timestamps to datetime objects
        meta.timestamp = fromtimestamp(meta.timestamp)

        index = _create_ns_datetime_index(self.datetime, self.fs, nsamples)
        return _read_tev_impl(tev_name, meta.fp_loc.values, block_size,
                              meta.channel.values, meta.shank.values, spikes,
                              index, columns.sortlevel(1)[0])


def _create_ns_datetime_index(start, fs, nsamples):
    """Create a datetime index in nanoseconds

    Parameters
    ----------
    start : datetime_like
    fs : Series
    nsamples : int

    Returns
    -------
    index : DatetimeIndex
    """
    ns = int(1e9 / fs)
    dtstart = np.datetime64(start)
    dt = dtstart + np.arange(nsamples) * np.timedelta64(ns, 'ns')
    return DatetimeIndex(dt, freq=ns * pd.datetools.Nano(), name='time',
                         tz='US/Eastern')


def _reshape_spikes(df, group_inds):
    reshaped = df.take(group_inds, axis=0)
    shp = reshaped.shape
    shpsrt = np.argsort(reshaped.shape)[::-1]
    nchannels = shp[shpsrt[-1]]
    newshp = reshaped.size // nchannels, nchannels
    return reshaped.transpose(shpsrt).reshape(newshp)


def _read_tev(filename, fp_locs, block_size, channel, shank, spikes,
              index, columns):
    assert isinstance(filename, basestring)
    assert isinstance(fp_locs, (np.ndarray, collections.Sequence))
    assert isinstance(block_size, (numbers.Integral, np.integer))
    assert ispower2(block_size)
    assert isinstance(channel, (np.ndarray, collections.Sequence))
    assert isinstance(shank, (np.ndarray, collections.Sequence))
    assert len(channel) == len(shank)
    assert isinstance(spikes, np.ndarray)
    assert spikes.shape[1] == block_size
    assert isinstance(index, pd.Index)
    assert isinstance(columns, pd.Index)
    return _read_tev_impl(filename, fp_locs, block_size, channel, shank,
                          spikes, index, columns)


def _read_tev_impl(filename, fp_locs, block_size, channel, shank, spikes,
                   index, columns):
    _read_tev_raw(filename, fp_locs, block_size, spikes)

    new_cols = {'channel': channel, 'shank': shank}
    df = DataFrame(spikes)

    for k, v in new_cols.iteritems():
        df[k] = v

    group_inds = DataFrame(df.groupby('channel').indices)

    for name in new_cols.iterkeys():
        del df[name]

    new_a = _reshape_spikes(df.values, group_inds.values)
    df = DataFrame(new_a, index, columns.reorder_levels((1, 0)))
    df.sort_index(axis=1, inplace=True)
    return SpikeDataFrame(df, dtype=float)


if __name__ == '__main__':
    f = ('/home/phillip/Data/xcorr_data/Spont_Spikes_091210_p17rat_s4_'
         '657umV/Spont_Spikes_091210_p17rat_s4_657umV')
    tank = PandasTank(f)
    sp = tank.spikes
