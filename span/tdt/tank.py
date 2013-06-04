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
import numbers
import os
import re
import warnings

import numpy as np
import pandas as pd
from numpy import nan as NA
from pandas import DataFrame, DatetimeIndex, Series
from pandas.util.decorators import cache_readonly

from span.tdt._read_tev import _read_tev_raw
from span.tdt.spikedataframe import SpikeDataFrame
from span.tdt.spikeglobals import TdtEventTypes, TdtDataTypes
from span.utils import (thunkify, cached_property, fromtimestamp,
                        assert_nonzero_existing_file, ispower2, OrderedDict,
                        num2name, LOCAL_TZ, remove_first_pc)


def _python_read_tev_raw(filename, fp_locs, block_size, spikes):
    dt = spikes.dtype
    with open(filename, 'rb') as f:
        for i, loc in enumerate(fp_locs):
            f.seek(loc)
            spikes[i] = np.fromfile(f, dt, block_size)


def _first_int_group(regex, name):
    try:
        return int(regex.search(name).group(1))
    except (TypeError, ValueError):  # pragma: no cover
        return None


class TdtTank(object):
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

    _site_re = re.compile(r'(?:.*s(?:ite)?_?(\d{1,2}))?')
    _age_re = re.compile(r'(?<=_)[pP](\d+)')

    _header_ext = 'tsq'
    _raw_ext = 'tev'

    def __init__(self, path, electrode_map, clean=False):
        super(TdtTank, self).__init__()
        self.electrode_map = electrode_map

        tank_with_ext = path + os.extsep
        tev_path = tank_with_ext + self._raw_ext
        tsq_path = tank_with_ext + self._header_ext

        assert_nonzero_existing_file(tev_path)
        assert_nonzero_existing_file(tsq_path)

        self.path = path
        self.name = os.path.basename(path)

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
        raw_names = map(lambda x: x or NA, map(num2name, unames))
        self.names = Series(raw_names, index=unames)
        names = self.names

        self.raw.name = names[self.raw.name].reset_index(drop=True)
        self._name_mapper = dict(zip(names.str.lower(), names))

        def _try_get_na(x):
            try:
                return x.item()
            except (ValueError, IndexError):
                return NA

        fs_nona = self.raw.fs.dropna()
        name_nona = self.raw.name.dropna()
        diter = ((name, _try_get_na(fs_nona[name_nona == num].head(1)))
                 for num, name in self.names.dropna().iteritems())
        self.fs = Series(dict(diter))
        self.data_names = self.fs.dropna().index.values.astype(np.str_)
        self.clean = clean

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
        mapper = self._name_mapper

        # check to see if something similar was given
        lowered_name = name.lower()

        if lowered_name != name and lowered_name in mapper:
            raise AttributeError('Tried to retrieve the attribute '
                                 '\'%s\', did you mean \'%s\'?'
                                 % (name, lowered_name))

        return self._tev(mapper[name], self.clean)

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
        ind = Series(self.electrode_map.shank, self.electrode_map.channel)
        tsq['shank'] = ind[tsq.channel].reset_index(tsq.index, drop=True)

        tsq.type = TdtEventTypes[tsq.type].values
        tsq.format = TdtDataTypes[tsq.format].values

        tsq.timestamp[np.logical_not(tsq.timestamp)] = NA
        tsq.fs[np.logical_not(tsq.fs)] = NA

        # trim the fat
        dt = self.dtype
        stream = tsq.type == 'stream'
        tsq.size.ix[stream] -= dt.itemsize / dt['size'].itemsize

        not_null_strobe = tsq.strobe.notnull()

        for key in ('channel', 'sort_code', 'fp_loc'):
            try:
                tsq[key][not_null_strobe] = NA
            except ValueError:
                tsq[key] = tsq[key].astype(float)
                tsq[key][not_null_strobe] = NA

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', FutureWarning)
            tsq.sort_index(axis=1, inplace=True)

        return tsq

    @thunkify
    def _get_tsq_event(self, event_name):
        """Read the metadata (TSQ) file of a TDT Tank.

        Parameters
        ----------
        event_name : str

        Returns
        -------
        b : pandas.DataFrame
            Recording metadata
        """
        tsq = self.raw

        # make sure there's at least one event
        p = self.path

        # get the row of the metadata where its value equals the name-number
        num_names = tsq.name
        row = self.names[num_names].isin([event_name]).values
        assert row.any(), 'no event named %s in tank: %s' % (event_name, p)

        # get all the metadata for those events
        tsq = tsq[row]

        # convert to integer where possible
        try:
            tsq.channel = tsq.channel.astype(int)
            tsq.shank = tsq.shank.astype(int)
        except ValueError:
            pass

        first_row = row.argmax()
        return tsq, tsq.format[first_row], tsq.size[first_row]

    def tsq(self, event_name):
        return self._get_tsq_event(event_name)()

    @property
    def raw(self):
        return self._raw_tsq()()

    def _tev(self, event_name, clean=True):
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
        return self._read_tev(event_name, clean)()

    @thunkify
    def _read_tev(self, event_name, clean):
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
        meta, dtype, block_size = self.tsq(event_name)

        nchannels = meta.channel.dropna().nunique()
        nblocks = meta.shape[0]
        nsamples = nblocks * block_size // nchannels

        # raw ndarray for data
        spikes = DataFrame(np.empty((nblocks, block_size), dtype=dtype))

        tev_name = self.path + os.extsep + self._raw_ext

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', FutureWarning)
            meta.reset_index(drop=True, inplace=True)

        # convert timestamps to datetime objects
        meta.timestamp = fromtimestamp(meta.timestamp)
        meta.fp_loc = meta.fp_loc.astype(int)

        index = _create_ns_datetime_index(self.datetime, self.fs[event_name],
                                          nsamples)
        sdf = _read_tev(tev_name, meta, block_size, spikes, index,
                        self.electrode_map, clean)
        sdf.isclean = clean
        return sdf


PandasTank = TdtTank


def _create_ns_datetime_index(start, fs, nsamples, name='datetime'):
    """Create a DatetimeIndex in nanoseconds

    Parameters
    ----------
    start : datetime
    fs : float
    nsamples : int
    name : str, optional

    returns
    -------
    index : DatetimeIndex
    """
    ns = int(1e9 / fs)
    dtstart = np.datetime64(start)
    dt = dtstart + np.arange(nsamples) * np.timedelta64(ns, 'ns')
    freq = ns * pd.datetools.Nano()
    return DatetimeIndex(dt, freq=freq, name=name, tz=LOCAL_TZ)


def _reshape_spikes(df, group_inds):
    out = df.take(group_inds, axis=0)
    shp = out.shape
    shpsrt = np.argsort(shp)[::-1]
    nchannels = shp[shpsrt[-1]]
    return out.transpose(shpsrt).reshape(out.size // nchannels, -1)


def _read_tev(filename, meta, block_size, spikes, index, electrode_map, clean):
    assert isinstance(filename, basestring), 'filename must be a string'
    assert isinstance(block_size, (numbers.Integral, np.integer)), \
        'block_size must be an integer'
    assert ispower2(block_size), 'block_size must be a power of 2'
    assert isinstance(spikes, DataFrame), 'spikes must be a DataFrame'
    assert spikes.shape[1] == block_size, \
        'number of columns of spikes must equal block_size'
    assert isinstance(index, pd.Index), 'index must be an instance of Index'
    assert clean in (0, 1, False, True), 'clean must be a boolean or 0 or 1'

    return _read_tev_impl(filename, meta, block_size, spikes, index,
                          electrode_map, clean)


_raw_reader = _read_tev_raw


def _read_tev_impl(filename, meta, block_size, spikes, index, electrode_map,
                   clean):
    fp_loc, channel = meta.fp_loc, meta.channel
    _raw_reader(filename, fp_loc.values, block_size, spikes.values)

    items = spikes.groupby(channel).indices.items()
    items.sort()

    d = OrderedDict(items)

    group_inds = np.column_stack(d.itervalues())
    reshaped = _reshape_spikes(spikes.values, group_inds)
    raw = reshaped.take(electrode_map.channel, axis=1)
    df = SpikeDataFrame(raw, index, electrode_map.index, dtype=float)

    return remove_first_pc(df) if clean else df


if __name__ == '__main__':
    from span import ElectrodeMap, NeuroNexusMap
    span_data_path = os.environ['SPAN_DATA_PATH']
    elec_map = ElectrodeMap(NeuroNexusMap.values, 50, 125)
    f = os.path.join(span_data_path, 'Spont_Spikes_091210_p17rat_s4_657umV')
    tank = PandasTank(f, elec_map)
    sp = tank.spik
