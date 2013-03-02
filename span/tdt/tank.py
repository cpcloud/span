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
import warnings

try:
    from collections import OrderedDict
except ImportError:
    try:
        from thread import get_ident as _get_ident
    except ImportError:
        from dummy_thread import get_ident as _get_ident

    try:
        from _abcoll import KeysView, ValuesView, ItemsView
    except ImportError:
        pass

    class OrderedDict(dict):
        'Dictionary that remembers insertion order'
        # An inherited dict maps keys to values.
        # The inherited dict provides __getitem__, __len__, __contains__, and get.
        # The remaining methods are order-aware.
        # Big-O running times for all methods are the same as for regular
        # dictionaries.

        # The internal self.__map dictionary maps keys to links in a doubly linked list.
        # The circular doubly linked list starts and ends with a sentinel element.
        # The sentinel element never gets deleted (this simplifies the algorithm).
        # Each link is stored as a list of length three:  [PREV, NEXT, KEY].

        def __init__(self, *args, **kwds):
            '''Initialize an ordered dictionary.  Signature is the same as for
            regular dictionaries, but keyword arguments are not recommended
            because their insertion order is arbitrary.

            '''
            if len(args) > 1:
                raise TypeError(
                    'expected at most 1 arguments, got %d' % len(args))
            try:
                self.__root
            except AttributeError:
                self.__root = root = []                     # sentinel node
                root[:] = [root, root, None]
                self.__map = {}
            self.__update(*args, **kwds)

        def __setitem__(self, key, value, dict_setitem=dict.__setitem__):
            'od.__setitem__(i, y) <==> od[i]=y'
            # Setting a new item creates a new link which goes at the end of the linked
            # list, and the inherited dictionary is updated with the new
            # key/value pair.
            if key not in self:
                root = self.__root
                last = root[0]
                last[1] = root[0] = self.__map[key] = [last, root, key]
            dict_setitem(self, key, value)

        def __delitem__(self, key, dict_delitem=dict.__delitem__):
            'od.__delitem__(y) <==> del od[y]'
            # Deleting an existing item uses self.__map to find the link which is
            # then removed by updating the links in the predecessor and
            # successor nodes.
            dict_delitem(self, key)
            link_prev, link_next, key = self.__map.pop(key)
            link_prev[1] = link_next
            link_next[0] = link_prev

        def __iter__(self):
            'od.__iter__() <==> iter(od)'
            root = self.__root
            curr = root[1]
            while curr is not root:
                yield curr[2]
                curr = curr[1]

        def __reversed__(self):
            'od.__reversed__() <==> reversed(od)'
            root = self.__root
            curr = root[0]
            while curr is not root:
                yield curr[2]
                curr = curr[0]

        def clear(self):
            'od.clear() -> None.  Remove all items from od.'
            try:
                for node in self.__map.itervalues():
                    del node[:]
                root = self.__root
                root[:] = [root, root, None]
                self.__map.clear()
            except AttributeError:
                pass
            dict.clear(self)

        def popitem(self, last=True):
            '''od.popitem() -> (k, v), return and remove a (key, value) pair.
            Pairs are returned in LIFO order if last is true or FIFO order if false.

            '''
            if not self:
                raise KeyError('dictionary is empty')
            root = self.__root
            if last:
                link = root[0]
                link_prev = link[0]
                link_prev[1] = root
                root[0] = link_prev
            else:
                link = root[1]
                link_next = link[1]
                root[1] = link_next
                link_next[0] = root
            key = link[2]
            del self.__map[key]
            value = dict.pop(self, key)
            return key, value

        # -- the following methods do not depend on the internal structure --

        def keys(self):
            'od.keys() -> list of keys in od'
            return list(self)

        def values(self):
            'od.values() -> list of values in od'
            return [self[key] for key in self]

        def items(self):
            'od.items() -> list of (key, value) pairs in od'
            return [(key, self[key]) for key in self]

        def iterkeys(self):
            'od.iterkeys() -> an iterator over the keys in od'
            return iter(self)

        def itervalues(self):
            'od.itervalues -> an iterator over the values in od'
            for k in self:
                yield self[k]

        def iteritems(self):
            'od.iteritems -> an iterator over the (key, value) items in od'
            for k in self:
                yield (k, self[k])

        def update(*args, **kwds):
            '''od.update(E, **F) -> None.  Update od from dict/iterable E and F.

            If E is a dict instance, does:           for k in E: od[k] = E[k]
            If E has a .keys() method, does:         for k in E.keys(): od[k] = E[k]
            Or if E is an iterable of items, does:   for k, v in E: od[k] = v
            In either case, this is followed by:     for k, v in F.items(): od[k] = v

            '''
            if len(args) > 2:
                raise TypeError('update() takes at most 2 positional '
                                'arguments (%d given)' % (len(args),))
            elif not args:
                raise TypeError('update() takes at least 1 argument (0 given)')
            self = args[0]
            # Make progressively weaker assumptions about "other"
            other = ()
            if len(args) == 2:
                other = args[1]
            if isinstance(other, dict):
                for key in other:
                    self[key] = other[key]
            elif hasattr(other, 'keys'):
                for key in other.keys():
                    self[key] = other[key]
            else:
                for key, value in other:
                    self[key] = value
            for key, value in kwds.items():
                self[key] = value

        __update = update  # let subclasses override update without breaking __init__

        __marker = object()

        def pop(self, key, default=__marker):
            '''od.pop(k[,d]) -> v, remove specified key and return the corresponding value.
            If key is not found, d is returned if given, otherwise KeyError is raised.

            '''
            if key in self:
                result = self[key]
                del self[key]
                return result
            if default is self.__marker:
                raise KeyError(key)
            return default

        def setdefault(self, key, default=None):
            'od.setdefault(k[,d]) -> od.get(k,d), also set od[k]=d if k not in od'
            if key in self:
                return self[key]
            self[key] = default
            return default

        def __repr__(self, _repr_running={}):
            'od.__repr__() <==> repr(od)'
            call_key = id(self), _get_ident()
            if call_key in _repr_running:
                return '...'
            _repr_running[call_key] = 1
            try:
                if not self:
                    return '%s()' % (self.__class__.__name__,)
                return '%s(%r)' % (self.__class__.__name__, self.items())
            finally:
                del _repr_running[call_key]

        def __reduce__(self):
            'Return state information for pickling'
            items = [[k, self[k]] for k in self]
            inst_dict = vars(self).copy()
            for k in vars(OrderedDict()):
                inst_dict.pop(k, None)
            if inst_dict:
                return (self.__class__, (items,), inst_dict)
            return self.__class__, (items,)

        def copy(self):
            'od.copy() -> a shallow copy of od'
            return self.__class__(self)

        @classmethod
        def fromkeys(cls, iterable, value=None):
            '''OD.fromkeys(S[, v]) -> New ordered dictionary with keys from S
            and values equal to v (which defaults to None).

            '''
            d = cls()
            for key in iterable:
                d[key] = value
            return d

        def __eq__(self, other):
            '''od.__eq__(y) <==> od==y.  Comparison to another OD is order-sensitive
            while comparison to a regular mapping is order-insensitive.

            '''
            if isinstance(other, OrderedDict):
                return len(self) == len(other) and self.items() == other.items()
            return dict.__eq__(self, other)

        def __ne__(self, other):
            return not self == other

        # -- the following methods are only used in Python 2.7 --

        def viewkeys(self):
            "od.viewkeys() -> a set-like object providing a view on od's keys"
            return KeysView(self)

        def viewvalues(self):
            "od.viewvalues() -> an object providing a view on od's values"
            return ValuesView(self)

        def viewitems(self):
            "od.viewitems() -> a set-like object providing a view on od's items"
            return ItemsView(self)
    ## end of http://code.activestate.com/recipes/576693/ }}}


import numpy as np
from numpy import nan as NA
from pandas import DataFrame, DatetimeIndex
import pandas as pd
from six.moves import xrange
from pytz import UnknownTimeZoneError

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
        tsq = DataFrame(np.fromfile(tsq_name, self.dtype))
        inds = tsq.strobe <= np.finfo(np.float64).eps
        tsq.strobe[inds] = NA

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

        tev_name = self.path + os.extsep + self._raw_ext
        meta.reset_index(drop=True, inplace=True)

        # convert timestamps to datetime objects (vectorized)
        meta.timestamp = fromtimestamp(meta.timestamp)

        index = _create_ns_datetime_index(self.datetime, self.fs, nsamples)
        return _read_tev(tev_name, meta.fp_loc.values, block_size,
                         meta.channel.values, meta.shank.values,
                         DataFrame(spikes), index,
                         columns.reorder_levels((1, 0)))


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

    try:
        return DatetimeIndex(dt, freq=ns * pd.datetools.Nano(), name='time',
                             tz='US/Eastern')
    except UnknownTimeZoneError:
        warnings.warn('Time zone not found, you might need to reinstall '
                      'pytz or matplotlib or both', RuntimeWarning)
        return DatetimeIndex(dt, freq=ns * pd.datetools.Nano(), name='time')


def _reshape_spikes(df, group_inds):
    reshaped = df.take(group_inds, axis=0)
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
    assert isinstance(spikes, (np.ndarray, DataFrame)), \
        'spikes must be an ndarray or a DataFrame'
    assert spikes.shape[1] == block_size, \
        'number of columns of spikes must equal block_size'
    assert isinstance(index, pd.Index), 'index must be an instance of Index'
    assert isinstance(columns, pd.Index), \
        'columns must be an instance of Index'
    return _read_tev_impl(filename, fp_locs, block_size, channel, shank,
                          spikes, index, columns)


def _read_tev_impl(filename, fp_locs, block_size, channel, shank, spikes,
                   index, columns):
    _read_tev_raw(filename, fp_locs, block_size, spikes.values)

    items = spikes.groupby(channel).indices.items()
    items.sort()

    group_inds = np.column_stack(OrderedDict(items).itervalues())
    reshaped = _reshape_spikes(spikes.values, group_inds)
    d = DataFrame(reshaped, index, columns)
    d.sort_index(axis=1, inplace=True)
    return SpikeDataFrame(d, dtype=float)


if __name__ == '__main__':
    f = ('/home/phillip/Data/xcorr_data/Spont_Spikes_091210_p17rat_s4_'
         '657umV/Spont_Spikes_091210_p17rat_s4_657umV')
    tank = PandasTank(f)
    sp = tank.spikes
