#!/usr/bin/env python

"""
Examples
--------
"""

from future_builtins import zip

import os
import abc
import re

import numpy as np
from numpy import (float32 as f4, int32 as i4, uint32 as u4, uint16 as u2,
                   float64 as f8, int64 as i8)
from pandas import Series, DataFrame, date_range, datetools

from span.tdt.spikeglobals import Indexer, EventTypes, DataTypes
from span.tdt.spikedataframe import SpikeDataFrame
from span.tdt._read_tev import read_tev

from span.utils import (name2num, thunkify, cached_property, fromtimestamp,
                        assert_nonzero_existing_file)


TsqFields = ('size', 'type', 'name', 'channel', 'sort_code', 'timestamp',
             'fp_loc', 'format', 'fs')

TsqNumpyTypes = i4, i4, u4, u2, u2, f8, i8, i4, f4


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


def _match_int(pattern, string, get_exc=False, excs=(AttributeError, ValueError,
                                                     TypeError)):
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
    r : int or None or tuple of int or None and Exception
    """
    try:
        r = int(_get_first_match(pattern, string))
    except excs as e:
        r = None

    if get_exc:
        r = r, e

    return r


class TdtTankAbstractBase(object):
    """Interface to tank reading methods.

    Attributes
    ----------
    tsq
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self):
        super(TdtTankAbstractBase, self).__init__()

    @abc.abstractmethod
    def _read_tev(self, event_name):
        pass

    @property
    @thunkify
    def _read_tsq(self):
        """Read the metadata (TSQ) file of a TDT Tank.

        Returns
        -------
        b : pandas.DataFrame
            Recording metadata
        """
        # create the path name
        tsq_name = self.path + os.extsep + self.header_ext

        # read in the raw data as a numpy rec array and conver to DataFrame
        b = DataFrame(np.fromfile(tsq_name, dtype=self.tsq_dtype))

        # zero based indexing
        b.channel -= 1
        b.channel = b.channel.astype(f8)

        # -1s are invalid
        b.channel[b.channel == -1] = np.nan

        b.type = EventTypes[b.type].reset_index(drop=True)
        b.format = DataTypes[b.format].reset_index(drop=True)

        b.timestamp[np.logical_not(b.timestamp)] = np.nan
        b.fs[np.logical_not(b.fs)] = np.nan

        # fragile subtraction (i.e., what if TDT changes this value?)
        b.size -= 10

        # create some new indices based on the electrode array
        srt = Indexer.sort('channel').reset_index(drop=True)
        shank = srt.shank[b.channel].reset_index(drop=True)
        side = srt.side[b.channel].reset_index(drop=True)

        return b.join(shank).join(side)

    @cached_property
    def tsq(self): return self._read_tsq()

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

    site_re
    age_re

    header_ext
    raw_ext

    path
    name
    age
    site
    """

    fields = TsqFields
    np_types = TsqNumpyTypes
    tsq_dtype = np.dtype(list(zip(fields, np_types)))
    types = Series(list(map(np.dtype, np_types)), index=fields)

    site_re = re.compile(r'(?:.*s(?:ite)?(?:|_)?(\d+))?')
    age_re = re.compile(r'.*[pP](\d+).*')

    header_ext = 'tsq'
    raw_ext = 'tev'

    def __init__(self, path):
        super(TdtTankBase, self).__init__()

        tank_with_ext = path + os.extsep
        tev_path = tank_with_ext + self.raw_ext
        tsq_path = tank_with_ext + self.header_ext

        assert_nonzero_existing_file(tev_path)
        assert_nonzero_existing_file(tsq_path)

        self.path = path
        self.name = os.path.basename(path)
        self.age = _match_int(self.age_re, self.name)
        self.site = _match_int(self.site_re, self.name)

    @cached_property
    def spikes(self): return self.tev('Spik')

    @cached_property
    def lfps(self): return self.tev('LFPs')


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
    def _read_tev(self, event_name):
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
        # convert the event_name to a number
        name = name2num(event_name)

        # get the row of the metadata where its value equals the name-number
        row = self.tsq.name == name

        # make sure there's at least one event
        assert row.any(), 'no event named %s in tank: %s' % (event_name,
                                                             self.path)

        # get all the metadata for those events
        meta = self.tsq[row]

        # convert to integer where possible
        meta.channel = meta.channel.astype(int)
        meta.shank = meta.shank.astype(int)

        # first row of event type
        first_row = np.argmax(row)

        # data type of this event
        dtype = meta.format[first_row]

        # number of samples per chunk
        nsamples = meta.size[first_row] * np.dtype(dtype).itemsize / 4

        # raw ndarray for data
        spikes = np.empty((meta.fp_loc.size, nsamples), dtype)

        # tev filename
        tev_name = self.path + os.extsep + self.raw_ext

        # read in the TEV data to spikes
        read_tev(tev_name, nsamples, meta.fp_loc, spikes)

        # convert timestamps to datetime objects
        meta.timestamp = fromtimestamp(meta.timestamp)

        # create the channel groups

        meta = meta.reset_index(drop=True)
        groups = DataFrame(meta.groupby('channel').groups).values

        # create a thunk to start this computation
        sdf = _reshape_spikes(spikes, groups, meta, meta.fs.unique().item(),
                              meta.channel.dropna().nunique(),
                              meta.timestamp[0])
        return sdf()


@thunkify
def _reshape_spikes(raw, group_indices, meta, fs, nchans, date):
    """Reshape the spikes to a :math:`m\times n` ``SpikeDataFrame`` where
    :math:`m` is the number of samples and :math:`n` is the number of channels.

    Parameters
    ----------
    raw : array_like
    group_indices : array_like
    meta : DataFrame
    fs : float
    nchans : int
    date : datetime

    Returns
    -------
    reshaped_spikes : SpikeDataFrame
    """
    from span.tdt.spikeglobals import ChannelIndex as columns

    vals = raw[group_indices]
    shpsort = np.argsort(vals.shape)[::-1]
    newshp = int(vals.size / nchans), nchans
    valsr = vals.transpose(shpsort).reshape(newshp)

    us_per_sample = round(1e6 / fs) * datetools.Micro()
    index = date_range(date, periods=max(valsr.shape), freq=us_per_sample,
                       name='time', tz='US/Eastern')
    return SpikeDataFrame(valsr, meta, index=index, columns=columns)
