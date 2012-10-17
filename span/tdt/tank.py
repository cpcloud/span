#!/usr/bin/env python

"""Encapsulate TDT's Tank files.
"""

from future_builtins import map, zip

import os
import abc
import re

import numpy as np
from numpy import (float32, int32, int16, int8, uint32, uint16, float64, int64,
                   nan)
import pandas as pd
from pandas import DataFrame, MultiIndex

from span.tdt.spikeglobals import Indexer
from span.tdt.spikedataframe import SpikeDataFrame
from span.tdt._read_tev import read_tev

from span.utils import name2num, thunkify, cached_property


TYPES_TABLE = ((float32, 1, float32),
               (int32, 1, int32),
               (int16, 2, int16),
               (int8, 4, int8))

TsqFields = ('size', 'type', 'name', 'channel', 'sort_code', 'timestamp',
             'fp_loc', 'format', 'fs')

TsqNumpyTypes = (int32, int32, uint32, uint16, uint16, float64, int64, int32,
                 float32)


def nonzero_existing_file(f):
    return os.path.exists(f) and os.path.isfile(f) and os.path.getsize(f) > 0


def get_first_match(pattern, string):
    return re.match(pattern, string).group(1)


def match_int(pattern, string, get_exc=False, excs=(AttributeError, ValueError,
                                                    TypeError)):
    """Convert a string matched from a regex to an integer or return None.

    Parameters
    ----------
    pattern : str or compiled regex
    string : str
    get_exc : bool, optional
    excs : Exception, optional

    Returns
    -------
    r : int or None or tuple of int or None and Exception
    """
    try:
        r = int(get_first_match(pattern, string))
    except excs as e:
        r = None

    if get_exc:
        r = r, e

    return r


class TdtTankBase(object):
    """Abstract base class encapsulating a TDT Tank.

    Parameters
    ----------
    tankname : str

    Attributes
    ----------
    fields
    np_types
    tsq_dtype

    date_re
    site_re
    age_re

    header_ext
    raw_ext

    date
    _read_tsq
    """
    __metaclass__ = abc.ABCMeta

    fields = TsqFields
    np_types = TsqNumpyTypes
    tsq_dtype = np.dtype(list(zip(fields, np_types)))

    date_re = re.compile(r'.*(\d{6}).*')
    site_re = re.compile(r'(.*s(?:ite)?(?:\s|_)*(\d+))?')
    age_re = re.compile(r'.*[pP](\d+).*')

    header_ext = 'tsq'
    raw_ext = 'tev'

    def __init__(self, path):
        super(TdtTankBase, self).__init__()

        tank_with_ext = path + os.extsep

        assert nonzero_existing_file(tank_with_ext + self.raw_ext), \
            '{0} does not exist'.format(tank_with_ext + self.raw_ext)

        assert nonzero_existing_file(tank_with_ext + self.header_ext), \
            '{0} does not exist'.format(tank_with_ext + self.header_ext)

        self.__path = path
        self.__name = os.path.basename(path)
        self.__age = match_int(self.age_re, self.name)
        self.__site = match_int(self.site_re, self.name)
        self.__date = self._parse_date(self.name)

    @property
    def date(self): return self.__date

    @property
    def path(self): return self.__path

    @property
    def name(self): return self.__name

    @property
    def age(self): return self.__age

    @property
    def site(self): return self.__site

    def _parse_date(self, basename):
        """Parse a date from a directory name.

        Parameters
        ----------
        basename : str
            A directory name.

        Returns
        -------
        date : datetime.date
            The date parsed from the directory name.
        """
        try:
            date = self.date_re.match(basename).group(1)
        except AttributeError:
            now = pd.datetime.now()
            month, day, year = now.month, now.day, now.year
        else:
            sep = '/'
            dates = zip(date[::2], date[1::2])
            datetmp = os.sep.join(i + j for i, j in dates).split(sep)
            month, day, year = map(int, datetmp)

        return pd.datetime(year=year + 2000, month=month, day=day).date()

    @abc.abstractmethod
    def _read_tev(self, event_name):
        """Read a TDT *.tev file and parse a particular set of events.

        Parameters
        ----------
        event_name : str
            The name of the event, must be 4 letters long

        Returns
        -------
        d : span.tdt.SpikeDataFrame
            An instance of span.tdt.SpikeDataFrame with a few extra methods
            that are commonly used in spike analysis.
        """
        pass

    @property
    @thunkify
    def _read_tsq(self):
        """Read the meta data file of a TDT Tank.

        Returns
        -------
        b : pandas.DataFrame
        """
        tsq_name = self.path + os.extsep + self.header_ext
        raw = np.fromfile(tsq_name, dtype=self.tsq_dtype)
        b = DataFrame(raw)
        b.channel = (b.channel - 1).astype(float64)
        b.channel[b.channel == -1] = nan
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

        Returns
        -------
        data : SpikeDataFrame
        """
        return self._read_tev(event_name)()

    @cached_property
    def spikes(self): return self.tev('Spik')

    @cached_property
    def lfps(self): return self.tev('LFPs')


class PandasTank(TdtTankBase):
    """Implement the abstract methods from TdtTankBase.

    Parameters
    ----------
    tankname : str
        Name of the tank files without the tsq or tev extension.

    See Also
    --------
    TdtTankBase
        Base class implementing metadata reading and properties.
    """
    def __init__(self, tankname):
        super(PandasTank, self).__init__(tankname)

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

        # fragile subtraction
        meta.size -= 10

        # first row of event type
        first_row = np.argmax(row)

        # data type of this event
        fmt = meta.format[first_row]

        # locations of samples in the TEV file
        fp_loc = meta.fp_loc

        # number of samples per chunk
        nsamples = meta.size[first_row] * TYPES_TABLE[fmt][1]

        # dtype of event type
        dtype = np.dtype(TYPES_TABLE[fmt][2]).type

        # raw ndarray for data
        spikes = np.empty((fp_loc.size, nsamples), dtype=dtype)

        # tev filename
        tev_name = self.path + os.extsep + self.raw_ext

        # read in the TEV data to spikes
        read_tev(tev_name, nsamples, fp_loc, spikes)

        # create a pandas MultiIndex with metadata
        index_arrays = (meta.side, meta.shank, meta.channel, meta.timestamp,
                        meta.fp_loc)
        index = MultiIndex.from_arrays(index_arrays)

        # create a spike data frame with a bunch of the meta data
        return SpikeDataFrame(spikes, meta.reset_index(drop=True),
                              date=self.date, index=index, dtype=dtype)
