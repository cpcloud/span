#!/usr/bin/env python

"""
"""

from future_builtins import map, zip

import os
import abc
import re
import mmap
import contextlib

import numpy as np
import pandas as pd

from span.tdt.spikeglobals import Indexer
from span.tdt.spikedataframe import SpikeDataFrame
from span.utils import name2num, thunkify, cached_property

TYPES_TABLE = ((np.float32, 1, np.float32),
               (np.int32, 1, np.int32),
               (np.int16, 2, np.int16),
               (np.int8, 4, np.int8))

TsqFields = ('size', 'type', 'name', 'channel', 'sort_code', 'timestamp',
             'fp_loc', 'format', 'fs')

TsqNumpyTypes = (np.int32, np.int32, np.uint32, np.uint16, np.uint16, np.float64,
                 np.int64, np.int32, np.float32)

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

    def __init__(self, tankname):
        super(TdtTankBase, self).__init__()
        assert isinstance(tankname, basestring), "tankname must be a string"
        basename = os.path.basename(tankname)

        self.tankname = tankname
        self.animal_age = int(self.age_re.match(basename).group(1))
        self.__date = self._parse_date(basename)

        try:
            site = int(self.site_re.match(basename).group(1))
        except (AttributeError, ValueError, TypeError):
            site = np.nan

        self.site = site

    def _parse_date(self, basename):
        """Parse a date from a directory name.

        Parameters
        ----------
        basename : str
            A directory name.

        Returns
        -------
        bdate : datetime.date
            The date parsed from the directory name.
        """
        try:
            date = self.date_re.match(basename).group(1)
        except AttributeError:
            now = pd.datetime.now()
            month, day, year = now.month, now.day, now.year
        else:
            dates = zip(date[::2], date[1::2])
            datetmp = os.sep.join(i + j for i, j in dates).split(os.sep)
            month, day, year = map(int, datetmp)

        return pd.datetime(year=year + 2000, month=month, day=day).date()

    @property
    def date(self):
        """Return the date of the recording as a date object.

        Returns
        -------
        self.__date : datetime.date
            The date of the recording.
        """
        return self.__date

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
        b = pd.DataFrame(np.fromfile(self.tankname + os.extsep + self.header_ext,
                                     dtype=self.tsq_dtype))
        b.channel = (b.channel - 1).astype(float)
        b.channel[b.channel == -1] = np.nan
        srt = Indexer.sort('channel').reset_index(drop=True)
        shank = srt.shank[b.channel].reset_index(drop=True)
        side = srt.side[b.channel].reset_index(drop=True)
        return b.join(shank).join(side)

    @cached_property
    def tsq(self): return self._read_tsq()

    def tev(self, event_name):
        """Return the data from a particular event."""
        return self._read_tev(event_name)()

    @cached_property
    def spikes(self): return self.tev('Spik')

    @cached_property
    def lfps(self): return self.tev('LFPs')


class PandasTank(TdtTankBase):
    """Encapsulate a TdtTankBase that returns its events as a special kind of
    Pandas DataFrame.

    Parameters
    ----------
    tankname : str
        Name of the tank files without the tsq or tev extension.

    See Also
    --------
    TdtTankBase
        Base class implementing metadata reading and properties
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
        name = name2num(event_name)
        row = self.tsq.name == name
        assert row.any(), 'no event named %s in tank: %s' % (event_name,
                                                             self.tankname)

        meta = self.tsq[row]
        meta.channel = meta.channel.astype(int)
        meta.shank = meta.shank.astype(int)
        meta.size -= 10

        first_row = np.argmax(row)

        fmt = meta.format[first_row]
        
        fp_loc = meta.fp_loc        

        nsamples = meta.size[first_row] * TYPES_TABLE[fmt][1]
        dtype = np.dtype(TYPES_TABLE[fmt][2]).type
        spikes = np.empty((fp_loc.size, nsamples), dtype=dtype)
        tev_name = self.tankname + os.extsep + self.raw_ext

        with open(tev_name, 'rb') as tev_fileobj:
            with contextlib.closing(mmap.mmap(tev_fileobj.fileno(), 0,
                                              access=mmap.ACCESS_READ)) as tev:
                for i, offset in enumerate(fp_loc):
                    spikes[i] = np.frombuffer(tev, dtype, nsamples, offset)

        index_arrays = (meta.side, meta.shank, meta.channel, meta.timestamp,
                        meta.fp_loc)
        index = pd.MultiIndex.from_arrays(index_arrays)
        return SpikeDataFrame(spikes, meta.reset_index(drop=True), index=index,
                              dtype=dtype)
