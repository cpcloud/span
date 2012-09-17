"""
"""

import os
import abc
import re
import mmap
import contextlib

from itertools import imap as map, izip as zip

from span.tdt.spikeglobals import *
from span.tdt.spikedataframe import SpikeDataFrame
from span.tdt.decorate import thunkify, cached_property
from span.utils import name2num

class TdtTankBase(object):
    """
    """
    __metaclass__ = abc.ABCMeta
    fields = TsqFields
    np_types = TsqNumpyTypes
    date_re = re.compile(r'.*(\d{6}).*')
    site_re = re.compile(r'.*s(?:ite)?(?:\s|_)*(\d+)')
    header_ext = 'tsq'
    raw_ext = 'tev'
    tsq_dtype = np.dtype(list(zip(fields, np_types)))
    age_re = re.compile(r'.*[pP](\d+).*')

    def __init__(self, tankname):
        super(TdtTankBase, self).__init__()
        basename = os.path.basename(tankname)

        self.tankname = tankname
        self.animal_age = int(self.age_re.match(basename).group(1))

        self.__date = self._parse_date(basename)
        self.site = int(self.site_re.match(basename).group(1))

    def _parse_date(self, basename):
        try:
            date = self.date_re.match(basename).group(1)
        except AttributeError:
            now = pd.datetime.now()
            month, day, year = now.month, now.day, now.year
        else:
            datetmp = os.sep.join(i + j for i, j in zip(date[::2],
                                                         date[1::2])).split(os.sep)
            month, day, year = map(int, datetmp)

        return pd.datetime(year=year + 2000, month=month, day=day).date()

    @property
    def date(self): return str(self.__date)

    @abc.abstractmethod
    def _read_tev(self, event_name):
        pass

    @thunkify
    def _read_tsq(self):
        b = pd.DataFrame(np.fromfile(self.tankname + os.extsep + self.header_ext,
                                     dtype=self.tsq_dtype))
        b.channel = b.channel.astype(float) - 1
        b.channel[b.channel == -1] = np.nan
        shank = Indexer.shank[b.channel].reset_index(drop=True)
        side = Indexer.side[b.channel].reset_index(drop=True)
        return b.join(shank).join(side)

    @cached_property
    def tsq(self): return self._read_tsq()()

    def tev(self, event_name): return self._read_tev(event_name)()

    @cached_property
    def spikes(self): return self.tev('Spik')


class PandasTank(TdtTankBase):
    def __init__(self, tankname):
        super(PandasTank, self).__init__(tankname)

    @thunkify
    def _read_tev(self, event_name):
        """Read a TDT Tank tev files."""
        name = name2num(event_name)
        row = name == self.tsq.name
        table = ((np.float32, 1, np.float32),
                 (np.int32, 1, np.int32),
                 (np.int16, 2, np.int16),
                 (np.int8, 4, np.int8))
        first_row = (row == 1).argmax()
        fmt = self.tsq.format[first_row]
        chans = self.tsq.channel[row]
        fp_loc = self.tsq.file_pointer_location[row]
        nsamples = (self.tsq.size[first_row] - 10) * table[fmt][1]
        dtype = np.dtype(table[fmt][2]).type
        spikes = np.empty((fp_loc.size, nsamples), dtype=dtype)
        tev_name = self.tankname + os.extsep + self.raw_ext
        with open(tev_name, 'rb') as tev:
            with contextlib.closing(mmap.mmap(tev.fileno(), 0,
                                              access=mmap.ACCESS_READ)) as tev:
                for i, offset in enumerate(fp_loc):
                    spikes[i] = np.frombuffer(tev, dtype, nsamples, offset)
        shanks, side = self.tsq.shank[row], self.tsq.side[row]
        index = pd.MultiIndex.from_arrays((shanks, chans, side))
        return SpikeDataFrame(spikes, meta=self.tsq, index=index, dtype=dtype)
