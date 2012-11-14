import unittest
from warnings import warn
import os

from glob import glob

import numpy as np

from span.tdt import read_tev, PandasTank


class TestReadTev(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = os.path.join(os.path.expanduser('~'), 'Data', 'xcorr_data',
                                'Spont_Spikes_091210_p17rat_s4_657umV')
        cls.path = glob(os.path.join(path, '*%stev' % os.extsep))[0]
        cls.tank = PandasTank(cls.path[:-4])

    @classmethod
    def tearDownClass(cls):
        del cls.tank, cls.path

    def test_read_tev(self):
        names = 'Spik', 'LFPs'

        for name in names:
            tsq, _ = self.tank.tsq(name)
            fp_locs = tsq.fp_loc
            nsamples, chunk_size = fp_locs.size, tsq.size.unique().max()

            del tsq

            try:
                spikes = np.empty((nsamples, chunk_size), np.float32)
            except MemoryError:
                warn('Out of memory when creating TEV file output array')
            else:
                read_tev(self.path, chunk_size, fp_locs, spikes)

                # should be at least on the order of millivolts
                mag = np.log10(np.abs(spikes).mean())
                self.assertLessEqual(mag, -3.0)

                del spikes, mag

            del fp_locs
