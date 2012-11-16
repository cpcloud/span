import unittest
from warnings import warn
import os

from glob import glob

import numpy as np

from span.tdt import read_tev, PandasTank


class TestReadTev(unittest.TestCase):
    def setUp(self):
        home = os.path.expanduser('~')
        path = os.path.join(home, 'Data', 'xcorr_data',
                            'Spont_Spikes_091210_p17rat_s4_657umV')
        self.path = glob(os.path.join(path, '*%stev' % os.extsep))[0]
        self.tank = PandasTank(self.path[:-4])
        self.names = 'Spik', 'LFPs'

    def tearDown(self):
        del self.names, self.tank, self.path

    def test_read_tev(self):

        for name in self.names:
            tsq, _ = self.tank.tsq(name)
            fp_locs = tsq.fp_loc
            nsamples, chunk_size = fp_locs.size, tsq.size.unique().max()

            del tsq

            spikes = np.empty((nsamples, chunk_size), np.float32)

            read_tev(self.path, chunk_size, fp_locs, spikes)

            # should be at least on the order of millivolts
            mag = np.log10(np.abs(spikes).mean())
            self.assertLessEqual(mag, -3.0)

            del spikes, mag, fp_locs
