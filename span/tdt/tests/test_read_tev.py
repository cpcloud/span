import unittest
import os

from glob import glob

import numpy as np

from span.tdt import read_tev, PandasTank

class TestReadTev(unittest.TestCase):
    """Test the Cython'ed read_tev function.
    """
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
        tsq = self.tank.tsq
        fp_locs = tsq.fp_loc
        nsamples, chunk_size = tsq.shape[0], tsq.size.unique().max()

        del tsq

        try:
            spikes = np.empty((nsamples, chunk_size), np.float32)
        except MemoryError:
            pass
        else:
            read_tev(self.path, nsamples, fp_locs, spikes)

            # should be on the order of millivolts
            mag = np.log10(np.abs(spikes).mean())
            self.assertLessEqual(mag, -3.0)
