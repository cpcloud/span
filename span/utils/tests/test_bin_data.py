import unittest

import numpy as np
from numpy.random import rand, randint

from span.utils import bin_data, cast


class TestBinDataNone(unittest.TestCase):
    def test_bin_data_none(self):
        x = rand(randint(25, 200), randint(10, 20)) > 0.5
        binsize = randint(10, max(x.shape))
        bins = cast(np.r_[:binsize:x.shape[0]], np.uint64)
        binned = np.empty((bins.shape[0] - 1, x.shape[1]), np.uint64)
        bin_data(x.view(np.uint8), bins, binned)
        self.assertRaises(ValueError, bin_data, x, bins, binned)
        self.assertTupleEqual(binned.shape, (bins.shape[0] - 1, x.shape[1]))
        self.assertEqual(binned.dtype, np.uint64)
