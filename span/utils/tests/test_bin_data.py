import unittest
import itertools

import numpy as np

from span.utils import bin_data, cast


class TestBinData(unittest.TestCase):
    def test_bin_data(self):
        binsizes = np.arange(4) + 1
        shapes = (11, 4), (4, 11), (10, 3), (3, 10), (2, 2)
        args = itertools.product(binsizes.tolist(), *zip(*shapes))

        for nrows, ncols, binsize in args:
            x = np.random.rand(nrows, ncols)

            bins = cast(np.r_[:binsize:x.shape[0]], np.uint64)
            binned = np.empty((bins.shape[0] - 1, x.shape[1]), np.uint64)
            bin_data(x.view(np.uint8), bins, binned)

            self.assertRaises(ValueError, bin_data, x, bins, binned)
            self.assertRaises(ValueError, bin_data, x.view(np.int8), bins,
                              binned)
            self.assertTupleEqual(binned.shape, (bins.size - 1, ncols))
            self.assertEqual(binned.dtype, np.uint64)
