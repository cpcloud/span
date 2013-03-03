import unittest

import numpy as np
from numpy.random import rand, randint

from span.utils import clear_refrac, nextpow2


class TestClearRefrac(unittest.TestCase):
    def setUp(self):
        m, n = 20, 4
        fs = ms = 2

        self.x = rand(m, n)
        self.window = int(np.floor(ms / 1e3 * fs))

        if not self.window:
            self.window += 1

    def tearDown(self):
        del self.window, self.x

    def test_many_threshes(self):
        thr = self.x > rand(self.x.shape[1])
        cleared = thr.copy()
        clear_refrac(cleared, self.window)
        self.assertFalse(np.array_equal(thr, cleared))

    def test_one_thresh(self):
        thr = self.x > 0.5
        cleared = thr.copy()
        clear_refrac(cleared, self.window)

        self.assertFalse(np.array_equal(thr, cleared))
