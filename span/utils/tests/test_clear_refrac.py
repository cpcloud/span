import unittest

import numpy as np
from numpy.random import rand, randint

from span.utils import clear_refrac, nextpow2


class TestClearRefracModule(unittest.TestCase):
    def setUp(self):
        m, n = randint(10, 101), randint(4, 5)
        ms = randint(2, 10)
        fs = int(2 ** nextpow2(randint(1, m / ms)))

        self.x = rand(m, n)
        self.window = int(np.floor(ms / 1e3 * fs))

        if not self.window or self.window < 0:
            self.window = abs(self.window) + 1

    def tearDown(self):
        del self.window, self.x

    def test_many_threshes(self):
        thr = self.x > rand(self.x.shape[1])
        cleared = thr.copy()
        clear_refrac(thr.view(np.uint8), self.window)
        self.assertRaises(ValueError, clear_refrac, thr, self.window)
        self.assertFalse(np.array_equal(thr, cleared))

    def test_one_thresh(self):
        thr = self.x > rand()
        cleared = thr.copy()
        clear_refrac(thr.view(np.uint8), self.window)
        self.assertRaises(ValueError, clear_refrac, thr, self.window)
        self.assertFalse(np.array_equal(thr, cleared))
