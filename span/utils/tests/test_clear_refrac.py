import unittest

import numpy as np
from numpy.random import rand, randint

from span.utils import clear_refrac, nextpow2


class TestClearRefracModule(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        m, n = randint(1000, 10001), randint(10, 20)
        ms = randint(2, 10)
        fs = int(2 ** nextpow2(randint(1, m / ms)))

        cls.x = rand(m, n)
        cls.window = int(np.floor(ms / 1e3 * fs))

    def test_many_threshes(self):
        thr = self.x > rand(self.x.shape[1])
        cleared = thr.copy()
        clear_refrac(thr, self.window)
        self.assertFalse(np.array_equal(thr, cleared))

    def test_one_thresh(self):
        thr = self.x > rand()
        cleared = thr.copy()
        clear_refrac(thr, self.window)
        self.assertFalse(np.array_equal(thr, cleared))
