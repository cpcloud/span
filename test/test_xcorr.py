import unittest

import numpy as np

from span.tdt.xcorr import xcorr

class TestXCorr(unittest.TestCase):
    def setUp(self):
        self.m = 600
        self.n = 20
        self.matrix = np.random.randn(self.m, self.n)
        self.vector = self.matrix[:, np.random.randint(self.n)]

    def assert_matrix_shape(self, c):
        assert c.ndim == 2
        
        m, n = c.shape
        
        assert m == 2 * self.m - 1
        assert n == self.n ** 2, 'n:{}, wanted: {}'.format(n, self.n ** 2)

    def assert_vector_shape(self, v):
        assert v.ndim == 1
        
        m, = v.shape
        
        assert m == 2 * self.m - 1

    def test_all(self):
        scale_types = None, 'none', 'unbiased', 'biased', 'normalize'
        for scale_type in scale_types:
            c = xcorr(self.matrix, scale_type=scale_type)
            self.assert_matrix_shape(c)
            
            v = xcorr(self.vector, scale_type=scale_type)
            self.assert_vector_shape(v)
