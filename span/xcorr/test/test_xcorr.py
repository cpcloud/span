import unittest

import numpy as np
from numpy.testing import assert_allclose

from span.tdt.xcorr import xcorr


class TestXCorr(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = 100
        cls.n = 50
        cls.matrix = np.random.randn(cls.m, cls.n)
        cls.vector = cls.matrix[:, np.random.randint(cls.n)]

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

            vbig, vsmall = np.random.randn(self.m), np.random.randn(self.m - 20)
            vbs = xcorr(vbig, vsmall, scale_type=scale_type)
            self.assert_vector_shape(vbs)

            vbb = xcorr(vbig, vbig, scale_type=scale_type)
            self.assert_vector_shape(vbb)

            self.assertRaises(AssertionError, xcorr, self.matrix,
                              scale_type=scale_type,
                              maxlags=self.matrix.shape[0] +
                              10)
            self.assertRaises(AssertionError, xcorr, vbig, vsmall,
                              scale_type=scale_type,
                              maxlags=vbig.size + 10)
            self.assertRaises(AssertionError, xcorr, self.vector,
                              scale_type=scale_type,
                              maxlags=self.vector.size + 10)
            self.assertRaises(AssertionError, xcorr, vbig, scale_type=scale_type,
                              maxlags=vbig.size + 10)
            if scale_type == 'normalize':
                _, nc = c.shape
                ncsqrt = int(np.sqrt(nc))
                jkl = np.diag(np.r_[:nc].reshape((ncsqrt, ncsqrt)))

                # lag 0s must be 1.0 for normalized
                assert_allclose(c.ix[0, jkl], 1.0)
                assert v.ix[0] == 1.0