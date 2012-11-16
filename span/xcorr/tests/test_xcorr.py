import unittest

import numpy as np
from numpy.testing import assert_allclose

from span.xcorr import mult_mat_xcorr
from span.xcorr.xcorr import xcorr
from span.utils import nextpow2, get_fft_funcs


class TestXCorr(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = 100
        cls.n = 50
        cls.matrix = np.random.randn(cls.m, cls.n)
        cls.vector = cls.matrix[:, np.random.randint(cls.n)]

    @classmethod
    def tearDownClass(cls):
        del cls.vector, cls.matrix, cls.m, cls.n

    def assert_matrix_shape(self, c):
        self.assertEqual(c.ndim, 2)

        m, n = c.shape

        self.assertEqual(m, 2 * self.m - 1)
        self.assertEqual(n, self.n ** 2)

    def assert_vector_shape(self, v):
        self.assertEqual(v.ndim, 1)

        m, = v.shape

        self.assertEqual(m, 2 * self.m - 1)

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

        def test_vector_vs_numpy_correlate(self):
            n = 10
            x, y = map(randn, itertools.repeat(n))
            xc_np = np.correlate(x, y, mode='full')
            xc_span = span.xcorr.xcorr(x, y, detrend=None, scale_type=None)

            assert_allclose(xc_np, xc_span.values)

        def test_matrix_vs_numpy_correlate(self):
            m, n = 20, 10
            x = randn(m, n)

            xc_np = np.zeros((2 * m - 1, n ** 2))

            for ci, i, j in zip(itertools.count(), xrange(n), xrange(n)):
                xi, xj = x[:, i], x[:, j]
                xc_np[:, ci] = np.correlate(xi, xj, mode='full')

            xc_span = span.xcorr.xcorr(x, detrend=None, scale_type=None)

            assert_allclose(xc_np, xc_span.values)


def test_mult_mat_xcorr():
    x = np.random.randn(np.random.randint(50, 71), np.random.randint(2, 21))
    m, n = x.shape
    ifft, fft = get_fft_funcs(x)
    nfft = int(2 ** nextpow2(m))
    X = fft(x.T, nfft)
    Xc = X.conj()
    mx, nx = X.shape

    c = np.empty((mx ** 2, nx), dtype=X.dtype)
    oc = np.empty((mx ** 2, nx), dtype=X.dtype)

    mult_mat_xcorr(X, Xc, oc, n, nx)

    for i in xrange(n):
        c[i * n:(i + 1) * n] = X[i] * Xc

    assert_allclose(c, oc)

    cc, occ = ifft(c, nfft).T, ifft(oc, nfft).T

    assert_allclose(cc, occ)
