import unittest
from itertools import product as cartprod

import numpy as np
from numpy.random import randn, randint
from numpy.testing import assert_allclose
from numpy import sum, abs

from pandas import DataFrame, Int64Index

from span.xcorr import mult_mat_xcorr
from span.xcorr.xcorr import xcorr
from span.utils import (nextpow2, get_fft_funcs, detrend_none,
                        detrend_mean, detrend_linear, cartesian)


class TestXCorr(unittest.TestCase):
    def setUp(self):
        self.m, self.n = 100, 50
        self.matrix = randn(self.m, self.n)
        self.vector = self.matrix[:, randint(self.n)]

    def tearDown(self):
        del self.vector, self.matrix, self.m, self.n

    def test_matrix_vs_numpy_correlate(self):
        m, n = 20, 10
        x = randn(m, n)

        rng = xrange(n)

        detrends = detrend_mean, detrend_none, detrend_linear
        scale_types = 'normalize', 'none', 'unbiased', 'biased', None
        maxlags = 10, None

        args = cartprod(maxlags, detrends, scale_types)

        xc_np = np.zeros((2 * m - 1, n ** 2))
        cart = cartesian((np.arange(n), np.arange(n)))

        cols = map(tuple, cart)
        for ml, dt, st in args:
            xc_np.fill(0)
            mml = ml if ml is not None else m

            xc_span = DataFrame(xc_np.copy(), columns=cols,
                                index=Int64Index(np.r_[1 - m:m]))

            for ci, (i, j) in enumerate(cartprod(rng, rng)):
                xi, xj = dt(x[:, i]), dt(x[:, j])

                xc_np[:, ci] = np.correlate(xi, xj, mode='full')

                if st == 'normalize':
                    cxx0 = sum(abs(xi) ** 2)
                    cyy0 = sum(abs(xj) ** 2)
                    xc_np[:, ci] /= np.sqrt(cxx0 * cyy0)

                elif st == 'biased':
                    xc_np[:, ci] /= m

                elif st == 'unbiased':
                    xc_np[:, ci] /= m - np.abs(np.r_[1 - m:m])

                xc_span[i, j] = xcorr(xi, xj, detrend=dt, scale_type=st)

            assert_allclose(xc_np, xc_span)

            xc = xcorr(x, detrend=dt, scale_type=st, maxlags=ml)
            assert_allclose(xc_span.ix[1 - mml:mml - 1], xc)

    def test_numpy_matrix_input(self):
        x = randn(10, 15)
        detrends = detrend_mean, detrend_none, detrend_linear
        scale_types = 'normalize', 'none', 'unbiased', 'biased', None
        maxlags = 10, None

        args = cartprod(maxlags, detrends, scale_types)

        for ml, dt, st in args:
            xcnn = xcorr(x, detrend=dt, scale_type=st, maxlags=ml)
            self.assertIsInstance(xcnn, np.ndarray)

    def test_numpy_vector_input(self):
        x = randn(10)
        detrends = detrend_mean, detrend_none, detrend_linear
        scale_types = 'normalize', 'none', 'unbiased', 'biased', None
        maxlags = 10, None

        args = cartprod(maxlags, detrends, scale_types)

        for ml, dt, st in args:
            y = randn(10)
            xcnn = xcorr(x, detrend=dt, scale_type=st, maxlags=ml)
            xcnn2 = xcorr(x, y, detrend=dt, scale_type=st, maxlags=ml)
            xcnn3 = xcorr(x, y=None, detrend=dt, scale_type=st, maxlags=ml)

            self.assertIsInstance(xcnn, np.ndarray)
            self.assertIsInstance(xcnn2, np.ndarray)
            self.assertIsInstance(xcnn3, np.ndarray)


def test_mult_mat_xcorr():
    x = randn(randint(50, 71), randint(2, 21))
    m, n = x.shape
    ifft, fft = get_fft_funcs(x)
    nfft = int(2 ** nextpow2(m))
    X = fft(x.T, nfft)
    Xc = X.conj()
    mx, nx = X.shape

    c = np.empty((mx ** 2, nx), dtype=X.dtype)
    oc = c.copy()

    mult_mat_xcorr(X, Xc, oc, n, nx)

    for i in xrange(n):
        c[i * n:(i + 1) * n] = X[i] * Xc

    assert_allclose(c, oc)

    cc, occ = map(lambda x: ifft(x, nfft).T, (c, oc))

    assert_allclose(cc, occ)
