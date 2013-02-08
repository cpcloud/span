import unittest
from itertools import product as cartprod

import numpy as np
from numpy.random import randn, randint
from numpy.testing import assert_allclose
from numpy import sum, abs

from pandas import DataFrame, Int64Index, Series

from span.xcorr import mult_mat_xcorr
from span.xcorr.xcorr import (xcorr, mult_mat_xcorr_numba,
                              mult_mat_xcorr_cython_parallel,
                              mult_mat_xcorr_cython_serial,
                              mult_mat_xcorr_numba_sliced,
                              mult_mat_xcorr_python)
from span.utils import (nextpow2, get_fft_funcs, detrend_none,
                        detrend_mean, detrend_linear, cartesian)


class TestXCorr(unittest.TestCase):
    def test_matrix_vs_numpy_correlate(self):
        m, n = 5, 3
        x = randn(m, n)

        rng = xrange(n)

        detrends = detrend_mean, detrend_none, detrend_linear, None
        scale_types = 'normalize', None, 'unbiased', 'biased'
        maxlags = None, 2, 20

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

            if ml > m:
                self.assertRaises(AssertionError, xcorr, x, detrend=dt,
                                  scale_type=st, maxlags=ml)
            else:
                xcc = xcorr(x, detrend=dt, scale_type=st, maxlags=ml)
                assert_allclose(xc_np, xc_span)
                dd = DataFrame(xc_np, index=np.r_[1 - m:m])

                # funky pandas df indexing must subtract one because endpoints
                # are inclusive
                assert_allclose(dd.ix[1 - mml:mml - 1].values, xcc)

                self.assertIsInstance(xcc, np.ndarray)

            if ml > m:
                self.assertRaises(AssertionError, xcorr, DataFrame(x),
                                  detrend=dt, scale_type=st, maxlags=ml)
            else:
                xccdf = xcorr(DataFrame(x), detrend=dt, scale_type=st,
                              maxlags=ml)
                assert_allclose(xccdf, xcc)
                self.assertIsInstance(xccdf, DataFrame)

            if ml > m:
                self.assertRaises(AssertionError, xcorr, Series(x[:, 0]),
                                  detrend=dt, scale_type=st, maxlags=ml)
                self.assertRaises(AssertionError, xcorr, x[:, 0],
                                  detrend=dt, scale_type=st, maxlags=ml)
            else:
                xc_s = xcorr(Series(x[:, 0]), detrend=dt, scale_type=st,
                             maxlags=ml)
                xc_s2 = xcorr(Series(x[:, 0]), y=None, detrend=dt,
                              scale_type=st, maxlags=ml)

                assert_allclose(xc_s, xc_s2)

            if ml > m:
                self.assertRaises(AssertionError, xcorr, Series(x[:, 0]),
                                  Series(x[:, 1]), detrend=dt, scale_type=st,
                                  maxlags=ml)
                self.assertRaises(AssertionError, xcorr, x[:, 0], x[:, 1],
                                  detrend=dt, scale_type=st, maxlags=ml)
            else:
                xc_s2 = xcorr(Series(x[:, 0]), Series(x[:, 1]), detrend=dt,
                              scale_type=st, maxlags=ml)
                xc_s2_np = xcorr(x[:, 0], x[:, 1], detrend=dt, scale_type=st,
                                 maxlags=ml)
                assert_allclose(xc_s2, xc_s2_np)

    def test_numpy_matrix_input(self):
        x = randn(10, 15)
        detrends = detrend_mean, detrend_none, detrend_linear, None
        scale_types = 'normalize', None, 'unbiased', 'biased'
        maxlags = 8, None, 100

        args = cartprod(maxlags, detrends, scale_types)

        for ml, dt, st in args:
            if ml > x.shape[0]:
                self.assertRaises(AssertionError, xcorr, x, detrend=dt,
                                  scale_type=st, maxlags=ml)
            else:
                xcnn = xcorr(x, detrend=dt, scale_type=st, maxlags=ml)
                self.assertIsInstance(xcnn, np.ndarray)

    def test_numpy_vector_input(self):
        x = randn(3)
        detrends = detrend_mean, detrend_none, detrend_linear, None
        scale_types = 'normalize', 'none', 'unbiased', 'biased', None
        maxlags = 2, None, 100

        args = cartprod(maxlags, detrends, scale_types)

        for ml, dt, st in args:
            y = randn(randint(2, 3))

            if ml > max(x.shape + y.shape):
                self.assertRaises(AssertionError, xcorr, x, detrend=dt,
                                  scale_type=st, maxlags=ml)
                self.assertRaises(AssertionError, xcorr, x, y, detrend=dt,
                                  scale_type=st, maxlags=ml)
                self.assertRaises(AssertionError, xcorr, x, y=None, detrend=dt,
                                  scale_type=st, maxlags=ml)
            else:
                xcnn = xcorr(x, detrend=dt, scale_type=st, maxlags=ml)
                xcnn2 = xcorr(x, y, detrend=dt, scale_type=st, maxlags=ml)
                xcnn3 = xcorr(x, y=None, detrend=dt, scale_type=st, maxlags=ml)

                self.assertIsInstance(xcnn, np.ndarray)
                self.assertIsInstance(xcnn2, np.ndarray)
                self.assertIsInstance(xcnn3, np.ndarray)


class TestMultMatXcorr(unittest.TestCase):
    def setUp(self):
        x = randn(randint(2, 4), randint(2, 4))
        m, n = x.shape
        ifft, fft = get_fft_funcs(x)
        nfft = int(2 ** nextpow2(m))
        X = fft(x.T, nfft)
        Xc = X.conj()
        self.n, nx = X.shape
        self.c = np.empty((n * n, nx), X.dtype)
        self.ground_truth = self.c.copy()

        self.X, self.Xc = X, Xc
        mult_mat_xcorr_python(X, Xc, self.ground_truth, n)

    def tearDown(self):
        del self.ground_truth, self.X, self.Xc, self.c, self.n

    def test_mult_mat_xcorrs(self):
        funcs = {mult_mat_xcorr_numba, mult_mat_xcorr_cython_parallel,
                 mult_mat_xcorr_cython_serial, mult_mat_xcorr_numba_sliced}

        for func in funcs:
            func(self.X, self.Xc, self.c, self.n)
            assert_allclose(self.c, self.ground_truth)

    def test_mult_mat_xcorr_high_level(self):
        assert_allclose(mult_mat_xcorr(self.X, self.Xc), self.ground_truth)
