
import unittest

import numpy as np
from numpy.random import randn, randint
from numpy.testing import assert_allclose

from pandas import DataFrame, Series
from pandas.util.testing import assert_frame_equal

from span.xcorr import mult_mat_xcorr, xcorr
from span.xcorr.xcorr import (mult_mat_xcorr_numba,
                              mult_mat_xcorr_cython_parallel,
                              mult_mat_xcorr_cython_serial,
                              mult_mat_xcorr_numba_sliced,
                              mult_mat_xcorr_python, crosscorr, autocorr,
                              matrixcorr)
from span.utils import (nextpow2, get_fft_funcs, detrend_none,
                        detrend_mean, detrend_linear, cartesian)
from span.testing import assert_array_equal, knownfailure


def is_symmetric(x):
    return np.array_equal(x, x.T)


def assert_symmetric(x):
    methods = {DataFrame: assert_frame_equal,
               np.ndarray: assert_array_equal}
    method = methods[type(x)]
    return method(x, x.T)


def row(x):
    m = x.shape[0]
    return np.tile(np.r_[:m][:, np.newaxis], m)


def col(x):
    n = x.shape[1]
    return np.tile(np.r_[:n][np.newaxis], (n, 1))


def correlate1d(x, y):
    xc = x - x.mean()
    yc = y - y.mean()
    z = np.sqrt(np.sum(np.abs(xc) ** 2) * np.sum(np.abs(yc) ** 2))
    return np.correlate(xc, yc, mode='full') / z


class TestXCorr(unittest.TestCase):
    def setUp(self):
        self.m, self.n = 3, 2
        self.x, self.y = randn(self.m), randn(self.n)
        self.xsame, self.ysame = self.x, randn(self.m)
        self.matrix = randn(self.m, self.n)

    def test_autocorr(self):
        x = self.x
        npc = np.correlate(x, x, mode='full')
        spc = xcorr(x)
        assert_allclose(npc, spc)

    def test_crosscorr_same_lengths(self):
        x, y = self.xsame, self.ysame
        npc = np.correlate(x, y, mode='full')
        spc = xcorr(x, y)
        assert_allclose(npc, spc)

    @knownfailure
    def test_crosscorr_diff_lengths(self):
        # this fails because np.correlate does something strange to
        # the input: it removes the padding that it adds
        # the span.xcorr.xcorr function replicates MATLAB's xcorr's behavior
        x, y = self.x, self.y
        npc = np.correlate(x, y, mode='full')
        spc = xcorr(x, y)
        assert_allclose(npc, spc)

    def test_lag0_matrix(self):
        x = self.matrix
        df = DataFrame(x)
        np_lag0 = np.corrcoef(x.T)
        pd_lag0 = df.corr()
        sp_lag0 = xcorr(df, detrend=detrend_mean, scale_type='normalize')

        assert_allclose(np_lag0, pd_lag0)
        assert_allclose(np_lag0.ravel(), sp_lag0.ix[0])
        assert_allclose(pd_lag0.values.ravel(), sp_lag0.ix[0])


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
