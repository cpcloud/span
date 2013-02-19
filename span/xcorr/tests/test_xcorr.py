import itertools
import unittest

import numpy as np
from numpy.random import randn, randint
from numpy.testing import assert_allclose

from pandas import DataFrame, Series
from pandas.util.testing import assert_frame_equal

from six.moves import map

from span.xcorr.xcorr import (xcorr, _mult_mat_xcorr,
                              _mult_mat_xcorr_cython_parallel,
                              _mult_mat_xcorr_python)
from span.utils import (nextpow2, get_fft_funcs, detrend_mean, detrend_none,
                        detrend_linear)
from span.testing import assert_array_equal, knownfailure, assert_raises


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


class TestXCorr(object):
    def setUp(self):
        self.m, self.n = 3, 2
        self.x, self.y = randn(self.m), randn(self.n)
        self.xsame, self.ysame = self.x, randn(self.m)
        self.matrix = randn(self.m, self.n)

    def tearDown(self):
        del self.matrix, self.ysame, self.xsame, self.y, self.x, self.m, self.n

    def test_autocorr(self):
        x = self.x
        npc = np.correlate(x, x, mode='full')
        spc = xcorr(x)
        assert_allclose(npc, spc)

        spc = xcorr(Series(x))
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

    def test_xcorr(self):
        scale_types = None, 'biased', 'unbiased', 'normalize', 'none'
        maxlags = 1, 2, 100000
        detrends = None, detrend_none, detrend_mean, detrend_linear
        args = itertools.product(maxlags, scale_types, detrends)

        for maxlag, scale_type, detrend in args:
            yield self.xcorr_builder, maxlag, scale_type, detrend

    def xcorr_builder(self, maxlag, scale_type, detrend):
        inputs = ((self.x, self.y),
                  (self.matrix,),
                  (self.xsame, self.ysame),
                  list(map(Series, (self.x, self.y))),
                  (DataFrame(self.matrix),),
                  list(map(Series, (self.xsame, self.ysame))),
                  [self.x],
                  [Series(self.x)])

        for inp in inputs:
            lens = reduce(lambda x, y: x + y, map(lambda x: x.shape, inp))
            kwargs = dict(maxlags=maxlag, detrend=detrend,
                          scale_type=scale_type)

            if maxlag > 2 * max(lens) - 1:
                assert_raises(AssertionError, xcorr, *inp, **kwargs)
            else:
                sp_xc = xcorr(*inp, **kwargs)


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
        _mult_mat_xcorr_python(X, Xc, self.ground_truth, n)

    def tearDown(self):
        del self.ground_truth, self.X, self.Xc, self.c, self.n

    def test_mult_mat_xcorrs(self):
        funcs = {_mult_mat_xcorr_cython_parallel}

        for func in funcs:
            func(self.X, self.Xc, self.c, self.n)
            assert_allclose(self.c, self.ground_truth)

    def test_mult_mat_xcorr_high_level(self):
        assert_allclose(_mult_mat_xcorr(self.X, self.Xc), self.ground_truth)
