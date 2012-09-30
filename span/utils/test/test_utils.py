import unittest
import string
import random
import itertools
import operator
import functools

import numpy as np
from numpy.random import randint, rand, randn
from numpy.testing import assert_allclose, assert_array_equal

import pandas as pd

from pylab import gca

from span.utils import (
    bin_data, cast, dirsize, fractional, get_fft_funcs, group_indices,
    iscomplex, isvector, name2num, nans, ndlinspace, ndtuples, nextpow2,
    pad_larger, pad_larger2, remove_legend, cartesian)


def rand_array_delegate(func, n, ndims):
    return func(*randint(n, size=ndims).tolist())


def randn_array(n=100, ndims=3):
    return rand_array_delegate(randn, n, ndims)


def rand_int_tuple(m=5, n=10): return randint(1, m, size=n)


def test_nextpow2():
    int_max = 1000
    n = randint(int_max)
    np2 = nextpow2(n)
    assert 2 ** np2 > n, '2 ** np2 == {} <= n == {}'.format(2 ** np2, n)
    assert_allclose(np2, np.log2(2 ** np2))


def test_fractional():
    n = 1
    m = 100
    x = randn(n)
    xi = randint(m)
    assert fractional(x)
    assert fractional(rand())
    assert not fractional(xi)
    assert not fractional(randint(1, np.iinfo(int).max))


def test_ndtuples():
    t = rand_int_tuple()
    k = ndtuples(*t)
    set_k = np.unique(k.ravel())
    set_k.sort()
    assert np.array_equal(set_k, np.arange(max(t)))


class TestCartesian(unittest.TestCase):
    def test_cartesian(self):
        ncols = randint(2, 6)
        sizes = [randint(5, 10) for _ in xrange(ncols)]
        prod_arrays = map(randn, sizes)
        c = cartesian(prod_arrays)
        self.assertEqual(c.size, np.prod(sizes) * ncols)


def test_dirsize():
    cd_ds = dirsize()


def test_ndlinspace():
    assert False


def test_nans():
    m, n = 1000, 10
    x = nans((m, n))
    assert np.isnan(x).all(), 'not all values are nans'
    assert x.dtype == np.float64


def test_remove_legend():
    ax = gca()
    remove_legend(ax)
    assert ax.legend_ is None


def test_num2name():
    assert False


def test_group_indices():
    assert False


class TestPadLarger(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        nbig = 20
        nsmall = 10
        ndims = 1
        cls.xbig = randn_array(n=nbig, ndims=ndims)
        cls.ysmall = randn_array(n=nsmall, ndims=ndims)
        cls.xsmall = randn_array(n=nsmall, ndims=ndims)
        cls.ybig = randn_array(n=nbig, ndims=ndims)
        cls.bigs = randint(nsmall, nbig)
        cls.smalls = randint(nsmall)

    def test_pad_larger2(self):
        x, y, lsize = pad_larger2(self.xbig, self.ysmall)
        assert lsize == max(x.shape + y.shape)

        x, y, lsize = pad_larger2(self.xsmall, self.ybig)
        assert lsize == max(x.shape + y.shape)

    def test_pad_larger(self):
        assert False
        # x, y, lsize = pad_larger(self.x, self.y)
        # assert lsize == max(x.shape + y.shape)


def test_iscomplex():
    n = 20
    x = randn(n, n) + 1j
    assert iscomplex(x), 'x is not complex and has type {}'.format(x.dtype)


def test_get_fft_funcs():
    m, n = [int(2 ** nextpow2(randint(1, 100))) for _ in xrange(2)]
    x = randn(m, n)
    xc = x + 1j

    ifft, fft = get_fft_funcs(xc, xc)
    assert fft is np.fft.fft
    assert ifft is np.fft.ifft
    ravelxc = xc.ravel()
    assert_allclose(ifft(fft(ravelxc)), ravelxc)

    ifft, fft = get_fft_funcs(x, x)
    assert fft is np.fft.rfft
    assert ifft is np.fft.irfft
    ravelx = x.ravel()
    assert_allclose(ifft(fft(ravelx)), ravelx)


def test_name2num():
    num_to_test = 10
    str_len = 4
    letters = string.ascii_letters
    x = pd.Series(dict(zip(letters, map(ord, letters))))
    base = 256 ** np.arange(str_len)
    for _ in xrange(num_to_test):
        name = random.sample(letters, str_len)
        num = name2num(name)
        mn, mx = base.dot(np.repeat(x.min(), 4)), base.dot(np.repeat(x.max(), 4))
        assert mn <= num <= mx


class TestCast(unittest.TestCase):
    def test_cast(self):
        dtypes = np.cast.keys()
        copies = True, False
        arg_sets = itertools.product(dtypes, copies)
        for arg_set in arg_sets:
            dtype, copy = arg_set
            a = rand(10)
            if dtype == np.void:
                self.assertRaises(TypeError, cast, a, dtype, copy)
            else:
                b = cast(a, dtype, copy)

            if a.dtype != dtype:
                self.assertNotEqual(a.dtype, b.dtype)


class TestIsVector(unittest.TestCase):
    def test_not_vector(self):
        x = np.random.rand(10)
        self.assertRaises(AttributeError, isvector, list(x))
        self.assertRaises(AttributeError, isvector, tuple(x))

        x = np.random.rand(10, 2)
        self.assertFalse(isvector(x))

    def test_isvector(self):
        x = np.random.rand(10)
        self.assert_(isvector(x))

        x = np.random.rand(10, 1)
        self.assert_(isvector(x))

        dims = (10,) + tuple(itertools.repeat(1, 31))
        x = np.random.rand(*dims)
        self.assert_(isvector(x))


def test_electrode_distance():
    assert False


def test_distance_map():
    assert False


class TestTrimmean(unittest.TestCase):
    def test_1d_array(self):
        assert False

    def test_2d_array(self):
        assert False

    def test_ndarray(self):
        assert False

    def test_series(self):
        assert False

    def test_dataframe(self):
        assert False
