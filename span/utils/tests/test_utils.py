import unittest
import string
import random
import itertools
import tempfile
import functools

import numpy as np
from numpy.random import randint, rand, randn
from numpy.testing import assert_allclose
from numpy.testing.decorators import slow
from nose.tools import nottest
from nose import SkipTest

import pandas as pd

from pylab import gca

from span.utils.utils import *
from span.utils.math import nextpow2

from span.testing import skip


def rand_array_delegate(func, n, ndims):
    return func(*randint(n, size=ndims).tolist())


def randn_array(n=50, ndims=3):
    return rand_array_delegate(randn, n, ndims)


def rand_int_tuple(m=5, n=10):
    return tuple(randint(1, m, size=n).tolist())


def test_nans():
    shape = 6, 3
    x = nans(shape)
    assert np.isnan(x).all(), 'not all values are nans'
    assert x.dtype == np.float64


class TestNansLike(unittest.TestCase):
    def test_series(self):
        x = pd.Series(randn(7))
        nas = nans_like(x)
        self.assert_(np.isnan(nas).all())

    def test_dataframe(self):
        x = pd.DataFrame(randn(10, 3))
        nas = nans_like(x)
        self.assert_(np.isnan(nas.values).all())

    def test_panel(self):
        x = pd.Panel(randn(5, 4, 3))
        nas = nans_like(x)

        # panel has no all member for some reason
        self.assert_(np.isnan(nas.values).all())

    def test_other(self):
        arrays = randn(10), randn(10, 4), randn(10, 8, 3)
        for array in arrays:
            nas = nans_like(array)
            self.assert_(np.isnan(nas).all())


def test_remove_legend():
    ax = gca()
    remove_legend(ax)
    assert ax.legend_ is None


@slow
def test_num2name():
    expected = 'Spik'
    name = num2name(name2num(expected))
    assert name == expected, '{} != {}'.format(name, expected)


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

    @classmethod
    def tearDownClass(cls):
        del cls.smalls, cls.bigs, cls.ybig, cls.xsmall, cls.ysmall, cls.xbig

    def test_pad_larger2(self):
        x, y, lsize = pad_larger2(self.xbig, self.ysmall)
        assert lsize == max(x.shape + y.shape)

        x, y, lsize = pad_larger2(self.xsmall, self.ybig)
        assert lsize == max(x.shape + y.shape)

    def test_pad_larger(self):
        x, y, lsize = pad_larger(self.xbig, self.ysmall)
        assert lsize == max(x.shape + y.shape)

        x, y, lsize = pad_larger2(self.xsmall, self.ybig)
        assert lsize == max(x.shape + y.shape)


class TestIsComplex(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        n = 20
        cls.x = randn(n, randint(1, n)) + rand() * 1j

    @classmethod
    def tearDownClass(cls):
        del cls.x

    def test_iscomplex_dataframe(self):
        x = pd.DataFrame(self.x)
        self.assert_(iscomplex(x),
                     'x is not complex and has dtypes {0}'.format(x.dtypes))

    def test_is_not_complex_dataframe(self):
        x = pd.DataFrame(self.x.real)
        self.assert_(not iscomplex(x))

    def test_iscomplex(self):
        x = self.x
        self.assert_(iscomplex(x),
                     'x is not complex and has type {0}'.format(x.dtype))

    def test_is_not_complex(self):
        x = self.x.real
        self.assert_(not iscomplex(x))


class TestHasComplex(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.n = 20
        n = cls.n
        cls.x = randn(n, randint(1, n)) + rand() * 1j

    @classmethod
    def tearDownClass(cls):
        del cls.x, cls.n

    def test_hascomplex_dataframe(self):
        x = pd.DataFrame(self.x)
        self.assert_(hascomplex(x))

    def test_not_hascomplex_dataframe(self):
        n = self.n
        x = pd.DataFrame(randn(n, randint(1, n)))
        self.assertFalse(hascomplex(x))

    def test_hascomplex(self):
        n = self.n
        x = randn(n, randint(1, n)) + 1j
        self.assert_(hascomplex(x))

    def test_not_hascomplex(self):
        n = self.n
        x = randn(n, randint(1, n))
        self.assertFalse(hascomplex(x))


def test_get_fft_funcs():
    m, n = [int(2 ** nextpow2(randint(5, 10))) for _ in xrange(2)]
    x = randn(m, n)
    xc = x + 1j

    ifft, fft = get_fft_funcs(xc, xc)
    assert fft is np.fft.fft, 'fft is not np.fft.fft'
    assert ifft is np.fft.ifft, 'ifft is not np.fft.ifft'
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
    mn = base.dot(np.repeat(x.min(), str_len))
    mx = base.dot(np.repeat(x.max(), str_len))

    for _ in xrange(num_to_test):
        name = random.sample(letters, str_len)
        num = name2num(name)
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


class TestNonzeroExistingFile(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.name = 'blah.npy'

    @classmethod
    def tearDownClass(cls):
        del cls.name

    def test_nonzero_existing_file(self):
        name = self.name

        with open(name, 'wb') as tf:
            randn(100).tofile(tf)

        self.assert_(nonzero_existing_file(name))

        os.remove(name)

        self.assertFalse(nonzero_existing_file(name))

    def test_assert_non_existing_file(self):
        name = self.name

        with open(name, 'wb') as tf:
            randn(100).tofile(tf)

        assert_nonzero_existing_file(name)

        os.remove(name)

        self.assertRaises(AssertionError, assert_nonzero_existing_file, name)


class TestTryConvertFirst(unittest.TestCase):
    def test_try_convert_first(self):
        assert False


class TestMi2Df(unittest.TestCase):
    def test_mi2df(self):
        assert False
