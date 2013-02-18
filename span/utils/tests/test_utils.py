import unittest
import string
import random
import itertools as itools
import os

import numpy as np
from numpy.random import randint, rand, randn

from pandas import Series, DataFrame, Panel, MultiIndex, Index
from six.moves import zip, map

from span.utils import (nextpow2, nans, nans_like, name2num, isvector,
                        iscomplex, hascomplex, get_fft_funcs, cast,
                        ndtuples, nonzero_existing_file,
                        assert_nonzero_existing_file, mi2df)

from span.testing import assert_allclose, assert_array_equal, rands


def rand_array_delegate(func, n, ndims):
    return func(*randint(n, size=ndims).tolist())


def randn_array(n=50, ndims=3):
    return rand_array_delegate(randn, n, ndims)


def rand_int_tuple(high=5, n=10):
    return tuple(randint(1, high, size=n))


def test_nans():
    shape = 6, 3
    x = nans(shape)
    assert np.isnan(x).all(), 'not all values are nans'
    assert x.dtype == np.float64


class TestNansLike(unittest.TestCase):
    def test_series(self):
        x = Series(randn(7))
        nas = nans_like(x)
        self.assert_(np.isnan(nas).all())

    def test_dataframe(self):
        x = DataFrame(randn(10, 3))
        nas = nans_like(x)
        self.assert_(np.isnan(nas.values).all())

    def test_panel(self):
        x = Panel(randn(5, 4, 3))
        nas = nans_like(x)

        # panel has no all member for some reason
        self.assert_(np.isnan(nas.values).all())

    def test_other(self):
        arrays = randn(2), randn(10, 4), randn(10, 8, 3)
        for array in arrays:
            nas = nans_like(array)
            self.assert_(np.isnan(nas).all())


class TestIsComplex(unittest.TestCase):
    def setUp(self):
        n = 4
        self.x = randn(n, randint(1, n)) * 1j

    def tearDown(self):
        del self.x

    def test_iscomplex_dataframe(self):
        x = DataFrame(self.x)
        self.assert_(iscomplex(x),
                     'x is not complex and has dtypes {0}'.format(x.dtypes))

    def test_is_not_complex_dataframe(self):
        x = DataFrame(self.x.real)
        self.assert_(not iscomplex(x))

    def test_iscomplex(self):
        x = self.x
        self.assert_(iscomplex(x),
                     'x is not complex and has type {0}'.format(x.dtype))

    def test_is_not_complex(self):
        x = self.x.real
        self.assert_(not iscomplex(x))


class TestHasComplex(unittest.TestCase):
    def setUp(self):
        self.n = 4
        self.x = randn(self.n, randint(1, self.n)) * 1j

    def tearDown(self):
        del self.x, self.n

    def test_hascomplex_dataframe(self):
        x = DataFrame(self.x)
        self.assert_(hascomplex(x))

    def test_not_hascomplex_dataframe(self):
        n = self.n
        x = DataFrame(randn(n, randint(1, n)))
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
    x = Series(dict(zip(letters, map(ord, letters))))
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
        arg_sets = itools.product(dtypes, copies)
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
    def setUp(self):
        self.matrix = np.random.randn(2, 3)
        self.vector = np.random.randn(3)
        self.scalar = np.random.randn()

    def tearDown(self):
        del self.scalar, self.vector, self.matrix

    def test_isvector(self):
        self.assertFalse(isvector(self.matrix))
        self.assertFalse(isvector(self.scalar))

        self.assert_(isvector(self.vector))
        self.assert_(isvector(self.vector[np.newaxis]))
        self.assert_(isvector(self.vector[:, np.newaxis]))
        self.assert_(isvector(list(self.vector)))
        self.assert_(isvector(tuple(self.vector)))

        dims = [2] + [1] * 31
        x = np.random.rand(*dims)
        self.assert_(isvector(x))


class TestNdtuples(unittest.TestCase):
    def test_ndtuples_0(self):
        zdims = 0, False, [], (), {}, np.array([])

        for zdim in zdims:
            self.assertRaises(AssertionError, ndtuples, zdim)

    def test_ndtuples_1(self):
        n = randint(1, 3)
        x = ndtuples(n)
        assert_array_equal(x, np.arange(n))

    def test_ndtuples_2(self):
        m, n = randint(2, 5), randint(2, 4)
        x = ndtuples(m, n)
        self.assertTupleEqual((m * n, 2), x.shape)

    def test_ndtuples_3(self):
        m, n, l = randint(2, 3), randint(2, 4), randint(3, 4)
        x = ndtuples(m, n, l)
        self.assertTupleEqual((m * n * l, 3), x.shape)


class TestNonzeroExistingFile(unittest.TestCase):
    def setUp(self):
        self.name = 'blah.npy'

    def tearDown(self):
        del self.name

    def test_nonzero_existing_file(self):
        name = self.name

        with open(name, 'wb') as tf:
            randn(2).tofile(tf)

        self.assert_(nonzero_existing_file(name))

        os.remove(name)

        self.assertFalse(nonzero_existing_file(name))

    def test_assert_non_existing_file(self):
        name = self.name

        with open(name, 'wb') as tf:
            randn(2).tofile(tf)

        assert_nonzero_existing_file(name)

        os.remove(name)

        self.assertRaises(AssertionError, assert_nonzero_existing_file, name)


class TestMi2Df(unittest.TestCase):
    def test_mi2df(self):

        class _BlobJect(object):
            pass

        dtypes = object, int, long, str, float

        n = 2
        for dtype in dtypes:
            s = np.array(list(rands(n)))
            i = randint(10, size=n)
            f = rand(n)
            o = rand(n).astype(object)
            bo = np.array(list(itools.repeat(_BlobJect(), n)))
            x = s, i, f, o, bo
            names = [rands(len(x)) for _ in xrange(len(x))]
            mi = MultiIndex.from_arrays(x, names=names)
            df = mi2df(mi)
            self.assertIsInstance(df, DataFrame)
            # self.assertListEqual(names, df.columns.tolist())
            # self.assertRaises(AssertionError, mi2df, Index([1, 2, 3]))
