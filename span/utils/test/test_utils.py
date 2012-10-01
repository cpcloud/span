import unittest
import string
import random
import itertools

import numpy as np
from numpy.random import randint, rand, randn
from numpy.testing import assert_allclose, assert_array_equal
from numpy.testing.decorators import slow
from nose.tools import nottest

import pandas as pd

from pylab import gca

from span.utils import (
    cast, dirsize, fractional, get_fft_funcs, group_indices,
    iscomplex, isvector, name2num, nans, ndlinspace, ndtuples, nextpow2,
    pad_larger, pad_larger2, remove_legend, cartesian, trimmean, num2name,
    detrend_none, detrend_mean, detrend_linear, hascomplex, compose, composemap)

from span.utils.utils import compose2


def rand_array_delegate(func, n, ndims):
    return func(*randint(n, size=ndims).tolist())


def randn_array(n=100, ndims=3): return rand_array_delegate(randn, n, ndims)


def rand_int_tuple(m=5, n=10): return randint(1, m, size=n)


def test_nextpow2():
    int_max = 100
    n = randint(int_max)
    np2 = nextpow2(n)
    tp2 = 2 ** np2
    assert tp2 > n, '{} <= {}'.format(tp2, n)
    assert_allclose(np2, np.log2(tp2))


def test_fractional():
    m, n = 100, 1
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


class TestNdlinspace(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ranges = randn(randint(5, 10), randint(10, 20)).tolist()

    def test_1elem(self):
        sizes = randint(5, 10)
        x = ndlinspace(self.ranges, sizes)

    def test_2elem(self):
        sizes = randint(5, 10, size=2)
        x = ndlinspace(self.ranges, *sizes.tolist())

    def test_3elem(self):
        sizes = randint(5, 10, size=3)
        x = ndlinspace(self.ranges, *sizes.tolist())
    

def test_nans():
    m, n = 100, 10
    x = nans((m, n))
    assert np.isnan(x).all(), 'not all values are nans'
    assert x.dtype == np.float64


def test_remove_legend():
    ax = gca()
    remove_legend(ax)
    assert ax.legend_ is None


@slow
def test_num2name():
    expected = 'Spik'
    name = num2name(name2num(expected))
    assert name == expected, '{} != {}'.format(name, expected)


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
        x, y, lsize = pad_larger(self.xbig, self.ysmall)
        assert lsize == max(x.shape + y.shape)

        x, y, lsize = pad_larger2(self.xsmall, self.ybig)
        assert lsize == max(x.shape + y.shape)



class TestIsComplex(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        n = 20
        cls.x = randn(n, randint(1, n)) + rand() * 1j
        
    def test_iscomplex_dataframe(self):
        x = pd.DataFrame(self.x)
        self.assert_(iscomplex(x), 'x is not complex and has dtypes {}'.format(x.dtypes))

    def test_is_not_complex_dataframe(self):
        pass

    def test_iscomplex(self):
        x = self.x
        self.assert_(iscomplex(x), 'x is not complex and has type {}'.format(x.dtype))

    def test_is_not_complex(self):
        pass


class TestHasComplex(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.n = 20
        n = cls.n
        cls.x = randn(n, randint(1, n)) + rand() * 1j
    
    def test_hascomplex_dataframe(self):
        x = pd.DataFrame(self.x)
        self.assert_(hascomplex(x))

    def test_not_hascomplex_dataframe(self):
        n = self.n
        x = pd.DataFrame(randn(n, randint(1, n)))
        self.assertFalse(hascomplex(x))

    def test_hascomplex(self):
        n = self.n
        x = randn(n, randint(1, n)) * 1j
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


def test_electrode_distance():
    assert False


def test_distance_map():
    assert False


class TestTrimmean(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.alphas = np.r_[:100]
        cls.includes = tuple(itertools.product((True, False), (True, False)))
        cls.axes = None, 0

    def test_number(self):
        x = float(randn())
        axes = self.axes
        arg_sets = itertools.product(self.alphas, self.includes, axes)
        for arg_set in arg_sets:
            alpha, include, axis = arg_set
            m = trimmean(x, alpha, include, axis)
            self.assertIsInstance(m, float)
            self.assertEqual(x, m)
            
    def test_0d_array(self):
        x = np.asanyarray(randn())
        axes = self.axes
        arg_sets = itertools.product(self.alphas, self.includes, axes)
        for arg_set in arg_sets:
            alpha, include, axis = arg_set
            m = trimmean(x, alpha, include, axis)
            self.assertIsInstance(m, float)
            self.assertEqual(x, m)
        
    def test_1d_array(self):
        x = randn(58)
        axes = self.axes
        arg_sets = itertools.product(self.alphas, self.includes, axes)
        for arg_set in arg_sets:
            alpha, include, axis = arg_set
            m = trimmean(x, alpha, include, axis)
            self.assertIsInstance(m, float)
            if isinstance(m, np.ndarray):
                self.assertEqual(m.dtype, float)

    def test_2d_array(self):
        x = randn(50, 13)
        axes = self.axes + (1,)
        arg_sets = itertools.product(self.alphas, self.includes, axes)
        for arg_set in arg_sets:
            alpha, include, axis = arg_set
            m = trimmean(x, alpha, include, axis)
            if axis is not None:
                self.assertEqual(m.ndim, x.ndim - 1)
            else:
                self.assertIsInstance(m, float)

    def test_3d_array(self):
        x = randn(10, 6, 4)
        axes = self.axes + (1, 2)
        arg_sets = itertools.product(self.alphas, self.includes, axes)
        for arg_set in arg_sets:
            alpha, include, axis = arg_set
            if axis is not None:
                self.assertRaises(Exception, trimmean, x, alpha, include, axis)
            else:
                m = trimmean(x, alpha, include, axis)
                self.assertIsInstance(m, float)

    def test_series(self):
        x = pd.Series(randn(18))
        axes = self.axes
        arg_sets = itertools.product(self.alphas, self.includes, axes)
        for arg_set in arg_sets:
            alpha, include, axis = arg_set
            print alpha, include, axis
            if axis == 1:
                self.assertRaises(TypeError, trimmean, x, alpha, include, axis)

    def test_dataframe(self):
        x = pd.DataFrame(randn(51, 17))
        axes = self.axes + (1,)
        arg_sets = itertools.product(self.alphas, self.includes, axes)
        for arg_set in arg_sets:
            alpha, include, axis = arg_set
            m = trimmean(x, alpha, include, axis)
            if axis is not None:
                self.assertEqual(m.ndim, x.ndim - 1)
            else:
                self.assertIsInstance(m, float)


class TestDetrend(unittest.TestCase):
    def test_detrend_none(self):
        x = np.random.randn(10, 11)
        dtx = detrend_none(x)
        assert_array_equal(x, dtx)

    def test_detrend_mean(self):
        x = np.random.randn(10, 9)
        dtx = detrend_mean(x)
        expect = x - x.mean()
        assert expect.dtype == dtx.dtype
        assert_array_equal(dtx, expect)
        assert_allclose(dtx.mean(), 0.0, atol=np.finfo(dtx.dtype).eps)

    def test_detrend_mean_dataframe(self):
        x = pd.DataFrame(np.random.randn(10, 13))
        dtx = detrend_mean(x)
        m = dtx.mean()
        eps = np.finfo(float).eps
        assert_allclose(m.values.squeeze(), np.zeros(m.shape),
                        atol=eps)
        print m.values.squeeze().size

    def test_detrend_linear(self):
        n = 100
        x = np.random.randn(n)
        dtx = detrend_linear(x)
        eps = np.finfo(dtx.dtype).eps
        ord_mag = int(np.floor(np.log10(n)))
        rtol = 10.0 ** (1 - ord_mag) + (ord_mag - 1)
        assert_allclose(dtx.mean(), 0.0, rtol=rtol, atol=eps)
        assert_allclose(dtx.std(), 1.0, rtol=rtol, atol=eps)

    def test_detrend_linear_series(self):
        n = 100
        x = pd.Series(np.random.randn(n))
        dtx = detrend_linear(x)
        m = dtx.mean()
        s = dtx.std()
        ord_mag = int(np.floor(np.log10(n)))
        rtol = 10.0 ** (1 - ord_mag) + (ord_mag - 1)
        eps = np.finfo(float).eps
        assert_allclose(m, 0.0, rtol=rtol, atol=eps)
        assert_allclose(s, 1.0, rtol=rtol, atol=eps)


class TestCompose2(unittest.TestCase):
    def test_compose2(self):
        # fail if not both callables
        f, g = 1, 2
        self.assertRaises(TypeError, compose2, f, g)

        f, g = 1, np.exp
        self.assertRaises(TypeError, compose2, f, g)

        f, g = np.log, 2
        self.assertRaises(TypeError, compose2, f, g)

        # don't fail if both callables
        f, g = np.log, np.exp
        h = compose2(f, g)
        x = randn(10, 20)
        assert_allclose(x, h(x))


class TestCompose(unittest.TestCase):
    def test_compose(self):
        # fail if not both callables
        f, g, h, q = 1, 2, np.log, np.exp
        self.assertRaises(TypeError, compose, f, g, h, q)

        f, g, h, q = 1, np.exp, 1.0, 'sd'
        self.assertRaises(TypeError, compose, f, g, h, q)

        f, g, h, q = np.log, 2, object(), []
        self.assertRaises(TypeError, compose, f, g, h, q)

        # don't fail if all callables
        f, g, h, q = np.log, np.exp, np.log, np.exp
        h = compose(f, g, h, q)
        x = randn(10, 20)
        assert_allclose(x, h(x))


class TestComposeMap(unittest.TestCase):
    def test_composemap(self):
        assert False
