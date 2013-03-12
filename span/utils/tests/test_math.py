from unittest import TestCase
import itertools as itools

import numpy as np
from numpy.random import randn, randint
from numpy.testing import assert_allclose, assert_array_equal

from pandas import Series, DataFrame, Panel, Panel4D

from span.utils import ndtuples
from span.utils.math import (detrend_none, detrend_mean,
                             detrend_linear, cartesian, nextpow2,
                             samples_per_ms, compose, composemap, compose2)

from span.utils.tests.test_utils import rand_int_tuple
from six.moves import map


class TestNdtuples(TestCase):
    def test_ndtuples_0(self):
        zdims = 0, -1, False

        for zdim_null in zdims:
            self.assertEqual(ndtuples(zdim_null).size, np.array([]).size)

        zdims = {}, [], (), np.array([])

        for zdim_type_error in zdims:
            self.assertRaises(TypeError, ndtuples, zdim_type_error)

    def test_ndtuples_1(self):
        n = randint(1, 3)
        x = ndtuples(n)
        assert_array_equal(x.ravel(), np.arange(n))

    def test_ndtuples_2(self):
        m, n = randint(2, 5), randint(2, 4)
        x = ndtuples(m, n)
        self.assertTupleEqual((m * n, 2), x.shape)

    def test_ndtuples_3(self):
        m, n, l = randint(2, 3), randint(2, 4), randint(3, 4)
        x = ndtuples(m, n, l)
        self.assertTupleEqual((m * n * l, 3), x.shape)


def test_ndtuples():
    t = rand_int_tuple()
    k = ndtuples(*t)
    uk = np.unique(k.ravel())
    uk.sort()
    assert_array_equal(uk.ravel(), np.arange(max(t)))


class TestDetrend(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.eps = np.finfo(float).eps

    @classmethod
    def tearDownClass(cls):
        del cls.eps

    def test_detrend_none(self):
        x = np.random.randn(2, 3)
        dtx = detrend_none(x)
        assert_array_equal(x, dtx)

    def test_detrend_mean(self):
        x = np.random.randn(3, 2)

        axes = xrange(x.ndim)
        for axis in axes:
            dtx = detrend_mean(x, axis)
            zs = np.zeros(dtx.shape[1 - axis])
            assert_allclose(dtx.mean(axis).ravel(), zs, atol=self.eps)

    def test_detrend_mean_series(self):
        x = Series(np.random.randn(3))

        axes = xrange(x.ndim)
        for axis in axes:
            dtx = detrend_mean(x, axis)
            zs = 0
            assert_allclose(dtx.mean(axis).ravel(), zs, atol=self.eps)

    def test_detrend_mean_dataframe(self):
        x = DataFrame(np.random.randn(3, 4))

        axes = xrange(x.ndim)
        for axis in axes:
            dtx = detrend_mean(x, axis)
            zs = np.zeros(dtx.shape[1 - axis])
            assert_allclose(dtx.mean(axis), zs, atol=self.eps)

    def test_detrend_mean_panel(self):
        x = Panel(np.random.randn(3, 4, 3))

        axes = xrange(x.ndim)
        for axis in axes:
            self.assertRaises(NotImplementedError, detrend_mean, x, axis)
            # dtx = detrend_mean(x, axis)
            # zs = np.zeros(dtx.shape[1 - axis])
            # assert_allclose(dtx.mean(axis), zs, atol=self.eps)

    def test_detrend_mean_panel4d(self):
        x = Panel4D(np.random.randn(3, 4, 3, 2))

        axes = xrange(x.ndim)
        for axis in axes:
            self.assertRaises(NotImplementedError, detrend_mean, x, axis)
            # dtx = detrend_mean(x, axis)
            # zs = np.zeros(dtx.shape[1 - axis])
            # assert_allclose(dtx.mean(axis), zs, atol=self.eps)

    def test_detrend_mean_scalar(self):
        x = np.random.randn()
        y = 1.0
        z = 1j + np.random.randn()
        i = 1

        for val in {x, y, z, i}:
            dt = detrend_mean(val)
            isnp = hasattr(dt, 'dtype')
            expec = val.dtype.type(0) if isnp else type(val)(0)

            self.assertEqual(expec, dt)

    def test_detrend_linear(self):
        n = 3
        x = np.arange(n)
        dtx = detrend_linear(x)
        m = dtx.mean()
        s = dtx.std(ddof=1)
        assert_allclose(m, 0.0, atol=self.eps * 5)
        assert_allclose(s, 0.0, atol=self.eps * 5)

    def test_detrend_linear_series(self):
        n = 3
        x = Series(np.arange(n))
        dtx = detrend_linear(x)
        m = dtx.mean()
        s = dtx.std()
        #ord_mag = int(np.floor(np.log10(n)))
        assert_allclose(m, 0.0, atol=self.eps * 5)
        assert_allclose(s, 0.0, atol=self.eps * 5)


class TestCartesian(TestCase):
    def test_cartesian(self):
        ncols = randint(2, 3)
        sizes = [randint(2, 4) for _ in xrange(ncols)]
        prod_arrays = list(map(randn, sizes))
        c = cartesian(*prod_arrays)
        self.assertEqual(c.size, np.prod(sizes) * ncols)


class TestNextPow2(TestCase):
    def test_nextpow2(self):
        int_max = 101
        n = randint(1, int_max)
        np2 = nextpow2(n)
        tp2 = 2 ** np2
        self.assertGreaterEqual(tp2, n)
        assert_allclose(np2, np.log2(tp2))

        self.assertEqual(nextpow2(0), np.iinfo(nextpow2(0).dtype).min)


class TestSamplesPerMs(TestCase):
    def test_samples_per_ms(self):
        args = np.arange(10)
        fs = 24414.0625
        r = list(map(type, (samples_per_ms(fs, arg) for arg in args)))
        self.assertListEqual(r, list(itools.repeat(int, len(r))))


class TestCompose2(TestCase):
    def test_compose2(self):
        # fail if not both callables
        f, g = 1, 2
        self.assertRaises(AssertionError, compose2, f, g)

        f, g = 1, np.exp
        self.assertRaises(AssertionError, compose2, f, g)

        f, g = np.log, 2
        self.assertRaises(AssertionError, compose2, f, g)

        # don't fail if both callables
        f, g = np.log, np.exp
        h = compose2(f, g)
        x = randn(10, 20)
        assert_allclose(x, h(x))


class TestCompose(TestCase):
    def test_compose(self):
        # fail if not both callables
        f, g, h, q = 1, 2, np.log, np.exp
        self.assertRaises(AssertionError, compose, f, g, h, q)

        f, g, h, q = 1, np.exp, 1.0, 'sd'
        self.assertRaises(AssertionError, compose, f, g, h, q)

        f, g, h, q = np.log, 2, object(), []
        self.assertRaises(AssertionError, compose, f, g, h, q)

        # don't fail if all callables
        f, g, h, q = np.log, np.exp, np.log, np.exp
        h = compose(f, g, h, q)
        x = randn(10, 20)
        assert_allclose(x, h(x))


def test_composemap():
    f, g = np.log, np.exp
    x, y = randn(11), randn(10)
    xnew, ynew = composemap(f, g)((x, y))
    assert_allclose(x, xnew)
    assert_allclose(y, ynew)
