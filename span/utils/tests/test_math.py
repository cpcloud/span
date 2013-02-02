from unittest import TestCase
import itertools as itools

import numpy as np
from numpy.random import randn, randint, rand
from numpy.testing import assert_allclose, assert_array_equal

from pandas import Series, DataFrame, Panel

from span.utils import ndtuples
from span.utils.math import (trimmean, sem, detrend_none, detrend_mean,
                             detrend_linear, cartesian, nextpow2, fractional,
                             samples_per_ms, compose, composemap, compose2)

from span.utils.tests.test_utils import rand_int_tuple


class TestTrimmean(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.alphas = np.r_[:100:20].tolist() + [99]
        cls.includes = tuple(itools.product((True, False), (True, False)))
        cls.axes = None, 0

    def test_number(self):
        x = float(randn())
        axes = self.axes
        arg_sets = itools.product(self.alphas, self.includes, axes)
        for arg_set in arg_sets:
            alpha, include, axis = arg_set
            m = trimmean(x, alpha, include, axis)
            self.assertIsInstance(m, float)
            self.assertEqual(x, m)

    def test_0d_array(self):
        x = np.asanyarray(randn())
        axes = self.axes
        arg_sets = itools.product(self.alphas, self.includes, axes)

        for alpha, include, axis in arg_sets:
            m = trimmean(x, alpha, include, axis)
            self.assertIsInstance(m, float)
            self.assertEqual(x, m)

    def test_1d_array(self):
        x = randn(5)
        axes = self.axes
        arg_sets = itools.product(self.alphas, self.includes, axes)
        for arg_set in arg_sets:
            alpha, include, axis = arg_set
            m = trimmean(x, alpha, include, axis)
            self.assertIsInstance(m, float)

    def test_2d_array(self):
        x = randn(5, 4)
        axes = self.axes + (1,)
        arg_sets = itools.product(self.alphas, self.includes, axes)
        for arg_set in arg_sets:
            alpha, include, axis = arg_set
            m = trimmean(x, alpha, include, axis)
            if axis is not None:
                self.assertEqual(m.ndim, x.ndim - 1)
            else:
                self.assertIsInstance(m, float)

    def test_3d_array(self):
        x = randn(4, 3, 2)
        axes = self.axes + (1, 2)
        arg_sets = itools.product(self.alphas, self.includes, axes)
        for arg_set in arg_sets:
            alpha, include, axis = arg_set
            if axis is not None:
                self.assertRaises(Exception, trimmean, x, alpha, include, axis)
            else:
                m = trimmean(x, alpha, include, axis)
                self.assertIsInstance(m, float)

    def test_series(self):
        x = Series(randn(2))
        axes = self.axes + (1,)
        arg_sets = itools.product(self.alphas, self.includes, axes)
        for arg_set in arg_sets:
            alpha, include, axis = arg_set

            if axis == 1:
                self.assertRaises(AssertionError, trimmean, x, alpha, include,
                                  axis)

    def test_dataframe(self):
        x = DataFrame(randn(3, 2))
        axes = self.axes + (1,)
        arg_sets = itools.product(self.alphas, self.includes, axes)
        for arg_set in arg_sets:
            alpha, include, axis = arg_set
            m = trimmean(x, alpha, include, axis)
            if axis is not None:
                self.assertEqual(m.ndim, x.ndim - 1)
            else:
                self.assertIsInstance(m, Series)


class TestSem(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.axes, cls.ddof = (0, 1), (0, 1)
        cls.args = tuple(itools.product(cls.axes, cls.ddof))

    @classmethod
    def tearDownClass(cls):
        del cls.args, cls.ddof, cls.axes

    def test_0d(self):
        x = randn()
        for axis, ddof in self.args:
            s = sem(x, axis, ddof)
            self.assertEqual(s, 0.0)

    def test_1d(self):
        x = randn(2)

        for axis, ddof in self.args:
            if axis == 1:
                self.assertRaises(IndexError, sem, x, axis, ddof)
            else:
                s = sem(x, axis, ddof)
                dtype = s.dtype
                self.assert_(np.issubdtype(dtype, np.floating))

    def test_2d(self):
        x = randn(3, 2)

        for axis, ddof in self.args:
            s = sem(x, axis, ddof)

            try:
                dtype = s.dtype
            except AttributeError:
                dtype = type(s)

            sshape = list(s.shape)
            not_xshape = list(filter(lambda a: a != x.shape[axis], x.shape))
            self.assertListEqual(not_xshape, sshape)
            self.assert_(np.issubdtype(dtype, np.floating))
            self.assertIsInstance(s, np.ndarray)

    def test_3d(self):
        x = randn(4, 3, 2)
        axes = self.axes + (2,)
        args = itools.product(axes, self.ddof)

        for axis, ddof in args:
            s = sem(x, axis, ddof)

            try:
                dtype = s.dtype
            except AttributeError:
                dtype = type(s)

            self.assert_(np.issubdtype(dtype, np.floating))
            sshape = list(s.shape)
            not_xshape = list(filter(lambda a: a != x.shape[axis], x.shape))
            self.assertListEqual(not_xshape, sshape)
            self.assertIsInstance(s, np.ndarray)

    def test_series(self):
        x = Series(randn(13))

        for axis, ddof in self.args:

            if axis == 1:
                self.assertRaises(IndexError, sem, x, axis, ddof)
            else:
                s = sem(x, axis, ddof)

                try:
                    dtype = s.dtype
                    self.assert_(np.issubdtype(dtype, np.floating))
                except AttributeError:
                    dtype = type(s)
                    self.assertIsInstance(s, (np.floating, float))

    def test_dataframe(self):
        x = DataFrame(randn(2, 3))

        for axis, ddof in self.args:
            s = sem(x, axis, ddof)
            self.assert_(np.issubdtype(s.dtype, np.floating))
            self.assertIsInstance(s, (Series, np.ndarray))

    def test_panel(self):
        x = Panel(randn(4, 3, 2))
        axes = self.axes + (2,)
        args = itools.product(axes, self.ddof)

        for axis, ddof in args:
            s = sem(x, axis, ddof)
            self.assertIsInstance(s, DataFrame)


def test_ndtuples():
    t = rand_int_tuple()
    k = ndtuples(*t)
    uk = np.unique(k.ravel())
    uk.sort()
    assert_array_equal(uk, np.arange(max(t)))


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

        axes = 0, 1
        for axis in axes:
            dtx = detrend_mean(x, axis)
            zs = np.zeros(dtx.shape[1 - axis])
            assert_allclose(dtx.mean(axis).ravel(), zs, atol=self.eps)

    def test_detrend_mean_dataframe(self):
        x = DataFrame(np.random.randn(3, 4))

        axes = 0, 1
        for axis in axes:
            dtx = detrend_mean(x, axis)
            zs = np.zeros(dtx.shape[1 - axis])
            assert_allclose(dtx.mean(axis), zs, atol=self.eps)

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
        ord_mag = int(np.floor(np.log10(n)))
        rtol = 10.0 ** ((1 - ord_mag) + (ord_mag - 1))
        assert_allclose(m, 0.0, atol=self.eps * 5)
        assert_allclose(s, 0.0, atol=self.eps * 5)


class TestCartesian(TestCase):
    def test_cartesian(self):
        ncols = randint(2, 3)
        sizes = [randint(2, 4) for _ in xrange(ncols)]
        prod_arrays = map(randn, sizes)
        c = cartesian(prod_arrays)
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


class TestFractional(TestCase):
    def test_fractional(self):
        m, n = 100, 1
        x = randn(n)
        xi = randint(m)
        self.assert_(fractional(x))
        self.assert_(fractional(rand()))
        self.assertFalse(fractional(xi))
        self.assertFalse(fractional(randint(1, np.iinfo(int).max)))


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
