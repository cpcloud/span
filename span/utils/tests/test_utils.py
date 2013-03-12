import os
import random
import string
import types
import unittest
import copy
import time

import numpy as np
import six
from numpy.random import randint, randn
from pandas import Series, DataFrame
from six.moves import zip, map

from span.testing import assert_allclose, assert_array_equal
from span.utils import (nextpow2, name2num, isvector,
                        iscomplex, get_fft_funcs,
                        assert_nonzero_existing_file, LOCAL_TZ,
                        ispower2)
from span.utils.utils import _diag_inds_n, _get_local_tz


def rand_int_tuple(high=5, n=10):
    return tuple(randint(1, high, size=n))


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


class TestAssertNonzeroExistingFile(unittest.TestCase):
    def setUp(self):
        self.name = 'blah.npy'

    def tearDown(self):
        del self.name

    def test_assert_non_existing_file(self):
        name = self.name

        with open(name, 'wb') as tf:
            randn(2).tofile(tf)

        assert_nonzero_existing_file(name)

        os.remove(name)

        self.assertRaises(AssertionError, assert_nonzero_existing_file, name)


class TestIsPower2(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_ispower2(self):
        assert ispower2(2) == 1
        self.assertEqual(ispower2(2), 1)
        self.assertEqual(ispower2(1), 0)
        self.assertEqual(ispower2(3), 0)
        self.assertEqual(ispower2(16), 4)


class TestDiagIndsN(unittest.TestCase):
    def test_diag_inds_n(self):
        n = randint(2, 10)
        x = np.arange(n * n).reshape(n, n)
        inds = _diag_inds_n(n)
        assert_array_equal(inds, np.diag(x))


class TestGetLocalTz(unittest.TestCase):
    def test_get_local_tz(self):
        self.assertIsInstance(LOCAL_TZ, (types.NoneType,) + six.string_types)
        self.assertIsInstance(_get_local_tz(),
                              (types.NoneType,) + six.string_types)

    def test_get_local_tz_fail(self):
        old_tzs = copy.deepcopy(time.tzname)
        new_tzs = 'asdfasdf',
        time.tzname = new_tzs
        self.assertIsNone(_get_local_tz())
        time.tzname = old_tzs
        self.assertIsInstance(_get_local_tz(),
                              (types.NoneType,) + six.string_types)
