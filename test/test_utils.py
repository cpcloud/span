import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from ..utils import (cast, ndtuples, dirsize, ndlinspace, nans, remove_legend,
                     name2num, group_indices, flatten, bin_data, summary,
                     nextpow2, fractional, zeropad, pad_larger2, pad_larger,
                     iscomplex)


def test_nextpow2():
    """
    """
    int_max = 10000
    n = np.random.randint(int_max)
    np2 = nextpow2(n)
    assert 2 ** np2 > n, '2 ** np2 == {} <= n == {}'.format(2 ** np2, n)
    assert np2 == np.log2(2 ** np2), '{} != np.log2({})'.format(np2, 2 ** np2)


def test_fractional():
    """
    """
    n = 1
    m = 1000
    x = np.random.randn(n)
    xi = np.random.randint(m)
    assert fractional(x)
    assert not fractional(xi)


class TestZeroPad(unittest.TestCase):
    def setUp(self):
        """
        """
        n = 50
        dims = 3
        self.x = np.random.randn(*np.random.randint(np.random.randint(n),
                                                    size=dims).tolist())
        self.s = np.random.randint(n)

    def test_s0(self):
        """Test the case of padding for a 1D array.
        """
        assert_array_equal(self.x, zeropad(self.x, 0))

    def test_sn(self):
        """
        """
        shp = np.asanyarray(self.x.shape)
        extra_zeros = 2 * self.s + np.zeros(self.x.ndim)
        desired_size = np.prod(shp + extra_zeros)
        padded = zeropad(self.x, self.s)
        actual_size = padded.size
        self.assertEqual(desired_size, actual_size)


class TestPadLarger():
    def setUp(self):
        """
        """

    def test_pad_larger2(self):
        """
        """

    def test_pad_larger(self):
        """
        """


def test_iscomplex():
    """
    """
    n = 1000
    x = np.random.randn(n, n) + 1j
    assert iscomplex(x), 'x is not complex and has type {}'.format(x.dtype)
    assert 


def test_get_fft_funcs():
    """
    """
