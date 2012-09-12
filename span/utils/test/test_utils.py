import unittest

import numpy as np
from numpy.random import randint, rand, randn
from numpy.testing import assert_allclose, assert_array_equal

from span.utils import (cast, ndtuples, dirsize, ndlinspace, nans, remove_legend,
                        name2num, group_indices, flatten, bin_data, summary,
                        nextpow2, fractional, zeropad, pad_larger2, pad_larger,
                        iscomplex, get_fft_funcs)


def rand_array_delegate(func, n, ndims):
    """
    """
    return func(*randint(n, size=ndims).tolist())


def rand_array(n=100, ndims=3):
    """
    """
    return rand_array_delegate(rand, n, ndims)


def randn_array(n=100, ndims=3):
    """
    """
    return rand_array_delegate(randn, n, ndims)


def rand_int_tuple(m=5, n=10):
    """
    """
    return randint(1, m, size=n)


def test_nextpow2():
    """
    """
    int_max = 1000
    n = randint(int_max)
    np2 = nextpow2(n)
    assert 2 ** np2 > n, '2 ** np2 == {} <= n == {}'.format(2 ** np2, n)
    assert_allclose(np2, np.log2(2 ** np2))


def test_fractional():
    """
    """
    n = 1
    m = 1000
    x = randn(n)
    xi = randint(m)
    assert fractional(x)
    assert fractional(rand())
    assert not fractional(xi)
    assert not fractional(randint(1, np.iinfo(int).max))


def test_ndtuples():
    t = rand_int_tuple()
    k = ndtuples(*t)
    set_k = set(np.unique(k.ravel()))
    assert 1


class TestZeroPad(unittest.TestCase):
    def setUp(self):
        """
        """
        n = 50
        ndims = 3
        self.x = randn_array(n=n, ndims=ndims)
        self.s = randint(n)

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


class TestPadLarger(unittest.TestCase):
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
    x = randn(n, n) + 1j
    assert iscomplex(x), 'x is not complex and has type {}'.format(x.dtype)


def test_get_fft_funcs():
    """
    """
    x = randn(randint(1, 1000), randint(1, 1000))
    xc = x + 1j
    
    ifft, fft = get_fft_funcs(xc, xc)
    assert fft is np.fft.fft
    assert ifft is np.fft.ifft
    assert_allclose(ifft(fft(xc.ravel())), xc.ravel())

    ifft, fft = get_fft_funcs(x, x)
    assert fft is np.fft.rfft
    assert ifft is np.fft.irfft
    assert_allclose(ifft(fft(x.ravel())), x.ravel())
        
    
