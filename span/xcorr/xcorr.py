#!/usr/bin/env python

# xcorr.py ---

# Copyright (C) 2012 Copyright (C) 2012 Phillip Cloud <cpcloud@gmail.com>

# Author: Phillip Cloud <cpcloud@gmail.com>

# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

import numpy as np
from pandas import Series, DataFrame
from span.utils import (detrend_mean, get_fft_funcs, isvector, nextpow2,
                        pad_larger)
from span.xcorr._mult_mat_xcorr import (_mult_mat_xcorr_parallel,
                                        _mult_mat_xcorr_serial)

import warnings

import numba
from numba import autojit, NumbaError, void, i8


try:
    T = numba.template("T")

    @autojit(void(T[:, :], T[:, :], T[:, :], i8, i8))
    def mult_mat_xcorr_numba(X, Xc, c, n, nx):
        for i in xrange(n):
            r = 0

            for j in xrange(i * n, (i + 1) * n):
                for k in xrange(nx):
                    c[j, k] = X[i, k] * Xc[r, k]

                r += 1
except NumbaError:
    pass


def mult_mat_xcorr_cython_parallel(X, Xc, c, n, nx):
    """Perform the necessary matrix-vector multiplication and fill the cross-
    correlation array. Slightly faster than pure Python.

    Parameters
    ----------
    X, Xc, c : c16[:, :]
    n, nx : ip

    Raises
    ------
    AssertionError
       If n <= 0 or nx <= 0
    """
    assert X is not None, '1st argument "X" must not be None'
    assert Xc is not None, '2nd argument "Xc" must not be None'
    _mult_mat_xcorr_parallel(X, Xc, c, n, nx)


def mult_mat_xcorr_cython_serial(X, Xc, c, n, nx):
    """Perform the necessary matrix-vector multiplication and fill the cross-
    correlation array. Slightly faster than pure Python.

    Parameters
    ----------
    X, Xc, c : c16[:, :]
    n, nx : ip

    Raises
    ------
    AssertionError
       If n <= 0 or nx <= 0
    """
    assert X is not None, '1st argument "X" must not be None'
    assert Xc is not None, '2nd argument "Xc" must not be None'
    _mult_mat_xcorr_serial(X, Xc, c, n, nx)


def mult_mat_xcorr_python(X, Xc, c, n, nx):
    assert X is not None, '1st argument "X" must not be None'
    assert Xc is not None, '2nd argument "Xc" must not be None'

    for i in xrange(n):
        c[i * n:(i + 1) * n] = X[i] * Xc


def mult_mat_xcorr(X, Xc):
    n, nx = X.shape
    c = np.empty((n ** 2, nx), X.dtype)

    try:
        mult_mat_xcorr_numba(X, Xc, c, n, nx)
    except (NameError, NumbaError):
        mult_mat_xcorr_cython_parallel(X, Xc, c, n, nx)

    return c


def autocorr(x, nfft):
    """Compute the autocorrelation of `x` using a FFT.

    Parameters
    ----------
    x : array_like
        Input array.

    nfft : int
        Number of FFT points.

    Returns
    -------
    r : array_like
        The autocorrelation of `x`.
    """
    ifft, fft = get_fft_funcs(x)
    a = np.abs(fft(x, nfft))
    a *= a
    return ifft(a, nfft)


def crosscorr(x, y, nfft):
    """Compute the cross correlation of `x` and `y` using an FFT.

    Parameters
    ----------
    x, y : array_like
        The arrays of which to compute the cross correlation.

    nfft : int
        The number of fft points.

    Returns
    -------
    c : array_like
        Cross correlation of `x` and `y`.
    """
    ifft, fft = get_fft_funcs(x, y)
    return ifft(fft(x, nfft) * fft(y, nfft).conj(), nfft)


def matrixcorr(x, nfft):
    """Cross-correlation of the columns of a matrix.

    Parameters
    ----------
    x : array_like
        The matrix from which to compute the cross correlations of each column
        with the others

    nfft : int
        The number of points used to compute the FFT (faster when this number
        is a power of 2).

    Returns
    -------
    c : array_like
        The cross correlation of the columns `x`.
    """
    _, n = x.shape
    ifft, fft = get_fft_funcs(x)
    X = fft(x.T, nfft)
    Xc = X.conj()
    c = mult_mat_xcorr(X, Xc)
    return ifft(c, nfft).T


def unbiased(c, x, y, lags, lsize):
    """Compute the unbiased estimate of `c`.

    This function returns `c` scaled by number of data points available at
    each lag.

    Parameters
    ----------
    c : array_like
        The cross correlation array

    x : array_like
    y : array_like

    lags : array_like

    lsize : int
        The size of the largest of the inputs to the cross correlation
        function.

    Returns
    -------
    c : array_like
        The unbiased estimate of the cross correlation.
    """
    d = lsize - np.abs(lags)
    d[np.logical_not(d)] = 1.0
    denom = np.tile(d[:, np.newaxis], (1, c.shape[1])) if c.ndim == 2 else d
    return c / denom


def biased(c, x, y, lags, lsize):
    """Compute the biased estimate of `c`.

    Parameters
    ----------
    c : array_like
        The cross correlation array

    x, y, lags : array_like, array_like, array_like

    lsize : int
        The size of the largest of the inputs to the cross correlation
        function.

    Returns
    -------
    csc : array_like
        The biased estimate of the cross correlation.

    Notes
    -----
    Conceptually, when you choose this scaling procedure you are
    ignoring the fact that there is a different amount of data at each of the
    different lags, thus this procedure is called biased. Only the lag
    0 cross/auto-correlation is a **true** estimate of the actual
    cross/auto-correlation function.

    See Also
    --------
    span.xcorr.xcorr.unbiased
    span.xcorr.xcorr.normalize
    """
    return c / lsize


def normalize(c, x, y, lags, lsize):
    """Normalize `c` by the lag 0 cross correlation

    Parameters
    ----------
    c : array_like
        The cross correlation array to normalize

    x : array_like
    y : array_like

    lags : array_like

    lsize : int
        The size of the largest of the inputs to the cross correlation
        function

    Raises
    ------
    AssertionError
        * If `c` is not a 1D or 2D array

    Returns
    -------
    c : array_like
        The normalized cross correlation.

    """
    assert c.ndim in (1, 2), 'invalid size of cross correlation array'

    if c.ndim == 1:
        ax2 = np.abs(x)
        ax2 *= ax2
        d = np.sum(ax2)

        if y is not None:
            ay2 = np.abs(y)
            ay2 *= ay2
            cy00 = np.sum(ay2)
            d *= cy00
        else:
            d *= d

        cdiv = np.sqrt(d)
    else:
        _, nc = c.shape
        ncsqrt = int(np.sqrt(nc))
        jkl = np.diag(np.r_[:nc].reshape(ncsqrt, ncsqrt))

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=FutureWarning)

            try:
                vals = c.ix[0, jkl]
            except AttributeError:
                vals = c[lags.max(), jkl]

        tmp = np.sqrt(vals)
        cdiv = np.outer(tmp, tmp).ravel()

    return c / cdiv


def none(c, x, y, lags, lsize):
    """Do nothing with the input and return `c`.

    Parameters
    ----------
    c, x, y, lags : array_like
    lsize : int
    """
    return c


_SCALE_FUNCTIONS = {
    None: none,
    'none': none,
    'unbiased': unbiased,
    'biased': biased,
    'normalize': normalize
}


_SCALE_KEYS = tuple(_SCALE_FUNCTIONS.keys())


def xcorr(x, y=None, maxlags=None, detrend=detrend_mean,
          scale_type='normalize'):
    """Compute the cross correlation of `x` and `y`.

    This function computes the cross correlation of `x` and `y`. It uses the
    equivalence of the cross correlation with the negative convolution computed
    using a FFT to achieve must faster cross correlation than is possible with
    the signal processing definition.

    By default it computes the normalized cross correlation.

    Parameters
    ----------
    x : array_like
        The array to correlate.

    y : array_like, optional
        If y is None or equal to `x` or x and y reference the same object,
        the autocorrelation is computed.

    maxlags : int, optional
        The maximum lag at which to compute the cross correlation. Must be less
        than or equal to the max(x.size, y.size) if not None.

    detrend : callable, optional
        A callable to detrend the data. It must take a single parameter and
        return an array

    scale_type : {None, 'none', 'unbiased', 'biased', 'normalize'}, optional
        * The type of scaling to perform on the data
        * The default of 'normalize' returns the cross correlation scaled by
          the lag 0 cross correlation i.e., the cross correlation scaled by the
          product of the standard deviations of the arrays at lag 0.

    Raises
    ------
    AssertionError
        * If `y` is not None and `x` is a matrix
        * If `x` is not a vector when `y` is None or `y` is `x` or
          ``all(x == y)``.
        * If `detrend` is not callable
        * If `scale_type` is not a string or ``None``
        * If `scale_type` is not in ``(None, 'none', 'unbiased', 'biased',
          'normalize')``
        * If `maxlags` ``>`` `lsize`, see source for details.

    Returns
    -------
    c : Series or DataFrame or array_like
        Autocorrelation of `x` if `y` is ``None``, cross-correlation of `x` if
        `x` is a matrix and `y` is ``None``, or the cross-correlation of `x`
        and `y` if both `x` and `y` are vectors.
    """
    assert x.ndim in (1, 2), 'x must be a 1D or 2D array'
    assert callable(detrend), 'detrend must be a callable object'
    assert isinstance(scale_type, basestring) or scale_type is None, \
        '"scale_type" must be a string or None'
    assert scale_type in _SCALE_KEYS, ('"scale_type" must be one of '
                                       '{0}'.format(_SCALE_KEYS))

    x = detrend(x)

    if x.ndim == 2 and np.greater(x.shape, 1).all():
        assert y is None, 'y argument not allowed when x is a 2D array'
        lsize = x.shape[0]
        inputs = x,
        corrfunc = matrixcorr
    elif y is None or y is x or np.array_equal(x, y):
        assert isvector(x), 'x must be 1D'
        lsize = max(x.shape)
        inputs = x,
        corrfunc = autocorr
    else:
        x, y, lsize = pad_larger(x, detrend(y))
        inputs = x, y
        corrfunc = crosscorr

    nfft = 2 ** nextpow2(2 * lsize - 1)
    ctmp = corrfunc(*inputs, nfft=nfft)

    if maxlags is None:
        maxlags = lsize

    assert maxlags <= lsize, ('max lags must be less than or equal to %i'
                              % lsize)

    lags = np.r_[1 - maxlags:maxlags]

    if isinstance(x, Series):
        return_type = lambda x, index: Series(x, index=index)
    elif isinstance(x, DataFrame):
        return_type = lambda x, index: DataFrame(x, index=index)
    elif isinstance(x, np.ndarray):
        return_type = lambda x, index: np.asanyarray(x)

    sc_func = _SCALE_FUNCTIONS[scale_type]
    return sc_func(return_type(ctmp[lags], index=lags), x, y, lags, lsize)
