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

import warnings

import numpy as np
from pandas import Series, DataFrame
from six.moves import xrange

from span.utils import get_fft_funcs, isvector, nextpow2
from span.xcorr._mult_mat_xcorr import _mult_mat_xcorr_parallel


def _mult_mat_xcorr_cython_parallel(X, Xc, c, n):
    """Perform the necessary matrix-vector multiplication and fill the cross-
    correlation array. Slightly faster than pure Python.

    Parameters
    ----------
    X, Xc, c : c16[:, :]
    n : ip

    Raises
    ------
    AssertionError
       If n <= 0 or nx <= 0
    """
    nx = c.shape[1]
    _mult_mat_xcorr_parallel(X, Xc, c, n, nx)


def _mult_mat_xcorr_python(X, Xc, c, n):
    for i in xrange(n):
        c[i * n:(i + 1) * n] = X[i] * Xc


def _mult_mat_xcorr(X, Xc):
    assert X is not None, '1st argument "X" must not be None'
    assert Xc is not None, '2nd argument "Xc" must not be None'

    n, nx = X.shape
    c = np.empty((n * n, nx), dtype=X.dtype)
    _mult_mat_xcorr_cython_parallel(X, Xc, c, n)
    return c


def _autocorr(x, nfft):
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


def _crosscorr(x, y, nfft):
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


def _matrixcorr(x, nfft):
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
    c = _mult_mat_xcorr(X, Xc)
    return ifft(c, nfft).T


def _unbiased(c, x, y, lags, lsize):
    r"""Compute an unbiased estimate of `c`.

    This function returns `c` scaled by the number of data points
    available at each lag.

    Parameters
    ----------
    c : array_like
        The cross correlation array

    x, y : array_like

    lags : array_like
        The lags array, e.g., :math:`\left[\ldots, -2, -1, 0, 1, 2,
        \ldots\right]`

    lsize : int
        The size of the largest of the inputs to the cross correlation
        function.

    Returns
    -------
    c : array_like
        The unbiased estimate of the cross correlation.
    """
    # max number of observations minus observations at each lag
    d = lsize - np.abs(lags)

    # protect divison by zero
    d[np.logical_not(d)] = 1.0

    # make the denominator repeat over the correct dimension
    denom = np.tile(d[:, np.newaxis], (1, c.shape[1])) if c.ndim == 2 else d
    return c / denom


def _biased(c, x, y, lags, lsize):
    """Compute a biased estimate of `c`.

    Parameters
    ----------
    c : array_like
        The unscaled cross correlation array

    x, y, lags : array_like
        Unused; here to keep the API sane

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
    ignoring the fact that there is a different amount of data at each
    of the different lags, thus this procedure is called "biased".
    Only the lag 0 cross-correlation is an unbiased estimate of
    thecross-correlation function.

    See Also
    --------
    span.xcorr.xcorr.unbiased
    span.xcorr.xcorr.normalize
    """
    return c / lsize


def _normalize(c, x, y, lags, lsize):
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
    assert c.ndim in (1, 2), ('invalid number of dimensions of cross '
                              'correlation array, INPUT: %d, EXPECTED: 1 or 2'
                              ', i.e., vector or matrix' % c.ndim)

    # vector
    if c.ndim == 1:
        # need this for either cross or auto
        ax2 = np.abs(x)
        ax2 *= ax2
        d = np.sum(ax2)

        # y is given so we computed a cross correlation
        if y is not None:
            ay2 = np.abs(y)
            ay2 *= ay2
            cy00 = np.sum(ay2)
            d *= cy00
            cdiv = np.sqrt(d)
        else:
            cdiv = d
    else:
        # matrix
        _, nc = c.shape
        ncsqrt = int(np.sqrt(nc))

        # need diagonal elements of array
        jkl = np.diag(np.r_[:nc].reshape(ncsqrt, ncsqrt))

        # ignore annoying numpy warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=FutureWarning)

            try:
                # pandas lag 0 jklth column pair
                vals = c.ix[0, jkl]
            except AttributeError:
                # not pandas so assume it's a numpy array
                vals = c[lags.max(), jkl]

        # scale by lag 0 of each column pair
        tmp = np.sqrt(vals)
        cdiv = np.outer(tmp, tmp).ravel()

    return c / cdiv


def _none(c, x, y, lags, lsize):
    """Do nothing with the input and return `c`.

    Parameters
    ----------
    c, x, y, lags : array_like
    lsize : int
    """
    return c


_SCALE_FUNCTIONS = {
    None: _none,
    'none': _none,
    'unbiased': _unbiased,
    'biased': _biased,
    'normalize': _normalize
}


_SCALE_KEYS = tuple(_SCALE_FUNCTIONS.keys())


def xcorr(x, y=None, maxlags=None, detrend=None, scale_type=None):
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
    assert callable(detrend) or detrend is None, \
        'detrend must be a callable object or None'
    assert isinstance(scale_type, basestring) or scale_type is None, \
        '"scale_type" must be a string or None'
    assert scale_type in _SCALE_KEYS, ('"scale_type" must be one of '
                                       '{0}'.format(_SCALE_KEYS))

    if detrend is None:
        detrend = lambda x: x

    x = detrend(x)

    if x.ndim == 2 and np.greater(x.shape, 1).all():
        assert y is None, 'y argument not allowed when x is a 2D array'
        lsize = x.shape[0]
        inputs = x,
        corrfunc = _matrixcorr
    elif y is None or y is x or np.array_equal(x, y) or np.allclose(x, y):
        assert isvector(x), 'x must be 1D'
        lsize = x.shape[0]
        inputs = x,
        corrfunc = _autocorr
    else:
        lsize = max(x.size, y.size)
        y = detrend(y)
        inputs = x, y
        corrfunc = _crosscorr

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

    # import ipdb
    sc_func = _SCALE_FUNCTIONS[scale_type]
    # ipdb.set_trace()
    return sc_func(return_type(ctmp[lags], index=lags), x, y, lags, lsize)
