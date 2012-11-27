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
import pandas as pd

from span.utils import (detrend_mean, get_fft_funcs, isvector, nextpow2,
                        pad_larger)

from span.xcorr._mult_mat_xcorr import mult_mat_xcorr


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
    mx, nx = X.shape
    c = np.empty((mx ** 2, nx), dtype=X.dtype)
    mult_mat_xcorr(X, Xc, c, n, nx)
    return ifft(c, nfft).T


def _unbiased(c, x, y, lsize):
    """Compute the unbiased estimate of `c`.

    This function returns `c` scaled by number of data points available at
    each lag.

    Parameters
    ----------
    c : array_like
        The cross correlation array

    x : array_like
    y : array_like

    lsize : int
        The size of the largest of the inputs to the cross correlation
        function.

    Returns
    -------
    c : array_like
        The unbiased estimate of the cross correlation.
    """
    d = lsize - np.abs(c.index).values
    denom = np.tile(d[:, np.newaxis], (1, c.shape[1])) if c.ndim == 2 else d
    rt = type(c)
    vals = c.values / denom

    try:
        return rt(vals, index=c.index)
    except (TypeError, AttributeError):
        return rt(vals)


def _biased(c, x, y, lsize):
    """Compute the biased estimate of `c`.

    Parameters
    ----------
    c : array_like
        The cross correlation array

    x : array_like
    y : array_like

    lsize : int
        The size of the largest of the inputs to the cross correlation
        function.

    Returns
    -------
    c : array_like
        The biased estimate of the cross correlation.
    """
    return c / lsize


def _normalize(c, x, y, lsize):
    """Normalize `c` by the lag 0 cross correlation

    Parameters
    ----------
    c : array_like
        The cross correlation array to normalize

    x : array_like
    y : array_like

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
        cx00 = np.sum(np.abs(x) ** 2)
        cy00 = np.sum(np.abs(y) ** 2) if y is not None else cx00
        cdiv = np.sqrt(cx00 * cy00)
    else:
        _, nc = c.shape
        ncsqrt = int(np.sqrt(nc))
        jkl = np.diag(np.r_[:nc].reshape(ncsqrt, ncsqrt))

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=FutureWarning)

            try:
                vals = c.ix[0, jkl]
            except AttributeError:
                vals = c[0, jkl]

            tmp = np.sqrt(vals)

        cdiv = np.outer(tmp, tmp).ravel()

    return c / cdiv


def _none(c, x, y, lsize):
    return c


_SCALE_FUNCTIONS = {
    None: _none,
    'none': _none,
    'unbiased': _unbiased,
    'biased': _biased,
    'normalize': _normalize
}


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
        * If `x` is not a vector when y is None or y is x or all(x == y)
        * If `detrend` is not callable
        * If `scale_type` is not a string or ``None``
        * If `maxlags` is > `lsize`, see source for details.

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

    if x.ndim == 2:
        x = pd.DataFrame(x)
    else:
        x = pd.Series(x)

    x = detrend(x)

    if x.ndim == 2 and np.greater(x.shape, 1).all():
        assert y is None, 'y argument not allowed when x is a 2D array'
        lsize = x.shape[0]
        inputs = x,
        corrfunc = _matrixcorr
    elif y is None or y is x or np.array_equal(x, y):
        assert isvector(x), 'x must be 1D'
        lsize = max(x.shape)
        inputs = x,
        corrfunc = _autocorr
    else:
        x, y, lsize = pad_larger(x, detrend(y))
        inputs = x, y
        corrfunc = _crosscorr

    ctmp = corrfunc(*inputs, nfft=2 ** nextpow2(2 * lsize - 1))

    if maxlags is None:
        maxlags = lsize

    assert maxlags <= lsize, ('max lags must be less than or equal to %i'
                              % lsize)

    lags = pd.Int64Index(np.r_[1 - maxlags:maxlags])

    if isinstance(x, pd.Series):
        return_type = lambda x, index: pd.Series(x, index=index)
    elif isinstance(x, pd.DataFrame):
        return_type = lambda x, index: pd.DataFrame(x, index=index)
    elif isinstance(x, np.ndarray):
        return_type = lambda x, index: np.asanyarray(x)

    scale_function = _SCALE_FUNCTIONS[scale_type]
    return scale_function(return_type(ctmp[lags], index=lags), x, y, lsize)
