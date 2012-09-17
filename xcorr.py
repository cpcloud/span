#!/usr/bin/env python

"""
Module for cross-correlation.
"""

import numpy as np

from span.utils import (nextpow2, pad_larger, get_fft_funcs, detrend_mean,
                        Series, DataFrame, isvector)


def autocorr(x, n):
    """Compute the autocorrelation of `x` using a FFT.

    Parameters
    ----------
    x : array_like
        Input array.

    n : int
        Number of FFT points.

    Returns
    -------
    r : array_like
        The autocorrelation of `x`.
    """        
    ifft, fft = get_fft_funcs(x)
    return ifft(np.abs(fft(x, n)) ** 2.0, n)


def crosscorr(x, y, n):
    """Compute the cross correlation of `x` and `y` using a FFT.


    Parameters
    ----------
    x, y : array_like
        The arrays of which to compute the cross correlation.

    n : int
        The number of fft points.

    Returns
    -------
    c : array_like
        Cross correlation of `x` and `y`.
    """
    ifft, fft = get_fft_funcs(x, y)
    return ifft(fft(x, n) * fft(y, n).conj(), n)


def matrixcorr(x, nfft=None):
    """Cross-correlation of the columns in a matrix

    Parameters
    ----------
    x : array_like
        The matrix from which to compute the cross correlations of each column
        with the others

    nfft : int, optional
        The number of points used to compute the FFT (faster when this number is
        a power of 2).

    Returns
    -------
    c : array_like
        The cross correlation of the columns `x`.
    """
    m, n = x.shape
    if nfft is None:
        nfft = int(2 ** nextpow2(2 * m - 1))
    ifft, fft = get_fft_funcs(x)
    X = fft(x.T, nfft)
    Xc = X.conj()
    mx, nx = X.shape
    c = np.empty((mx ** 2, nx), dtype=X.dtype)
    for i in xrange(n):
        c[i * n:(i + 1) * n] = X[i] * Xc
    return ifft(c, nfft).T, m


def unbiased(c, lsize):
    """Compute the unbiased estimate of `c`.

    This function returns `c` scaled by number of data points available at
    each lag.

    Parameters
    ----------
    c : array_like
        The cross correlation array

    lsize : int
        The size of the largest of the inputs to the cross correlation
        function.

    Returns
    -------
    c : array_like
        The unbiased estimate of the cross correlation.
    """
    return c / (lsize - np.abs(c.index))


def biased(c, lsize):
    """Compute the biased estimate of `c`.

    Parameters
    ----------
    c : array_like
        The cross correlation array

    lsize : int
        The size of the largest of the inputs to the cross correlation
        function.

    Returns
    -------
    c : array_like
        The biased estimate of the cross correlation.
    """
    return c / lsize


def normalize(c, lsize):
    """Normalize `c` by the lag 0 cross correlation

    Parameters
    ----------
    c : array_like
        The cross correlation array

    lsize : int
        The size of the largest of the inputs to the cross correlation
        function.

    Returns
    -------
    c : array_like
        The normalized cross correlation.
    """
    assert c.ndim in (1, 2), 'invalid size of cross correlation array'

    if c.ndim == 1:
        cdiv = c.ix[0]
    else:
        _, nc = c.shape
        ncsqrt = int(np.sqrt(nc))
        jkl = np.diag(np.r_[:nc].reshape((ncsqrt, ncsqrt)))
        tmp = np.sqrt(c.ix[0, jkl])
        cdiv = np.outer(tmp, tmp).ravel()
    return c / cdiv


SCALE_FUNCTIONS = {
    None: lambda c, lsize: c,
    'none': lambda c, lsize: c,
    'unbiased': unbiased,
    'biased': biased,
    'normalize': normalize
}


def xcorr(x, y=None, maxlags=None, detrend=detrend_mean, scale_type='normalize'):
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
        If y is None or is equal to `x` or x and y reference the same object,
        the autocorrelation is computed.

    maxlags : int, optional
        The maximum lag at which to compute the cross correlation. Must be less
        than or equal to the max(x.size, y.size) if not None.

    detrend : callable, optional
        A callable to detrend the data. It must take a single parameter and
        return an array

    scale_type : {None, 'none', 'unbiased', 'biased', 'normalize'}, optional
        The type of scaling to perform on the data
        The default of 'normalize' returns the cross correlation scaled by the
        lag 0 cross correlation i.e., the cross correlation scaled by the
        product of the standard deviations of the signals at lag 0.

    Returns
    -------
    c : Series or DataFrame
        The 2 * `maxlags` - 1 length pandas.Series or the 2 * `maxlags` - 1 by
        x.shape[1] ** 2 pandas.DataFrame of all the cross correlations of the
        columns of x
    """

    assert x.ndim in (1, 2), 'x must be a vector or matrix'

    x = detrend(x)

    if x.ndim == 2 and np.greater(x.shape, 1).all():
        assert y is None, 'y argument not allowed when x is a 2D array'
        ctmp, lsize = matrixcorr(x)
    elif y is None or y is x or np.array_equal(x, y):
        assert isvector(x), 'x must be 1D'
        lsize = max(x.shape)
        ctmp = autocorr(x, int(2 ** nextpow2(2 * lsize - 1)))
    else:
        x, y, lsize = pad_larger(x, detrend(y))
        ctmp = crosscorr(x, y, int(2 ** nextpow2(2 * lsize - 1)))

    if maxlags is None:
        maxlags = lsize
    else:
        assert maxlags <= lsize, 'max lags must be less than or equal to %i' % lsize

    lags = np.r_[1 - maxlags:maxlags]
    return_type = DataFrame if ctmp.ndim == 2 else Series
    scaler = SCALE_FUNCTIONS[scale_type]
    return scaler(return_type(ctmp[lags], index=lags), lsize)
