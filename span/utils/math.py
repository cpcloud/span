#!/usr/bin/env python

# math.py ---

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


from future_builtins import map

import numbers
import operator
import itertools as itools
import functools as fntools

import numpy as np
from pandas import Series, DataFrame


try:
    # weird bug in latest scipy
    from scipy.stats.mstats import trimboth

    def trimmean(x, alpha, inclusive=(False, False), axis=None):
        """Compute the `alpha`-trimmed mean of an array `x`.

        Parameters
        ----------
        x : array_like
            The array on which to operate.

        alpha : float or int
            A number between 0 and 100, left inclusive indicating the
            percentage of values to cut from `x`.

        inclusive : tuple of bools, optional
            Whether to round (True, True) or truncate the values
            (False, False). Defaults to truncation. Note that this is different
            from ``scipy.stats.mstats.trimboth``'s default.

        axis : int or None, optional
            The axis over which to operate. None flattens the array

        Returns
        -------
        m : Series
            The `alpha`-trimmed mean of `x` along axis `axis`.
        """
        assert 0 <= alpha < 100, 'alpha must be in the interval [0, 100)'
        assert len(inclusive) == 2, 'inclusive must have only 2 elements'

        if isinstance(x, (numbers.Number)) or (hasattr(x, 'size') and
                                               x.size == 1):
            return float(x)

        assert axis is None or 0 <= axis < x.ndim, \
            'axis must be None or less than x.ndim: {0}'.format(x.ndim)

        trimmed = trimboth(x, alpha / 100.0, inclusive, axis).mean(axis)

        index = None
        if isinstance(x, DataFrame):
            index = {0: x.columns, 1: x.index, None: None}[axis]

        return Series(trimmed, index=index)

except ImportError:  # pragma: no cover
    def trimmean(x, alpha, inclusive=(False, False), axis=None):
        raise NotImplementedError("Unable to import scipy.stats;" +
                                  " cannot define trimmean")


def sem(a, axis=0, ddof=1):
    """Return the standard error of the mean of an array.

    Parameters
    ----------
    a : array_like
    axis : int, optional
    dtype : dtype, optional
    out : array_like, optional
    ddof : int, optional

    Returns
    -------
    sem : array_like
    """
    if np.isscalar(a):
        return 0.0

    n = a.shape[axis]

    try:
        s = a.std(axis=axis, ddof=ddof)
    except:
        s = a.std(axis=axis)

    return s / np.sqrt(n)


def detrend_none(x):
    """Return the input array.

    Parameters
    ----------
    x : array_like
        The input array.

    Returns
    -------
    x : array_like
        The input array.
    """
    return x


def detrend_mean(x):
    """Subtract the mean of `x` from itself.

    Parameters
    ----------
    x : array_like
        The array to mean center.

    Returns
    -------
    c : array_like
        The mean centered `x`.
    """
    return x - x.mean()


# def detrend_linear(y):
#     """Linearly detrend `y`.

#     Parameters
#     ----------
#     y : array_like

#     Returns
#     -------
#     d : array_like
#     """
#     x = np.arange(min(y.shape), dtype=float)

#     if y.ndim == 2:
#         x = np.tile(x, (y.shape[1], 1)).T

#     c = np.cov(x, y, bias=1)
#     b = c[0, 1] / c[0, 0]
#     a = y.mean() - b * x.mean()
#     return y - (b * x + a)


def detrend_linear(y):
    n = len(y)
    bp = np.array([0])
    bp = np.unique(np.vstack((0, bp, n - 1)))
    lb = len(bp) - 1
    a = np.hstack((np.zeros((n, lb)), np.ones((n, 1))))

    for kb in xrange(lb):
        bpkb = bp[kb]
        m = n - bpkb
        a[np.r_[:m] + bpkb, kb] = np.r_[1:m + 1] / float(m)

    x, _, _, _ = np.linalg.lstsq(a, y)
    return y - np.dot(a, x)


def cartesian(arrays, out=None, dtype=None):
    r"""Returns the Cartesian product of arrays.

    The Cartesian product is defined as

.. math::
     A_{1} \times \cdots \times A_{n} =
     \left\{\left(a_{1},\ldots,a_{n}\right) : a_{1} \in A_{1}\textrm{ and }
     \cdots \textrm{ and }a_{n} \in A_{n}\right\}

    Notes
    -----
    This function works on arbitrary objects arrays and attempts to coerce the
    entire array to a single non-object dtype if possible.

    Parameters
    ----------
    arrays : tuple of array_like
    out : array_like

    Returns
    -------
    out : array_like
    """
    arrays = tuple(map(np.asanyarray, arrays))
    dtypes = tuple(map(operator.attrgetter('dtype'), arrays))
    all_dtypes_same = all(map(operator.eq, dtypes, itools.repeat(dtypes[0])))
    dtype = dtypes[0] if all_dtypes_same else object

    n = np.prod(tuple(map(operator.attrgetter('size'), arrays)))

    if out is None:
        out = np.empty((n, len(arrays)), dtype=dtype)

    m = n / arrays[0].size
    out[:, 0] = np.repeat(arrays[0], m)

    if arrays[1:]:
        cartesian(arrays[1:], out=out[:m, 1:])

        for j in xrange(1, arrays[0].size):
            out[j * m:(j + 1) * m, 1:] = out[:m, 1:]

    return out


def nextpow2(n):
    """Return the next power of 2 of a number.

    Parameters
    ----------
    n : array_like

    Returns
    -------
    ret : array_like
    """
    return np.ceil(np.log2(np.abs(np.asanyarray(n))))


def fractional(x):
    """Test whether an array has a fractional part.

    Parameters
    ----------
    x : array_like

    Returns
    -------
    ret : bool
        Whether the elements of x have a fractional part.
    """
    frac, _ = np.modf(np.asanyarray(x))
    return frac


def fs2ms(fs, millis):
    """Compute the number of samples in `ms` for a sample rate of `fs`

    Parameters
    ----------
    fs : float
        Sampling rate

    ms : float
        The refractory period in milliseconds.

    Returns
    -------
    win : int
        The refractory period in samples.
    """
    conv = 1e3
    return int(np.floor(millis / conv * fs))


def compose2(f, g):
    """Return a function that computes the composition of two functions.

    Parameters
    ----------
    f, g : callable

    Returns
    -------
    h : callable
    """
    if not all(map(callable, (f, g))):
        raise TypeError('f and g must both be callable')
    return lambda *args, **kwargs: f(g(*args, **kwargs))


def compose(*args):
    """Compose an arbitrary number of functions.

    Parameters
    ----------
    args : tuple of callables

    Returns
    -------
    h : callable
        Composition of callables in `args`.
    """
    f = fntools.partial(fntools.reduce, compose2)(args)
    f.__name__ = '({0})'.format(' . '.join(map(lambda x: x.__name__, args)))
    return f


def composemap(*args):
    """Compose an arbitrary number of mapped functions.

    Parameters
    ----------
    args : tuple of callables

    Returns
    -------
    h : callable
    """
    maps = itools.repeat(map, len(args))
    return fntools.reduce(compose2, map(fntools.partial, maps, args))
