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

import operator
import re
import itertools as itools
import functools as fntools


import numpy as np
import scipy.linalg
from pandas import Series, DataFrame, Panel, Panel4D
from six.moves import map


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


def detrend_mean(x, axis=0):
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
    if isinstance(x, Series):
        means = x.mean()
        return x - means
    elif isinstance(x, DataFrame):
        means = x.mean(axis)
        return x.sub(means, axis=1 - axis)
    elif isinstance(x, (Panel, Panel4D)):
        raise NotImplementedError('Detrending not implemented for Panel and '
                                  'Panel4D')
    elif np.isscalar(x):
        try:
            r = x.dtype.type(0)
        except AttributeError:
            r = type(x)(0)

        return r
    else:
        ma = np.ma.masked_where(np.isnan(x), x)
        means = np.atleast_1d(ma.mean(axis=axis))

        indexer = [slice(None)] * x.ndim
        indexer[axis] = np.newaxis
        m_ind = means[indexer]

        s = np.squeeze(x - m_ind)
        return s.item() if not s.ndim else s


def detrend_linear(y):
    """Linearly detrend `y`.

    Parameters
    ----------
    y : array_like

    Returns
    -------
    d : array_like
    """
    n = len(y)
    bp = np.array([0])
    bp = np.unique(np.vstack((0, bp, n - 1)))
    lb = len(bp) - 1
    zeros = np.zeros((n, lb))
    ones = np.ones((n, 1))
    zo = zeros, ones
    a = np.hstack(zo)

    for kb in xrange(lb):
        bpkb = bp[kb]
        m = n - bpkb
        a[np.r_[:m] + bpkb, kb] = np.r_[1:m + 1] / float(m)

    x, _, _, _ = np.linalg.lstsq(a, y)
    return y - np.dot(a, x)


def cartesian(*xs):
    r"""Returns the Cartesian product of arrays.

    The Cartesian product is defined as

        .. math::
            A_{1} \times \cdots \times A_{n} =
            \left\{\left(a_{1},\ldots,a_{n}\right) : a_{1} \in A_{1}
            \textrm{ and } \cdots \textrm{ and }a_{n} \in A_{n}\right\}

    Notes
    -----
    This function works on arbitrary object arrays and attempts to coerce the
    entire array to a single non-object dtype if possible.

    Parameters
    ----------
    arrays : tuple of array_like
    out : array_like

    Returns
    -------
    out : array_like
    """
    bcasted = np.broadcast_arrays(*np.ix_(*xs))
    rows, cols = reduce(operator.mul, bcasted[0].shape), len(bcasted)
    out = np.empty(rows * cols, dtype=bcasted[0].dtype)
    start, end = 0, rows

    for x in bcasted:
        out[start:end] = x.reshape(-1)
        start, end = end, end + rows

    return out.reshape(cols, rows).T


def nextpow2(n):
    """Return the next power of 2 of an array.

    Parameters
    ----------
    n : array_like

    Returns
    -------
    ret : array_like
    """
    f = compose(np.ceil, np.log2, np.abs, np.asanyarray)
    return f(n).astype(int)


def samples_per_ms(fs, millis):
    """Compute the number of samples in `ms` for a sample rate of `fs`

    Parameters
    ----------
    fs : float
        Sampling rate

    millis : float
        The refractory period in milliseconds.

    Returns
    -------
    win : int
        The refractory period in samples.
    """
    conv = 1000.0
    return int(np.floor(millis / conv * fs))


def compose2(f, g):
    """Return a function that computes the composition of two functions.

    Parameters
    ----------
    f, g : callable

    Raises
    ------
    AssertionError
       * If `f` and `g` are not both callable

    Returns
    -------
    h : callable
    """
    assert all(map(callable, (f, g))), 'f and g must both be callable'
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
    name_getter = operator.attrgetter('__name__')
    sbre = re.compile(r'[()]*')
    dotted_names = ' . '.join(map(name_getter, args))
    f.__name__ = '({0})'.format(sbre.sub('', dotted_names))
    return f


def composemap(*args):
    """Compose an arbitrary number of mapped functions.

    Parameters
    ----------
    args : tuple of callables

    Raises
    ------
    AssertionError
        * If not all arguments are callable

    Returns
    -------
    h : callable
    """
    assert all(map(callable, args)), 'all arguments must be callable'
    maps = itools.repeat(map, len(args))
    return fntools.reduce(compose2, map(fntools.partial, maps, args))


def _raw_cov(x):
    try:
        c = x.cov()
    except AttributeError:
        c = np.cov(x.T)

    return c


def _first_pc(x):
    c = _raw_cov(x)
    w, v = scipy.linalg.eig(c)
    return v[:, w.argmax()]


def _first_pc_cleaner_matrix(x):
    v = _first_pc(x)
    p = np.eye(v.size) - np.outer(v, v)

    try:
        return type(x)(p, x.columns, x.columns)
    except AttributeError:
        return p


def remove_first_pc(x):
    return x.dot(_first_pc_cleaner_matrix(x))
