#!/usr/bin/env python

# utils.py ---

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


"""A collection of utility functions."""

import os
import operator
import itertools
import functools
import numbers
from string import ascii_letters as _LETTERS

import numpy as np
from numpy.fft import fft, ifft, rfft, irfft

from pandas import datetime
from six.moves import map

from span.utils._clear_refrac import _clear_refrac as _clear_refrac_cython
from span.utils.math import cartesian

fromtimestamp = np.vectorize(datetime.fromtimestamp)


def ndtuples(*dims):
    """Create an array of arrays with `dims` dimensions.

    Return an array of the Cartesisan product of each of
    np.arange(dims[0]), ..., np.arange(dims[1]).

    Parameters
    ----------
    dims : tuple of int
        A tuple of dimensions to use to create the arange of each element.

    Raises
    ------
    AssertionError
        If no arguments are given

    Returns
    -------
    cur : array_like

    See Also
    --------
    span.utils.math.cartesian
        `ndtuples` is a special case of the Cartesian product
    """
    assert dims, 'no arguments given'
    assert all(map(lambda x: isinstance(x, (numbers.Integral)), dims)), \
        'all arguments must be integers'
    assert all(map(lambda x: x > 0, dims)), \
        'all arguments must be greater than 0'

    dims = list(dims)
    n = dims.pop()
    cur = np.arange(n)[:, np.newaxis]

    while dims:
        d = dims.pop()
        cur = np.kron(np.ones((d, 1), int), cur)
        front = np.arange(d).repeat(n)[:, np.newaxis]
        cur = np.hstack((front, cur))
        n *= d

    return cur.squeeze()


def name2num(name, base=256):
    """Convert an event name's string representation to a number.

    Parameters
    ----------
    name : str
        The name of the event.

    base : int, optional
        The base to use to compute the numerical representation of `name`.

    Returns
    -------
    ret : int
        The number corresponding to TDT's numerical representation of an event
        type string.
    """
    return (base ** np.r_[:len(name)]).dot(tuple(map(ord, name)))


_ORDS = list(map(ord, _LETTERS))
_MAXLEN = 4
_BIG_ORDS = cartesian([_ORDS] * _MAXLEN)


def num2name(num, base=256, maxlen=_MAXLEN):
    rhs = base ** np.arange(maxlen)
    row = _BIG_ORDS[np.dot(_BIG_ORDS, rhs) == num].ravel()
    return ''.join(map(chr, row))


def iscomplex(x):
    """Test whether `x` is any type of complex array.

    Parameters
    ----------
    x : array_like

    Returns
    -------
    r : bool
        Whether x's dtype is a sub dtype or equal to complex.
    """
    try:
        return np.issubdtype(x.dtype, np.complexfloating)
    except AttributeError:
        return any(map(np.issubdtype, x.dtypes,
                       itertools.repeat(np.complexfloating)))


def get_fft_funcs(*arrays):
    """Get the correct fft functions for the input type.

    Parameters
    ----------
    arrays : tuple of array_like
        Arrays to be checked for complex dtype.

    Returns
    -------
    r : tuple of callables
        The fft and ifft appropriate for the dtype of input.
    """
    return (ifft, fft) if any(map(iscomplex, arrays)) else (irfft, rfft)


def isvector(x):
    """Test whether `x` is a vector, i.e., ...

    Parameters
    ----------
    x : array_like

    Raises
    ------
    AssertionError

    Returns
    -------
    b : bool
    """
    try:
        newx = np.asanyarray(x)
        return functools.reduce(operator.mul, newx.shape) == max(newx.shape)
    except:
        return False


def assert_nonzero_existing_file(f):
    assert os.path.exists(f), '%s does not exist'
    assert os.path.isfile(f), '%s is not a file'
    assert os.path.getsize(f) > 0, \
        '%s exists and is a file, but it has a size of 0'


def clear_refrac(a, window):
    """Clear the refractory period of a boolean array.

    Parameters
    ----------
    a : array_like
    window : npy_intp

    Notes
    -----
    If ``a.dtype == np.bool_`` in Python then this function will not work
    unless ``a.view(uint8)`` is passed.

    Raises
    ------
    AssertionError
    If `window` is less than or equal to 0
    """
    assert isinstance(a, np.ndarray), 'a must be a numpy array'
    assert isinstance(window, (numbers.Integral, np.integer)), \
        '"window" must be an integer'
    assert window > 0, '"window" must be greater than 0'
    _clear_refrac_cython(a.view(np.int8), window)


def ispower2(x):
    b = np.log2(x)
    e, m = np.modf(b)
    return 0 if e else m
