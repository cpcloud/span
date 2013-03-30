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
import functools
import itertools
import operator
import os
import string
from string import ascii_letters as _LETTERS
import time
import warnings

import numpy as np
from numpy.fft import fft, ifft, rfft, irfft
from pandas import datetime, MultiIndex
from six.moves import map
import pytz
import numba as nb

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
    return cartesian(*map(np.arange, dims))


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
    return np.dot(base ** np.arange(len(name)), [ord(c) for c in name])


_ORDS = list(map(ord, _LETTERS))
_MAXLEN = 4
_BIG_ORDS = cartesian(*([_ORDS] * _MAXLEN))


def num2name(num, base=256, maxlen=_MAXLEN):
    # if the number could not possibly be a name
    if (num < _BIG_ORDS[0]).all():
        return ''

    rhs = base ** np.arange(maxlen)
    out = _BIG_ORDS[_BIG_ORDS.dot(rhs) == num].ravel()
    return ''.join(map(chr, out))


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
    assert os.path.exists(f), '%s does not exist' % f
    assert os.path.isfile(f), '%s is not a file' % f
    assert os.path.getsize(f) > 0, \
        '%s exists and is a file, but it has a size of 0' % f


def ispower2(x):
    b = np.log2(x)
    e, m = np.modf(b)
    return 0 if e or not m else m


def create_repeating_multi_index(columns, index_start_string='i'):
    """Create an appropriate index for cross correlation.

    Parameters
    ----------
    columns : MultiIndex
    index_start_string : basestring

    Returns
    -------
    mi : MultiIndex

    Notes
    -----
    I'm not sure if this is actually slick, or just insane seems like
    functional idioms are so concise as to be confusing sometimes,
    although maybe I'm just slow.

    This absolutely does not handle the case where there are more than
    52 levels in the index, because i haven't had a chance to think
    about it yet..
    """
    if not isinstance(columns, MultiIndex):
        try:
            inp = columns.values, columns.values
        except AttributeError:
            inp = columns, columns

        f = MultiIndex.from_arrays
        return f(cartesian(*inp).T)

    colnames = columns.names

    # get the index of the starting index string provided
    letters = string.ascii_letters
    first_ind = letters.index(index_start_string)

    # repeat endlessly
    cycle_letters = itertools.cycle(letters)

    # slice from the index of the first letter to that plus the number
    # of names
    sliced = itertools.islice(cycle_letters, first_ind, first_ind +
                              len(colnames))

    # alternate names and index letter
    srt = sorted(itertools.product(colnames, sliced), key=lambda x: x[-1])
    names = itertools.imap(' '.join, srt)

    # number of columns
    ncols = len(columns)

    # number levels
    nlevels = len(columns.levels)

    # {0, ..., ncols - 1} ^ nlevels
    xrs = itertools.product(*itertools.repeat(xrange(ncols), nlevels))

    all_inds = (tuple(itertools.chain.from_iterable(columns[i] for i in inds))
                for inds in xrs)

    return MultiIndex.from_tuples(tuple(all_inds), names=list(names))


def _diag_inds_n(n):
    return (n + 1) * np.arange(n)


def _get_local_tz():
    tznames = list(time.tzname)

    while tznames:
        try:
            return pytz.timezone(tznames.pop()).zone
        except pytz.UnknownTimeZoneError:
            pass
    else:
        warnings.warn('No time zone found, you may need to reinstall '
                      'pytz or matplotlib or both')
        return None


LOCAL_TZ = _get_local_tz()

_T, _U = map(nb.template, ('_T', '_U'))


@nb.autojit(nb.void(_T[:, :], _U), locals=dict(sp1=nb.int_))
def clear_refrac(a, window):
    nsamples, nchannels = a.shape

    for channel in range(nchannels):
        sample = 0

        while sample + window < nsamples:
            if a[sample, channel]:
                sp1 = sample + 1
                a[sp1:sp1 + window, channel] = False
                sample += window

            sample += 1
