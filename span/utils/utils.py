"""A collection of utility functions."""

from future_builtins import map, zip

import os
import operator
import glob
import itertools
import functools
import numbers

import numpy as np
import pandas as pd
import scipy.stats


try:
    from pylab import (detrend_linear, detrend_mean, detrend_none, figure, gca,
                       subplots)
except RuntimeError:
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


    def detrend_linear(y):
        """Linearly detrend `y`.

        Parameters
        ----------
        y : array_like

        Returns
        -------
        d : array_like
        """
        x = np.arange(len(y), dtype=float)
        c = np.cov(x, y, bias=1.0)
        b = c[0, 1] / c[0, 0]
        a = y.mean() - b * x.mean()
        return y - (b * x + a)

    def remove_legend(ax=None):
        pass

    gca = NotImplemented
    figure = NotImplemented
    subplots = NotImplemented


def cast(a, dtype, copy=False):
    """Cast `a` to dtype `dtype`.

    Attempt to cast `a` to a different dtype `dtype` without copying.

    Parameters
    ----------
    a : array_like
        The array to cast.

    dtype : numpy.dtype
        The dtype to cast the input array to.

    copy : bool, optional
        Whether to copy the data given the previous arguments.

    Raises
    ------
    AssertionError

    Returns
    -------
    r : array_like
        The array `a` casted to type dtype
    """
    assert hasattr(a, 'dtype'), 'first argument has no "dtype" attribute'
    assert hasattr(a, 'astype'), 'first argument has no "astype" attribute'

    if a.dtype == dtype:
        return a

    try:
        r = a.astype(dtype, order='K', casting='safe', subok=True, copy=copy)
    except TypeError:
        r = a.astype(dtype)
    return r


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
    cartesian
        `ndtuples` is a special case of the Cartesian product
    """
    assert dims, 'no arguments given'
    dims = list(dims)
    n = dims.pop()
    cur = np.arange(n)[:, np.newaxis]
    while dims:
        d = dims.pop()
        cur = np.kron(np.ones((d, 1), int), cur)
        front = np.arange(d).repeat(n)[:, np.newaxis]
        cur = np.hstack((front, cur))
        n *= d
    return cur


def cartesian(arrays, out=None, dtype=None):
    """Cartesian product of arrays.

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
    all_dtypes_same = all(map(operator.eq, dtypes, itertools.repeat(dtypes[0])))
    dtype = dtypes[0] if all_dtypes_same else np.object_
    
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


def dirsize(d='.'):
    """Recusively compute the size of a directory.

    Parameters
    ----------
    d : str, optional
        The directory of which to compute the size. Defaults to the current
        directory.

    Returns
    -------
    s : int
        The size of the directory `d`.
    """
    s = os.path.getsize(d)
    for item in glob.glob(os.path.join(d, '*')):
        path = os.path.join(d, item)
        if os.path.isfile(path):
            s += os.path.getsize(path)
        elif os.path.isdir(path):
            s += dirsize(path)
    return s


def ndlinspace(ranges, *nelems):
    """Create `n` linspaces between the ranges of `ranges` with `nelems`
    elements.

    Parameters
    ----------
    ranges : array_like
    nelems : int, optional

    Returns
    -------
    n_linspaces : array_like
    """
    x = ndtuples(*nelems) + 1.0
    b = np.asanyarray(nelems)
    zipped = zip(*((r[0], r[1]) for r in ranges))
    lbounds, ubounds = map(np.fromiter, zipped, (float, float))
    return (lbounds + (x - 1) / (b - 1) * (ubounds - lbounds)).T


def nans(shape):
    """Create an array of NaNs

    Parameters
    ----------
    shape : tuple
        The shape tuple of the array of nans to create.

    dtype : numpy.dtype, optional
        The dtype of the new nan array. Defaults to float because only
        float nan arrays are support by NumPy.

    Returns
    -------
    a : array_like
    """
    a = np.empty(shape)
    a.fill(np.nan)
    return a


def remove_legend(ax=None):
    """Remove legend for ax or the current axes if ax is None.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis whose legend will be hidden
    """
    if ax is None:
        ax = gca()
    ax.legend_ = None


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
    return (base ** np.r_[:len(name)]).dot(np.fromiter(map(ord, name), int))


# TODO: THIS IS SO SLOW!
def num2name(num, base=256, slen=4):
    """Inverse of `name2num`.

    Parameters
    ----------
    num : int
        The number to convert to a valid string.

    base : int, optional
        The base to use for conversion.

    slen : int, optional
        The allowable length of the word.

    Returns
    -------
    ret : str
        The string associated with `num`.
    """
    import string
    letters = string.ascii_letters
    x = pd.Series(dict(zip(letters, map(ord, letters))))
    base_vec = base ** np.r_[:slen]
    xad = x[ndtuples(*itertools.repeat(len(letters), slen))] * base_vec
    w = xad[xad.sum(1) == num].squeeze() / base_vec
    return ''.join(map(chr, w))


def group_indices(group, dtype=int):
    """Return the indices of a particular grouping as a `pandas.DataFrame`.

    Parameters
    ----------
    group : pandas.Grouper
    dtype : dtype

    Returns
    -------
    inds : pandas.DataFrame
    """
    inds = pd.DataFrame(group.indices)
    inds.columns = cast(inds.columns, dtype)
    return inds


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


def pad_larger2(x, y):
    """Pad the larger of two arrays and the return the arrays and the size of
    the larger array.

    Parameters
    ----------
    x, y : array_like

    Returns
    -------
    x, y : array_like
    lsize : int
        The size of the larger of `x` and `y`.
    """
    xsize, ysize = x.size, y.size
    lsize = max(xsize, ysize)
    if xsize != ysize:
        size_diff = lsize - min(xsize, ysize)

        pad_func = lambda a: np.pad(a, (0, size_diff), mode='constant',
                               constant_values=(0,))

        if xsize > ysize:
            y = pad_func(y)
        else:
            x = pad_func(x)

    return x, y, lsize


def pad_larger(*arrays):
    """Pad the smallest of `n` arrays.

    Parameters
    ----------
    arrays : tuple of array_like

    Raises
    ------
    AssertionError

    Returns
    -------
    ret : list
        List of zero padded arrays.
    """
    assert all(map(isinstance, arrays, itertools.repeat(np.ndarray))), \
    'all arguments must be instances of ndarray or implement the ndarray interface'
    if len(arrays) == 2:
        return pad_larger2(*arrays)

    sizes = np.fromiter(map(operator.attrgetter('size'), arrays), int)
    lsize = sizes.max()

    ret = ()
    for array, size in zip(arrays, sizes):
        size_diff = lsize - size
        ret += np.pad(array, (0, size_diff), 'constant', constant_values=(0,)),
    ret += lsize,
    return ret


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
        cfloat = np.complexfloating
        return any(map(np.issubdtype, x.dtypes, itertools.repeat(cfloat)))


def hascomplex(x):
    """Check whether an array has all complex entries.

    Parameters
    ----------
    x : array_like

    Returns
    -------
    r : bool
    """
    try:
        return iscomplex(x) and not np.logical_not(x.imag).all()
    except AttributeError:
        return iscomplex(x) and not np.logical_not(x.values.imag).all()


def get_fft_funcs(*arrays):
    """Get the correct fft functions for the input type.

    Parameters
    ----------
    arrays : tuple of array_like
        Arrays to be checked for complex dtype.

    Returns
    -------
    r : 2-tuple of callables
        The fft and ifft appropriate for the dtype of input.
    """
    r = np.fft.irfft, np.fft.rfft
    arecomplex = functools.partial(map, iscomplex)
    if any(arecomplex(arrays)):
        r = np.fft.ifft, np.fft.fft
    return r


# def electrode_distance_old(fromij, toij, between_shank=125, within_shank=100):
#     fromi, fromj = fromij
#     toi, toj = toij

#     col_diff = (toj - fromj) * between_shank
#     row_diff = (toi - fromi) * within_shank
#     return np.sqrt(col_diff ** 2 + row_diff ** 2)


def electrode_distance(locs, bs=125.0, ws=100.0):
    """Compute the distance between two electrodes given an index and between and
    within shank distance.

    Parameters
    ----------
    fromij : tuple
    toij : tuple
    between_shank : int, optional
    within_shank : int, optional

    Returns
    -------
    d : float
        The distance between electrodes at e_ij and e_kl.
    """
    assert locs.ndim == 2, 'invalid locations array'
    ncols = locs.shape[1]
    ncols2 = int(np.floor(ncols / 2.0))
    d = ((locs[:, ncols2:ncols] - locs[:, :ncols2]) * [bs, ws]) ** 2.0
    s = locs.shape[0]
    si = int(np.sqrt(s))
    dist = np.sqrt(d.sum(axis=1))
    assert ~(dist[np.diag(np.r_[:s].reshape((si, si)))]).all(), \
        'self distance is not 0'
    return dist


def distance_map(nshanks=4, electrodes_per_shank=4):
    """Create an electrode distance map.

    Parameters
    ----------
    nshanks : int, optional
    electrodes_per_shank : int, optional

    Returns
    -------
    ret : pandas.Series
    """
    indices = ndtuples(*itertools.repeat(electrodes_per_shank, nshanks))
    dists = electrode_distance(indices)
    nelectrodes = nshanks * electrodes_per_shank
    return pd.DataFrame(dists.reshape((nelectrodes, nelectrodes)))


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
    return functools.reduce(operator.mul, x.shape) == max(x.shape)


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
    return functools.partial(functools.reduce, compose2)(args)


def composemap(*args):
    """Compose an arbitrary number of mapped functions.

    Parameters
    ----------
    args : tuple of callables

    Returns
    -------
    h : callable

    """
    maps = itertools.repeat(map, len(args))
    return functools.reduce(compose2, map(functools.partial, maps, args))


def trimmean(x, alpha, inclusive=(False, False), axis=None):
    """Compute the `alpha`-trimmed mean of an array `x`.

    Parameters
    ----------
    x : array_like
        The array on which to operate.

    alpha : float or int
        A number between 0 and 100, left inclusive indicating the percentage of
        values to cut from `x`.

    inclusive : tuple of bools, optional
        Whether to round (True, True) or truncate the values (False, False).
        Defaults to truncation.
       
    axis : int or None, optional
        The axis over which to operate.

    Returns
    -------
    m : Series
        The `alpha`-trimmed mean of `x` along axis `axis`.
    """
    assert 0 <= alpha < 100, 'alpha must be in the interval [0, 100)'
    assert len(inclusive) == 2, 'inclusive must have only 2 elements'

    if isinstance(x, numbers.Number) or (hasattr(x, 'size') and x.size == 1):
        return float(x)
        
    assert axis is None or 0 <= axis < x.ndim, \
        'axis must be None or less than x.ndim:{0}'.format(x.ndim)
    
    return pd.Series(scipy.stats.mstats.trimboth(x, proportiontocut=alpha / 100.0,
                                                 inclusive=inclusive,
                                                 axis=axis).mean(axis))
