"""A collection of utility functions."""

from future_builtins import map, zip

import os
import types
import operator
import glob
import string
import itertools
import functools

from functools import reduce

import numpy as np
import pandas as pd

try:
    from pylab import detrend_none, detrend_mean, detrend_linear, gca
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


    def detrend_mean_inplace(x):
        """Center x in place.

        Parameters
        ----------
        x : array_like

        See Also
        --------
        detrend_mean
        """
        x -= x.mean()


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

    gca = NotImplemented


def get_names_and_threshes(f):
    """Read in an excel file and get the names of the recordings and their
    respective thresholds.

    Parameters
    ----------
    f : str
        The name of the Excel file to read.

    Returns
    -------
    dd : pandas.DataFrame
        A Pandas DataFrame corresponding to valid recordings.
    """
    et = pd.io.parsers.ExcelFile(f)
    nms = et.parse(et.sheet_names[-1])
    dd = pd.DataFrame(nms['Block']).join(nms['Base Theshold']).drop_duplicates()
    dd.columns = pd.Index(['block', 'threshold'])
    return dd


def cast(a, dtype, order='K', casting='unsafe', subok=True, copy=False):
    """Cast `a` to dtype `dtype`.

    Attempt to cast `a` to a different dtype `dtype` without copying.

    Parameters
    ----------
    a : array_like
        The array to cast.

    dtype : numpy.dtype
        The dtype to cast the input array to.

    order : str, optional
        The order in memory of the array.

    casting : str, optional
        Rules to use for casting.

    subok : bool, optional
        Whether or not to pass subclasses through.

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
        r = a.astype(dtype, order=order, casting=casting, subok=subok, copy=copy)
    except TypeError:
        r = a.astype(dtype)
    return r


def ndtuples(*dims):
    """Create a bunch of tuples with `dims` dimensions.

    Parameters
    ----------
    dims : tuple of int

    Returns
    -------
    cur : array_like
    """
    assert dims, 'no arguments given'
    assert all(map(isinstance, dims, itertools.repeat(int, len(dims)))), \
        'all arguments must be integers'
    dims = list(dims)
    n = dims.pop()
    cur = np.arange(n)[:, np.newaxis]
    while dims:
        d = dims.pop()
        cur = np.kron(np.ones((d, 1)), cur)
        front = np.arange(d).repeat(n)[:, np.newaxis]
        cur = np.hstack((front, cur))
        n *= d
    return cast(cur, int)


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
    """Create `n` linspaces between the ranges of `ranges` with `nelems` elements.

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
    lbounds, ubounds = map(np.fromiter, zip(*((r[0], r[1]) for r in ranges)),
                           (float, float))
    return lbounds + (x - 1.0) / (b - 1.0) * (ubounds - lbounds)


def nans(shape, dtype=float):
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
    a = np.empty(shape, dtype=dtype)
    a.fill(np.nan)
    return a


def nans_like(a, dtype=None, order='K', subok=True):
    """Create an array of nans with dtype, order, and class like `a`.

    Parameters
    ----------
    a : array_like
    dtype : np.dtype, optional
    order : str, optional
    subok : bool, optional

    Returns
    -------
    res : array_like
    """
    res = np.empty_like(a, dtype=dtype, order=order, subok=subok)
    np.copyto(res, np.nan, casting='unsafe')
    return res


def remove_legend(ax=None):
    """Remove legend for ax or the current axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
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
    base : int, optional
    slen : int, optional

    Returns
    -------
    ret : str
    """
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


# TODO: Make this function less ugly!
def summary(group, func, axis, skipna, level):
    """Perform a summary computation on a group.

    Parameters
    ----------
    group : pandas.Grouper
    func : str or callable

    Returns
    -------
    sumry : pandas.DataFrame or pandas.Series
    """
    # check to make sure that `func` is a string or function
    func_is_valid = any(map(isinstance, (func, func), (str, types.FunctionType)))
    assert func_is_valid, ("'func' must be a string or function: "
                           "type(func) == {0}".format(type(func)))

    if hasattr(group, func):
        getter = operator.attrgetter(func)
        chan_func = getter(group)
        chan_func_t = getter(chan_func().T)
        return chan_func_t(axis=axis, skipna=skipna, level=level)
    elif hasattr(func, '__name__') and hasattr(group, func.__name__):
        return summary(group, func.__name__, axis, skipna, level)
    else:
        f = lambda x: func(x, skipna=skipna)

    return group.apply(f, axis=axis)


def nextpow2(n):
    """Return the next power of 2 of a number.

    Parameters
    ----------
    n : array_like

    Returns
    -------
    ret : array_like
    """
    return np.ceil(np.log2(np.absolute(np.asanyarray(n))))


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
    return frac.any()


def zeropad(x, s=0):
    """Pad an array, `x`, with `s` zeros.

    Parameters
    ----------
    x : array_like
    s : int

    Raises
    ------
    AssertionError

    Returns
    -------
    ret : `x` padded with `s` zeros.
    """
    assert not fractional(s), \
        's must be an integer or floating point number with no fractional part'
    assert s >= 0, 's cannot be negative'
    if not s:
        return x
    return np.pad(x, s, mode='constant', constant_values=(0,))


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

        if xsize > ysize:
            y = zeropad(y, size_diff)
        else:
            x = zeropad(x, size_diff)

    return x, y, lsize


def pad_larger(*arrays):
    """Pad the smallest of `n` arrays.

    Parameters
    ----------
    arrays : tuple of array_like

    Returns
    -------
    ret : list
        List of zero padded arrays
    """
    if len(arrays) == 2:
        return pad_larger2(*arrays)

    sizes = np.fromiter(map(operator.attrgetter('size'), arrays), int)
    lsize = sizes.max()

    ret = ()
    for array, size in zip(arrays, sizes):
        size_diff = lsize - size
        ret += zeropad(array, size_diff),
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
        return np.issubdtype(x.dtype, complex)
    except AttributeError:
        return any(map(np.issubdtype, x.dtypes, itertools.repeat(complex, x.dtypes.size)))


def hascomplex(x):
    """Check whether an array has all complex entries.
    """
    return iscomplex(x) and not np.logical_not(x.imag).all()


def get_fft_funcs(*arrays):
    """Get the correct fft functions for the input type.

    Parameters
    ----------
    arrays : tuple
        Arrays to be checked for complex dtype.

    Returns
    -------
    r : 2-tuple of callables
        The fft and ifft appropriate for the dtype of input.
    """
    r = np.fft.irfft, np.fft.rfft
    arecomplex = functools.partial(map, iscomplex)#composemap(iscomplex)
    if any(arecomplex(arrays)):
        r = np.fft.ifft, np.fft.fft
    return r


def electrode_distance_old(fromij, toij, between_shank=125, within_shank=100):
    fromi, fromj = fromij
    toi, toj = toij

    col_diff = (toj - fromj) * between_shank
    row_diff = (toi - fromi) * within_shank
    return np.sqrt(col_diff ** 2 + row_diff ** 2)


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
    assert np.logical_not(dist[np.diag(np.r_[:s].reshape((si, si)))]).all(), \
        'self distance is not 0'
    return dist


def distance_map(nshanks=4, electrodes_per_shank=4):
    """Create an electrode distance map.

    Parameters
    ----------
    n : int

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
    assert hasattr(x, 'shape'), 'x has no shape attribute'
    return reduce(operator.mul, x.shape) == max(x.shape)
