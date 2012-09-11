"""A collection of utility functions.
"""

import os
import types
import operator
import glob
import string

from itertools import imap as map, izip as zip, repeat

import numpy as np
import pandas as pd

from span.tdt.functional import compose, composemap


def get_names_and_threshes(f):
    """Read in an excel file and get the names of the recordings and the
    threshold.

    Parameters
    ----------
    f : str

    Returns
    -------
    dd : pandas.DataFrame
    """
    et = pd.io.parsers.ExcelFile(f)
    nms = et.parse(et.sheet_names[-1])
    dd = pd.DataFrame(nms['Block']).join(nms['Base Theshold']).drop_duplicates()
    dd.columns = pd.Index(['block', 'threshold'])
    return dd


def cast(a, dtype=None, order='K', casting='unsafe', subok=True, copy=False):
    """Cast `a` to dtype `dtype`.

    Attempt to cast `a` to a different dtype `dtype` without copying

    Parameters
    ----------
    a : array_like
    dtype : dtype, optional
    order : str, optional
    casting : str, optional
    subok : bool, optional
    copy : bool, optional

    Raises
    ------
    AssertionError

    Returns
    -------
    r : array_like
    """
    assert hasattr(a, 'dtype'), 'first argument has no "dtype" attribute'
    if dtype is None:
        dtype = a.dtype
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
    assert all(map(isinstance, dims, repeat(int, len(dims)))), \
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
                           (np.float64, np.float64))
    return lbounds + (x - 1) / (b - 1) * (ubounds - lbounds)


def nans(size, dtype=float):
    """Create an array of NaNs

    Parameters
    ----------
    size : tuple
    dtype : descriptor, optional

    Returns
    -------
    a : array_like
    """
    a = np.zeros(size, dtype=dtype)
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
        ax = pylab.gca()
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
    base_vec = base ** np.r_[:slen]
    nletters = len(string.ascii_letters)
    x = pd.Series(dict(zip(string.ascii_letters, map(ord,
                                                     string.ascii_letters))))
    xad = x[ndtuples(*tuple(repeat(nletters, slen)))] * base_vec
    w = xad[np.where(xad.sum(1) == num)].squeeze()
    return ''.join(chr(c) for c in w / base_vec)


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


def flatten(data):
    """Flatten a DataFrame.

    Parameters
    ----------
    data : pandas.DataFrame

    Returns
    -------
    flattened : pandas.DataFrame or pandas.Series
    """
    try:
        # FIX: `stack` method is potentially very fragile
        return data.stack().reset_index(drop=True)
    except MemoryError:
        raise MemoryError('out of memory while trying to flatten')


def bin_data(data, bins):
    """Put data in bins.

    Parameters
    ----------
    data : pandas.DataFrame
    """
    nchannels = data.columns.size
    counts = pd.DataFrame(np.empty((bins.size - 1, nchannels)))
    zbins = list(zip(bins[:-1], bins[1:]))
    for column, dcolumn in data.iterkv():
        counts[column] = pd.Series([dcolumn.ix[bi:bj].sum()
                                    for bi, bj in zbins], name=column)
    return counts


# TODO: Make this function less ugly!
def summary(group, func):
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
    func_is_valid = any(map(isinstance, (func, func),
                                  (str, types.FunctionType)))
    assert func_is_valid, ("'func' must be a string or function: "
                           "type(func) == {0}".format(type(func)))

    # if `func` is a string
    if hasattr(group, func):
        getter = operator.attrgetter(func)
        chan_func = getter(group)
        chan_func_t = getter(chan_func().T)
        return chan_func_t()

    # else if it's a function and has the attribute `__name__`
    elif hasattr(func, '__name__') and \
            hasattr(group, func.__name__):
        # recurse
        return summary(func.__name__)

    # else if it's just a regular ole' function
    else:
        f = lambda x: func(SpikeDataFrame.flatten(x))

    # apply the function to the channel group
    return group.apply(f)


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
    return np.modf(np.asanyarray(x), frac).any()


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
    size_getter = operater.attrgetter('size')
    sizes = np.fromiter(map(size_getter, arrays), int)
    lsize = sizes.max()

    ret = []
    for array, size in zip(arrays, sizes):
        size_diff = lsize - size
        ret.append(zeropad(array, size_diff))

    ret.append(lsize)
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
    return np.issubdtype(x.dtype, complex)
    

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
    ndims_getter = operator.attrgetter('ndim')
    asserter = compose(ndims_getter, np.squeeze, np.array)
    assert all(map(lambda x: asserter(x) == 1, arrays)), 'all input arrays must be 1D'
    
    r = np.fft.irfft, np.fft.rfft
    if any(composemap(iscomplex, np.squeeze, np.asanyarray)(arrays)):
        r = np.fft.ifft, np.fft.fft
    return r


def spike_window(ms, fs, const=1e3):
    """Perform a transparent conversion from time to samples.

    Parameters
    ----------
    ms : int
    fs : float
    const : float, optional

    Returns
    -------
    Conversion of milliseconds to samples
    """
    return cast(np.floor(ms / const * fs), dtype=np.int32)


def electrode_distance(fromij, toij, between_shank=125, within_shank=100):
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
    fromi, fromj = fromij
    toi, toj = toij

    col_diff = (toj - fromj) * between_shank
    row_diff = (toi - fromi) * within_shank
    return np.sqrt(col_diff ** 2 + row_diff ** 2)


def distance_map(n=4):
    """Create an electrode distance map.

    Parameters
    ----------
    n : int

    Returns
    -------
    ret : pandas.Series
    """
    dists = np.zeros(n ** 4)
    t = 0
    rangen = np.arange(n)
    a, b, c, d = ndtuples(n, n, n, n).T
    for i in rangen:
        for j in rangen:
            for k in rangen:
                for l in rangen:
                    dists[t] = electrode_distance((i, j), (k, l))
                    t += 1
    index = pd.MultiIndex.from_arrays((a, b, c, d))
    return pd.Series(dists, index=index, name='distance')