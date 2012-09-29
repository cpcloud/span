"""A collection of utility functions."""

from future_builtins import map, zip

import os
import operator
import glob
import string
import itertools
import functools

import numpy as np
import pandas as pd

from span.utils.functional import compose
# from span.tdt import Indexer

try:
    from pylab import (
    detrend_linear, detrend_mean, detrend_none, figure, gca, subplots)
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

    gca = NotImplemented
    figure = NotImplemented
    subplots = NotImplemented


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
    """Create a bunch of tuples with `dims` dimensions.

    Parameters
    ----------
    dims : tuple of int

    Returns
    -------
    cur : array_like
    """
    assert dims, 'no arguments given'
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


def cartesian(arrays, out=None):
    """Cartesian product of arrays.

    Parameters
    ----------
    arrays : tuple of array_like
    out : array_like

    Returns
    -------
    out : array_like
    """
    arrays = tuple(map(np.asarray, arrays))
    dtype = np.object_
    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.empty([n, len(arrays)], dtype=dtype)
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
    assert all(map(isinstance, arrays, itertools.repeat(np.ndarray,
                                                        len(arrays)))), \
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
        return any(map(np.issubdtype, x.dtypes, itertools.repeat(np.complexfloating,
                                                                 x.dtypes.size)))


def hascomplex(x):
    """Check whether an array has all complex entries.

    Parameters
    ----------
    x : array_like

    Returns
    -------
    r : bool
    """
    return iscomplex(x) and not np.logical_not(x.imag).all()


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


# def plot_xcorr(xc, figsize=(40, 25), dpi=100, titlesize=4, labelsize=3,
#                sharex=True, sharey=True):
#     ###########################
#     # TODO: rewrite this crap #
#     ###########################

#     # create as many possible things as we can before
#     # instantiating any matplotlib figures and friends

#     # get the channel indexer
#     elec_map = Indexer.channel

#     # number of channels
#     nchannels = elec_map.size

#     # the channel index labels
#     left, right, _, _ = xc.index.labels

#     # indices of a lower triangular nchannels by nchannels array
#     lower_inds = np.tril_indices(nchannels)

#     # flatted and linearly indexed
#     flat_lower_inds = np.ravel_multi_index(np.vstack(lower_inds),
#                                            (nchannels, nchannels))

#     # make a list of strings for titles
#     title_strings = cast(np.vstack((left + 1, right + 1)), str).T.tolist()
#     title_strings = np.asanyarray(list(map(' vs. '.join, title_strings)))

#     # get only the ones we want
#     title_strings = title_strings[flat_lower_inds]

#     # create the subplots with linked axes
#     fig, axs = subplots(nchannels, nchannels, sharex=sharex, sharey=sharey,
#                         figsize=figsize, dpi=dpi)

#     # get the axes objects that we want to show
#     axs_to_show = axs.flat[flat_lower_inds]

#     # set the title on the axes objects that we want to see
#     titler = lambda ax, s, fs: ax.set_title(s, fontsize=fs)
#     sizes = itertools.repeat(titlesize, axs_to_show.size)
#     list(map(titler, axs_to_show.flat, title_strings, sizes))

#     # hide the ones we don't want
#     upper_inds = np.triu_indices(nchannels, 1)
#     flat_upper_inds = np.ravel_multi_index(np.vstack(upper_inds),
#                                            (nchannels, nchannels))
#     axs_to_hide = axs.flat[flat_upper_inds]
#     list(map(lambda ax: map(lambda tax: tax.set_visible(False), (ax.xaxis, ax.yaxis)),
#              axs_to_hide))
#     list(map(lambda ax: ax.set_frame_on(False), axs_to_hide))
#     list(map(remove_legend, axs.flat))

#     min_value = xc.min().min()
#     for indi, i in enumerate(elec_map):
#         for indj, j in enumerate(elec_map):
#             if indi >= indj:
#                 ax = axs[indi, indj]
#                 ax.tick_params(labelsize=labelsize, left=True,
#                                right=False, top=False, bottom=True,
#                                direction='out')
#                 xcij = xc.ix[i, j].T
#                 ax.vlines(xcij.index, min_value, xcij)
#     fig.tight_layout()
#     return fig, axs


def blob(x, y, area, color, ax):
    """Fill a square of area `area` with color `color` on axis `ax`.

    Parameters
    ----------
    x, y, area : number
    color : str
    ax : matplotlib.axes.Axes
    """
    hs = np.sqrt(area) / 2.0
    xcorn = np.array([x - hs, x + hs, x + hs, x - hs])
    ycorn = np.array([y - hs, y - hs, y + hs, y + hs])
    ax.fill(xcorn, ycorn, color, edgecolor=color)


def hinton(w, max_weight, ax):
    """Plot a Hinton diagram.

    Parameters
    ----------
    w : array_like
    max_weight : array_like
    ax : matplotlib.axes.Axes
    """
    w = np.asanyarray(w)
    if max_weight is None:
        f = compose(np.ceil, np.log2, np.max, np.abs)
        max_weight = 2 ** f(w)
        # max_weight = 2 ** np.ceil(np.log2(np.max(np.abs(w))))
        # max_weight = 2 ** np.ceil(np.log(np.max(np.abs(w))) / np.log(2))

    if ax is None:
        fig = figure()
        ax = fig.add_subplot(111)

    height, width = w.shape

    ax.fill(np.array([0, width, width, 0]),
            np.array([0, 0, height, height]), 'gray')
    ax.axis('off')
    ax.axis('equal')

    colors = {1: 'white', -1: 'black'}

    for x in xrange(width):
        for y in xrange(height):
            xx = x + 1
            yy = y + 1
            wyx = w[y, x]
            s = np.sign(wyx)
            blob(xx - 0.5, height - yy + 0.5, min(1, s * wyx / max_weight),
                 colors[s], ax)
