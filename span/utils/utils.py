
import numpy as np


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
    dims : ints

    Returns
    -------
    cur : array_like
    """
    if not dims:
        return ()
    dims = list(dims) # 100111_P3rat_site1
    n = dims.pop()
    cur = np.arange(n)[:, np.newaxis]
    while dims:
        d = dims.pop()
        cur = np.kron(np.ones((d, 1)), cur)
        front = np.arange(d).repeat(n)[:, np.newaxis]
        cur = np.hstack((front, cur))
        n *= d
    return cur


def dirsize(d='.'):
    """Recusively compute the size of a directory.

    Parameters
    ----------
    d : str, optional
        The directory of which to compute the size.

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
    ranges : list
    nelems : int, optional

    Returns
    -------
    n_linspaces : array_like
    """
    x = ndtuples(*nelems) + 1.0
    lbounds, ubounds = [], []
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
    """"Convert a string to a number

    Parameters
    ----------
    name : str
    base : int, optional

    Returns
    -------
    ret : int
    """
    return (base ** np.r_[:len(name)]).dot(np.fromiter(map(ord, name), dtype=int))
