import numpy as np
cimport numpy as np
cimport cython

ctypedef np.uint8_t uint8


@cython.wraparound(False)
@cython.boundscheck(False)
cdef void _bin(np.ndarray[uint8, ndim=2, cast=True] a, np.ndarray[long, ndim=1] bins,
               np.ndarray[long, ndim=2] out):
    """Sum the counts of spikes in `a` in each of the bins.

    Parameters
    ----------
    a : array_like
    bins : array_like
    out : array_like
    """
    cdef:
        long i, j, k, loc, bi, bj, n = a.shape[1], nbins = bins.size
        long* bin_data = <long*> bins.data
        long* out_data = <long*> out.data
        uint8* a_data = <uint8*> a.data

    for k in xrange(n):
        for i in xrange(nbins - 1):
            loc = i * n + k
            out_data[loc] = 0
            for j in xrange(bin_data[i], bin_data[i + 1]):
                 out_data[loc] += a_data[j * n + k]


@cython.wraparound(False)
@cython.boundscheck(False)
def bin(np.ndarray[uint8, ndim=2, cast=True] a not None,
        np.ndarray[long, ndim=1] bins not None,
        np.ndarray[long, ndim=2] out=None):
    """Wrapper around bin._bin.

    Parameters
    ----------
    a, bins : array_like
        The array whose values to count up in the bins given by bins.
    out : array, optional
        An optional output array. If given then it is destructively updated and
        also returned.

    Returns
    -------
    out : array_like
        The binned data from `a`.
    """
    cdef long m, n
    cdef tuple shape

    if out is None:
        m = bins.size - 1
        n = a.shape[1]
        shape = (m, n)
        out = np.empty(shape, dtype=np.long)
    _bin(a, bins, out)
    return out
