"""
"""

from numpy import asarray

from numpy cimport (uint8_t as uint8, ndarray, int64_t as int64, PyArray_EMPTY,
                    NPY_LONG, npy_intp, import_array)

cimport cython

import_array()

@cython.wraparound(False)
@cython.boundscheck(False)
cdef void _bin_data(uint8[:, :] a, int64[:] bins, int64[:, :] out) nogil:
    """Sum the counts of spikes in `a` in each of the bins.

    Parameters
    ----------
    a : array_like
    bins : array_like
    out : array_like
    """
    cdef:
        int64 i, j, k, v
        int64 m = out.shape[0], n = out.shape[1]
    
    for k in xrange(n):
        for i in xrange(m):
            v = 0

            for j in xrange(bins[i], bins[i + 1]):
                v += a[j, k]

            out[i, k] = v


@cython.wraparound(False)
@cython.boundscheck(False)
def bin_data(uint8[:, :] a not None, int64[:] bins not None):
    """Wrapper around bin_data._bin_data.

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
     # = np.empty((bins.shape[0] - 1, a.shape[1]),
                                               # dtype=np.int64)
    cdef:
        npy_intp dims[2]
        int64[:, :] out

    dims[0] = bins.shape[0] - 1
    dims[1] = a.shape[1]

    out = PyArray_EMPTY(2, dims, NPY_LONG, 0)

    _bin_data(a, bins, out)

    return asarray(out)
