"""
"""

# import numpy as np
from numpy cimport (uint8_t as uint8, ndarray, int64_t as int64, PyArray_EMPTY,
                    NPY_LONG, npy_intp, import_array)

from cython cimport wraparound, boundscheck
from cython.parallel cimport parallel, prange

import_array()

@wraparound(False)
@boundscheck(False)
cdef void _bin_data(ndarray[uint8, ndim=2, cast=True] a, ndarray[int64] bins,
                    ndarray[int64, ndim=2] out):
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
        int64 *out_data = NULL, *bin_data = NULL
        uint8* a_data = NULL    

    with nogil, parallel():
        out_data = <int64*> out.data
        bin_data = <int64*> bins.data
        a_data = <uint8*> a.data
        
        for k in prange(n, schedule='guided'):
            for i in xrange(m):
                v = 0
                for j in prange(bin_data[i], bin_data[i + 1], schedule='guided'):
                    v += a_data[j * n + k]
                out_data[i * n + k] = v


@wraparound(False)
@boundscheck(False)
cpdef bin_data(ndarray[uint8, ndim=2, cast=True] a, ndarray[int64] bins):
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
    cdef npy_intp dims[2]
    cdef ndarray[int64, ndim=2] out

    dims[0] = bins.shape[0] - 1
    dims[1] = a.shape[1]

    out = PyArray_EMPTY(2, dims, NPY_LONG, 0)

    _bin_data(a, bins, out)

    return out
