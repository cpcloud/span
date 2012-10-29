"""
"""

from numpy cimport (uint8_t as u1, ndarray, PyArray_EMPTY, NPY_ULONG, npy_intp,
                    import_array, uint64_t as u8)

from cython.parallel cimport prange, parallel

cimport cython

import_array()


@cython.wraparound(False)
@cython.boundscheck(False)
cdef void _bin_data(u1[:, :] a, u8[:] bins, u8[:, :] out):
    """Sum the counts of spikes in `a` in each of the bins.

    Parameters
    ----------
    a : array_like
    bins : array_like
    out : array_like
    """
    cdef npy_intp i, j, k
    cdef npy_intp m = out.shape[0], n = out.shape[1]

    with nogil, parallel():
        for k in prange(n):
            for i in xrange(m):
                out[i, k] = 0

                for j in xrange(bins[i], bins[i + 1]):
                     out[i, k] += a[j, k]


@cython.wraparound(False)
@cython.boundscheck(False)
def bin_data(u1[:, :] a not None, u8[:] bins not None):
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
    cdef npy_intp dims[2]

    dims[0] = bins.shape[0] - 1
    dims[1] = a.shape[1]

    # ndim, size of dims, type, c if 0 else fortran order
    cdef ndarray[u8, ndim=2] out = PyArray_EMPTY(2, dims, NPY_ULONG, 0)

    _bin_data(a, bins, out)

    return out
