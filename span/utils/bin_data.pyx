"""
"""

from numpy cimport (uint8_t as u1, ndarray, int64_t as i8, PyArray_EMPTY,
                    NPY_LONG, npy_intp, import_array)

from cython.parallel cimport prange, parallel

cimport cython

import_array()


@cython.wraparound(False)
@cython.boundscheck(False)
cdef void _bin_data(u1[:, :] a, i8[:] bins, i8[:, :] out) nogil:
    """Sum the counts of spikes in `a` in each of the bins.

    Parameters
    ----------
    a : array_like
    bins : array_like
    out : array_like
    """
    cdef i8 i, j, k, m = out.shape[0], n = out.shape[1]

    with parallel():
        for k in prange(n, schedule='guided'):
            for i in xrange(m):
                out[i, k] = 0

                for j in xrange(bins[i], bins[i + 1]):
                     out[i, k] += a[j, k]


@cython.wraparound(False)
@cython.boundscheck(False)
def bin_data(u1[:, :] a not None, i8[:] bins not None):
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
    cdef:
        npy_intp dims[2]
        ndarray[dtype=u1, ndim=2] out

    dims[0] = bins.shape[0] - 1
    dims[1] = a.shape[1]

    # ndim, size of dims, type, c if 0 else fortran order
    out = PyArray_EMPTY(2, dims, NPY_LONG, 0)

    _bin_data(a, bins, out)

    return out
