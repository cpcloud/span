from numpy cimport (uint8_t as u1, ndarray, PyArray_EMPTY, NPY_ULONG,
                    npy_intp as i8, import_array, uint64_t as u8)

from cython.parallel cimport prange, parallel

cimport cython

import_array()


@cython.wraparound(False)
@cython.boundscheck(False)
cdef void _bin_data(u1[:, :] a, u8[:] bins, u8[:, :] out) nogil:
    cdef i8 i, j, k, m, n

    m = out.shape[0], n = out.shape[1]

    with parallel():
        for k in prange(n, schedule='guided'):
            for i in xrange(m):
                out[i, k] = 0

                for j in xrange(bins[i], bins[i + 1]):
                     out[i, k] += a[j, k]


@cython.wraparound(False)
@cython.boundscheck(False)
def bin_data(u1[:, :] a not None, u8[:] bins not None):
    """Bin `a` (a boolean matrix) according to `bins`.

    For the :math:`k`th channel
        For the :math:`i`th sample
            Sum :math:`a_{jk}` where :math:`j \in` `bins`:math:`_{i},\ldots,`
            `bins`:math:`_{i+1}`.


    Parameters
    ----------
    a, bins : array_like
        The array whose values to count up in the bins given by `bins`.

    Returns
    -------
    out : array_like
        The binned data from `a`.
    """
    cdef:
        i8 dims[2]
        ndarray[u8, ndim=2] out

    dims[0] = bins.shape[0] - 1
    dims[1] = a.shape[1]

    # ndim, size of dims, type, c if 0 else fortran order
    out = PyArray_EMPTY(2, dims, NPY_ULONG, 0)

    _bin_data(a, bins, out)

    return out
