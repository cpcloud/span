from numpy cimport complex128_t as complex128, int64_t as int64, ndarray

from cython.parallel cimport parallel, prange
from cython cimport boundscheck, wraparound


@wraparound(False)
@boundscheck(False)
cpdef mult_mat_xcorr(ndarray[complex128, ndim=2] X,
                     ndarray[complex128, ndim=2] Xc,
                     ndarray[complex128, ndim=2] c, int64 n, int64 nx):
    """Perform the necessary matrix-vector multiplication and fill the cross-
    correlation array. Slightly faster than pure Python.

    Parameters
    ----------
    X, Xc, c : array
    n : int64
    """
    cdef:
        int64 i, j, k, r
        complex128 *c_data = NULL, *X_data = NULL, *Xc_data = NULL

    with nogil, parallel():
        c_data = <complex128*> c.data
        X_data = <complex128*> X.data
        Xc_data = <complex128*> Xc.data

        for i in prange(n, schedule='guided'):
            for r, j in enumerate(xrange(i * n, (i + 1) * n)):
                for k in prange(nx, schedule='guided'):
                    c_data[j * nx + k] = (X_data[i * nx + k] *
                                          Xc_data[r * nx + k])
