cimport numpy as np
cimport cython

from cython.parallel cimport parallel, prange


@cython.boundscheck(False)
@cython.wraparound(False)
def mult_mat_xcorr(np.ndarray[complex, ndim=2] X, np.ndarray[complex, ndim=2] Xc,
                   np.ndarray[complex, ndim=2] c, long n, long nx):
    """Perform the necessary matrix-vector multiplication and fill the cross-
    correlation array.

    Parameters
    ----------
    X, Xc, c : array
    n ; long

    Raises
    ------
    ValueError
        If data pointers are null. We don't want any mysterious segfaulting!
    """
    cdef:
        long i, j, k, r
        complex *c_data = NULL, *X_data = NULL, *Xc_data = NULL

    with nogil, parallel():
        c_data = <complex*> c.data
        X_data = <complex*> X.data
        Xc_data = <complex*> Xc.data

        if c_data and X_data and Xc_data:
            for i in prange(n, schedule='guided'):
                for r, j in enumerate(xrange(i * n, (i + 1) * n)):
                    for k in prange(nx, schedule='guided'):
                        c_data[j * nx + k] = (X_data[i * nx + k] *
                                              Xc_data[r * nx + k])
        else:
            with gil:
                raise ValueError('All data pointers are NULL')
