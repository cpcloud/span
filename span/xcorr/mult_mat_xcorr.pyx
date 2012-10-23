from numpy cimport complex128_t as complex128, int64_t as int64

from cython.parallel cimport parallel, prange
cimport cython


@cython.wraparound(False)
@cython.boundscheck(False)
cdef void _mult_mat_xcorr(complex128[:, :] X, complex128[:, :] Xc,
                          complex128[:, :] c, int64 n, int64 nx):
    """Perform the necessary matrix-vector multiplication and fill the cross-
    correlation array. Slightly faster than pure Python.

    Parameters
    ----------
    X, Xc, c : array_like
    n, nx : int64
    """
    cdef int64 i, j, k, r

    with nogil, parallel():
        for i in prange(n, schedule='guided'):
            for r, j in enumerate(xrange(i * n, (i + 1) * n)):
                for k in prange(nx, schedule='guided'):
                    c[j, k] = X[i, k] * Xc[r, k]


@cython.wraparound(False)
@cython.boundscheck(False)
def mult_mat_xcorr(complex128[:, :] X not None, complex128[:, :] Xc not None,
                   complex128[:, :] c not None, int64 n, int64 nx):
    assert n > 0, 'n must be greater than 0'
    assert nx > 0, 'nx must be greater than 0'
    _mult_mat_xcorr(X, Xc, c, n, nx)
