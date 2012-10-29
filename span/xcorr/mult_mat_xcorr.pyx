from numpy cimport complex128_t as c16, npy_intp

from cython.parallel cimport prange, parallel

cimport cython


@cython.wraparound(False)
@cython.boundscheck(False)
cdef void _mult_mat_xcorr(c16[:, :] X, c16[:, :] Xc, c16[:, :] c, npy_intp n,
                          npy_intp nx):
    """Perform the necessary matrix-vector multiplication and fill the cross-
    correlation array. Slightly faster than pure Python.

    Parameters
    ----------
    X, Xc, c : array_like
    n, nx : npy_intp
    """
    cdef npy_intp i, j, k, r

    with nogil, parallel():
        for i in prange(n, schedule='guided'):
            for r, j in enumerate(xrange(i * n, (i + 1) * n)):
                for k in xrange(nx):
                    c[j, k] = X[i, k] * Xc[r, k]


@cython.wraparound(False)
@cython.boundscheck(False)
def mult_mat_xcorr(c16[:, :] X not None, c16[:, :] Xc not None,
                   c16[:, :] c not None, npy_intp n, npy_intp nx):
    assert n > 0, 'n must be greater than 0'
    assert nx > 0, 'nx must be greater than 0'
    _mult_mat_xcorr(X, Xc, c, n, nx)
