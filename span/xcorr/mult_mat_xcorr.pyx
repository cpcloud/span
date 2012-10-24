from numpy cimport complex128_t as c16, int64_t as i8

from cython.parallel cimport prange, parallel

cimport cython


@cython.wraparound(False)
@cython.boundscheck(False)
cdef void _mult_mat_xcorr(c16[:, :] X, c16[:, :] Xc, c16[:, :] c, i8 n,
                          i8 nx) nogil:
    """Perform the necessary matrix-vector multiplication and fill the cross-
    correlation array. Slightly faster than pure Python.

    Parameters
    ----------
    X, Xc, c : array_like
    n, nx : i8
    """
    cdef i8 i, j, k, r

    with parallel():
        for i in prange(n, schedule='guided'):
            for r, j in enumerate(xrange(i * n, (i + 1) * n)):
                for k in xrange(nx):
                    c[j, k] = X[i, k] * Xc[r, k]


@cython.wraparound(False)
@cython.boundscheck(False)
def mult_mat_xcorr(c16[:, :] X not None, c16[:, :] Xc not None,
                   c16[:, :] c not None, i8 n, i8 nx):
    assert n > 0, 'n must be greater than 0'
    assert nx > 0, 'nx must be greater than 0'
    _mult_mat_xcorr(X, Xc, c, n, nx)
