# mult_mat_xcorr.pyx ---

# Copyright (C) 2012 Copyright (C) 2012 Phillip Cloud <cpcloud@gmail.com>

# Author: Phillip Cloud <cpcloud@gmail.com>

# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.


from numpy cimport complex128_t as c16, npy_intp as i8

from cython.parallel cimport prange, parallel

cimport cython


@cython.wraparound(False)
@cython.boundscheck(False)
cdef void _mult_mat_xcorr(c16[:, :] X, c16[:, :] Xc, c16[:, :] c, i8 n,
                          i8 nx) nogil:

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
    """Perform the necessary matrix-vector multiplication and fill the cross-
    correlation array. Slightly faster than pure Python.

    Parameters
    ----------
    X, Xc, c : c16[:, :]
    n, nx : i8

    Raises
    ------
    AssertionError
       If n <= 0 or nx <= 0
    """
    assert n > 0, 'n must be greater than 0'
    assert nx > 0, 'nx must be greater than 0'

    _mult_mat_xcorr(X, Xc, c, n, nx)
