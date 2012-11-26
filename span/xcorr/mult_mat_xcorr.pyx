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


from numpy cimport (complex64_t as c8, complex128_t as c16, npy_intp as ip,
                    int64_t as i8, float32_t as f4, float64_t as f8)

from cython.parallel cimport prange, parallel

cimport cython

ctypedef fused floating:
    f4
    f8

    c8
    c16


@cython.wraparound(False)
@cython.boundscheck(False)
cdef void _mult_mat_xcorr(floating[:, :] X, floating[:, :] Xc,
                          floating[:, :] c, ip n, ip nx) nogil:

    cdef ip i, j, k, r

    with parallel():
        for i in prange(n, schedule='guided'):
            for r, j in enumerate(xrange(i * n, (i + 1) * n)):
                for k in xrange(nx):
                    c[j, k] = X[i, k] * Xc[r, k]


@cython.wraparound(False)
@cython.boundscheck(False)
def mult_mat_xcorr(floating[:, :] X not None, floating[:, :] Xc not None,
                   floating[:, :] c not None, ip n, ip nx):
    """Perform the necessary matrix-vector multiplication and fill the cross-
    correlation array. Slightly faster than pure Python.

    Parameters
    ----------
    X, Xc, c : c16[:, :]
    n, nx : ip

    Raises
    ------
    AssertionError
       If n <= 0 or nx <= 0
    """
    assert n > 0, 'n must be greater than 0'
    assert nx > 0, 'nx must be greater than 0'

    _mult_mat_xcorr(X, Xc, c, n, nx)
