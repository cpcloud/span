# bin_data.pyx ---

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

cimport cython

from numpy cimport uint8_t as u1, uint64_t as u8, npy_intp as ip

from cython.parallel cimport prange, parallel


@cython.wraparound(False)
@cython.boundscheck(False)
cdef void __bin_data(u1[:, :] a, u8[:] bins, u8[:, :] out) nogil:
    cdef ip i, j, k, m, n

    with parallel():
        m = out.shape[0]
        n = out.shape[1]

        for k in prange(n, schedule='guided'):
            for i in xrange(m):
                out[i, k] = 0

                for j in xrange(bins[i], bins[i + 1]):
                     out[i, k] += a[j, k]


@cython.wraparound(False)
@cython.boundscheck(False)
cpdef _bin_data(u1[:, :] a, u8[:] bins, u8[:, :] out):
    __bin_data(a, bins, out)
