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
cdef void _bin_data(u1[:, :] a, u8[:] bins, u8[:, :] out):
    cdef ip i, j, k, m, n

    m = out.shape[0]
    n = out.shape[1]

    with nogil, parallel():
        for k in prange(n, schedule='guided'):
            for i in xrange(m):
                out[i, k] = 0

                for j in xrange(bins[i], bins[i + 1]):
                     out[i, k] += a[j, k]


@cython.wraparound(False)
@cython.boundscheck(False)
def bin_data(u1[:, :] a not None, u8[:] bins not None, u8[:, :] out not None):
    """Bin `a` (a boolean matrix) according to `bins`.

    For the :math:`k`th channel
        For the :math:`i`th sample
            Sum :math:`a_{jk}` where :math:`j \in` `bins`:math:`_{i},\ldots,`
            `bins`:math:`_{i+1}`.

    Parameters
    ----------
    a, bins, out : array_like
        The array whose values to count up in the bins given by `bins`. The
        result is stored in `out`.
    """
    _bin_data(a, bins, out)
