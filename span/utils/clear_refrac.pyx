# clear_refrac.pyx ---

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
from numpy cimport (npy_intp as ip, uint8_t as u1, uint16_t as u2,
                    uint32_t as u4, uint64_t as u8, int8_t as i1,
                    int16_t as i2, int32_t as i4, int64_t as i8)

ctypedef fused integral:
    u1
    u2
    u4
    u8

    i1
    i2
    i4
    i8

    size_t
    Py_ssize_t
    bint

    ip


@cython.wraparound(False)
@cython.boundscheck(False)
cdef int clear_refrac_impl(integral[:, :] a, ip window) nogil except -1:
    cdef ip channel, i, sample, sp1, nsamples, nchannels

    nsamples = a.shape[0]
    nchannels = a.shape[1]

    with nogil:
        for channel in range(nchannels):
            sample = 0

            while sample + window < nsamples:

                if a[sample, channel]:
                    sp1 = sample + 1
                    a[sp1:sp1 + window, channel] = False
                    sample += window

                sample += 1
    return 0


@cython.wraparound(False)
@cython.boundscheck(False)
cpdef int _clear_refrac(integral[:, :] a, ip window) nogil except -1:
    return clear_refrac_impl(a, window)
