# read_tev.pyx ---

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

from libc.stdio cimport fopen, fclose, fread, fseek, SEEK_SET, FILE
from libc.stdlib cimport malloc, free

cimport cython
from cython cimport floating, integral, view
from cython.parallel cimport prange, parallel

from numpy cimport npy_intp as ip, ndarray, int64_t as i8, float32_t as f4

cdef extern from "sys/stat.h":
    struct stat:
        long st_size

    int get_stat "stat" (char*, stat*)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef int _read_tev_serial(char* filename, integral[:, :] grouped, ip blocksize,
                           floating[:, :] spikes) nogil except -1:

    cdef:
        ip c, b, k, byte, low, high
        ip nchannels = grouped.shape[1], nblocks = grouped.shape[0]

        size_t f_bytes = sizeof(floating)
        floating* chunk = NULL
        FILE* f = NULL

    chunk = <floating*> malloc(f_bytes * blocksize)

    if not chunk:
        return -1

    f = fopen(filename, "rb")

    if not f:
        free(chunk)
        return -1

    for c in xrange(nchannels):
        for b in xrange(nblocks):

            fseek(f, grouped[b, c], SEEK_SET)

            if not fread(chunk, f_bytes, blocksize, f):
                free(chunk)
                fclose(f)
                return -1

            low = b * blocksize
            high = (b + 1) * blocksize

            for k, byte in enumerate(xrange(low, high)):
                spikes[byte, c] = chunk[k]

    free(chunk)
    fclose(f)
    return 0


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef int _read_tev_parallel(char* filename, integral[:, :] grouped, ip blocksize,
                             floating[:, :] spikes) nogil except -1:

    cdef:
        ip c, b, k, byte, low, high
        ip nchannels = grouped.shape[1], nblocks = grouped.shape[0]

        size_t f_bytes = sizeof(floating)
        floating* chunk = NULL
        FILE* f = NULL

    with parallel():
        chunk = <floating*> malloc(f_bytes * blocksize)

        if not chunk:
            return -1

        f = fopen(filename, 'rb')

        if not f:
            free(chunk)
            return -1

        for c in prange(nchannels, schedule='guided'):
            for b in range(nblocks):

                fseek(f, grouped[b, c], SEEK_SET)

                if not fread(chunk, f_bytes, blocksize, f):
                    free(chunk)
                    fclose(f)
                    return -1

                low = b * blocksize
                high = (b + 1) * blocksize

                for k, byte in enumerate(range(low, high)):
                    spikes[byte, c] = chunk[k]

        free(chunk)
        fclose(f)
    return 0
