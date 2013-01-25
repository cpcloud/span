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
from cython cimport floating, integral
from cython.parallel cimport prange, parallel

from numpy cimport npy_intp as ip, ndarray


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef int _read_tev_serial(char* filename, ip nsamples, integral[:] fp_locs,
                           floating[:, :] spikes):
    cdef:
        ip i, j
        ip n = fp_locs.shape[0]

        size_t f_bytes = sizeof(floating)

        floating* chunk = NULL

        FILE* f = NULL

    chunk = <floating*> malloc(f_bytes * nsamples)

    if not chunk:
        return -1

    f = fopen(filename, 'rb')

    if not f:
        free(chunk)
        chunk = NULL
        return -2

    for i in xrange(n):
        # go to the ith file pointer location
        fseek(f, fp_locs[i], SEEK_SET)

        # read floating_bytes * nsamples bytes into chunk_data
        if not fread(chunk, f_bytes, nsamples, f):
            return -3

        # assign the chunk data to the spikes array
        for j in xrange(nsamples):
            spikes[i, j] = chunk[j]

    # get rid of the chunk data
    free(chunk)
    chunk = NULL

    fclose(f)
    f = NULL

    return 0


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef int _read_tev_parallel(char* filename, integral[:, :] grouped,
                             ip blocksize,
                             floating[:, :] spikes) nogil except -1:

    cdef:
        ip channel, block, k, byte
        ip nchannels = grouped.shape[1], nblocks = grouped.shape[0]

        size_t f_bytes = sizeof(floating)
        floating* chunk = NULL
        FILE* f = NULL

    with parallel():
        chunk = <floating*> malloc(f_bytes * blocksize)

        if not chunk:
            return -1

        f = fopen(filename, "rb")

        if not f:
            free(chunk)
            chunk = NULL
            return -1

        for channel in prange(nchannels, schedule='static'):
            for block in xrange(nblocks):

                fseek(f, grouped[block, channel], SEEK_SET)
                fread(chunk, f_bytes, blocksize, f)

                for k, byte in enumerate(xrange(block * blocksize,
                                                (block + 1) * blocksize)):
                    spikes[byte, channel] = chunk[k]


        free(chunk)
        chunk = NULL

        fclose(f)
        f = NULL

        return 0
