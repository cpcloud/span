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

from numpy cimport npy_intp as ip


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef int _read_tev_parallel(char* filename, ip nsamples, integral[:] fp_locs,
                    floating[:, :] spikes):
    cdef:
        ip i, j
        ip n = fp_locs.shape[0]

        size_t f_bytes = sizeof(floating)

        floating* chunk = NULL

        FILE* f = NULL

    with nogil, parallel():
        chunk = <floating*> malloc(f_bytes * nsamples)

        if not chunk:
            return -1

        f = fopen(filename, 'rb')

        if not f:
            free(chunk)
            chunk = NULL
            return -2

        for i in prange(n):
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


# @cython.boundscheck(False)
# @cython.wraparound(False)
# def _read_tev(char* filename, integral nsamples,
#               integral[:] fp_locs not None,
#               floating[:, ::1] spikes not None):
#     cdef ip r = __read_tev(filename, nsamples, fp_locs, spikes)

#     if r:
#         if r == -1:
#             raise MemoryError('Error when allocating chunk')
#         elif r == -2:
#             raise IOError('Unable to open file %s' % filename)
#         else:
#             raise IOError('Unable to read chunk')
