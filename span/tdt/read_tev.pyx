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

ctypedef Py_ssize_t ip


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef int _read_tev_raw(const char* filename, integral[:] fp_locs,
                        ip block_size, floating[:, :] spikes) nogil except -1:
    cdef:
        ip n = fp_locs.shape[0], f_bytes = sizeof(floating), pos
        ip i, j, num_bytes = block_size * f_bytes
        floating* chunk = NULL
        FILE* f = NULL

    with nogil, parallel():
        chunk = <floating*> malloc(num_bytes)

        if chunk is NULL:
            with gil:
                raise MemoryError('Unable to allocate chunk')

        f = fopen(filename, "rb")

        if f is NULL:
            free(chunk)
            chunk = NULL

            with gil:
                raise IOError('Unable to open file %s' % filename)

        # static < guided < dynamic < runtime FOR THIS CASE
        for i in prange(n, schedule='static'):
            pos = fp_locs[i]

            if fseek(f, pos, SEEK_SET) == -1:
                free(chunk)
                fclose(f)

                with gil:
                    raise IOError('Unable to seek to file position %d' % pos)

            if not fread(chunk, num_bytes, 1, f):
                free(chunk)
                fclose(f)

                with gil:
                    raise IOError('Unable to read any more bytes from '
                                  '%s' % filename)

            for j in range(block_size):
                spikes[i, j] = chunk[j]

        fclose(f)
        f = NULL

        free(chunk)
        chunk = NULL

    return 0
