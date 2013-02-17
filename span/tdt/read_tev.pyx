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
cdef int read_tev_parallel_impl(char* filename, integral[:, :] grouped,
                                ip blocksize,
                                floating[:, :] spikes) nogil except -1:

    cdef:
        ip c, b, k, byte, low, high, pos, nchannels, nblocks

        size_t f_bytes = sizeof(floating)
        size_t num_bytes = f_bytes * blocksize
        floating* chunk = NULL
        FILE* f = NULL

    nchannels = grouped.shape[1]
    nblocks = grouped.shape[0]

    with nogil, parallel():
        chunk = <floating*> malloc(num_bytes)

        if not chunk:
            with gil:
                raise MemoryError('unable to allocate chunk of size %d bytes'
                                  % num_bytes)

        f = fopen(filename, 'rb')

        if not f:
            free(chunk)
            chunk = NULL

            with gil:
                raise IOError('unable to open file %s' % filename)

        for c in prange(nchannels, schedule='guided'):
            for b in range(nblocks):

                pos = grouped[b, c]

                if fseek(f, pos, SEEK_SET) == -1:
                    fclose(f)
                    free(chunk)

                    with gil:
                        raise IOError('could not seek to pos %d in %s'
                                      % (pos, filename))

                if not fread(chunk, num_bytes, 1, f):
                    fclose(f)
                    free(chunk)

                    with gil:
                        raise IOError('could not read from %s' % filename)

                low = b * blocksize
                high = (b + 1) * blocksize

                for k, byte in enumerate(range(low, high)):
                    spikes[byte, c] = chunk[k]

        free(chunk)
        chunk = NULL

        fclose(f)
        f = NULL

    return 0


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef _read_tev_parallel(char* filename, integral[:, :] grouped, ip blocksize,
                         floating[:, :] spikes):
    read_tev_parallel_impl(filename, grouped, blocksize, spikes)
