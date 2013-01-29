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
cpdef _read_tev_serial(char* filename, integral[:, :] grouped, ip blocksize,
                       floating[:, :] spikes):

    cdef:
        ip c, b, k, byte, low, high
        ip nchannels = grouped.shape[1], nblocks = grouped.shape[0]

        size_t f_bytes = sizeof(floating)
        floating* chunk = NULL
        FILE* f = NULL

    with nogil:
        chunk = <floating*> malloc(f_bytes * blocksize)

        if not chunk:
            with gil:
                raise MemoryError('Cannot allocate chunk of size '
                                  '%i' % blocksize)

        f = fopen(filename, "rb")

        if not f:
            free(chunk)

            with gil:
                raise IOError("Unable to open file %s" % filename)

        for c in xrange(nchannels):
            for b in xrange(nblocks):

                fseek(f, grouped[b, c], SEEK_SET)

                if not fread(chunk, f_bytes, blocksize, f):
                    free(chunk)
                    fclose(f)

                    with gil:
                        raise IOError('Unable to read %i elements from '
                                      '%s' % (blocksize, filename))

                low = b * blocksize
                high = (b + 1) * blocksize

                for k, byte in enumerate(xrange(low, high)):
                    spikes[byte, c] = chunk[k]

        free(chunk)
        fclose(f)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef _read_tev_parallel(char* filename, integral[:, :] grouped, ip blocksize,
                         floating[:, :] spikes):

    cdef:
        ip c, b, k, byte, low, high
        ip nchannels = grouped.shape[1], nblocks = grouped.shape[0]

        size_t f_bytes = sizeof(floating)
        floating* chunk = NULL
        FILE* f = NULL

    with nogil, parallel():
        chunk = <floating*> malloc(f_bytes * blocksize)

        if not chunk:
            with gil:
                raise MemoryError('Unable to allocate array of size '
                                  '%i' % blocksize)

        f = fopen(filename, "rb")

        if not f:
            free(chunk)

            with gil:
                raise IOError('Unable to open file %s' % filename)

        for c in prange(nchannels, schedule='static'):
            for b in xrange(nblocks):

                fseek(f, grouped[b, c], SEEK_SET)
                if not fread(chunk, f_bytes, blocksize, f):
                    free(chunk)
                    fclose(f)

                    with gil:
                        raise IOError('Unable to read %i elements from '
                                      '%s' % (blocksize, filename))

                low = b * blocksize
                high = (b + 1) * blocksize

                for k, byte in enumerate(xrange(low, high)):
                    spikes[byte, c] = chunk[k]


        free(chunk)
        fclose(f)
