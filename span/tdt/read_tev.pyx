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


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef _read_tev_parallel_specialized_unsafe(char* filename, i8[:, :] grouped,
                                            Py_ssize_t blocksize,
                                            f4[:, :] spikes):

    cdef:
        ip c, b, k, r, nchannels, nblocks

        Py_ssize_t f_bytes = sizeof(f4)
        Py_ssize_t num_bytes = f_bytes * blocksize

        f4* chunk = NULL
        # f4[:] _c
        FILE* f = NULL

    nchannels = grouped.shape[1]
    nblocks = grouped.shape[0]

    with nogil, parallel():
        chunk = <f4*> malloc(num_bytes)

        f = fopen(filename, "rb")

        for c in prange(nchannels, schedule='static'):
            for b in range(nblocks):

                fseek(f, grouped[b, c], SEEK_SET)
                fread(chunk, num_bytes, 1, f)

                for k, r in enumerate(range(b * blocksize,
                                            (b + 1) * blocksize)):
                    spikes[r, c] = chunk[k]

        free(chunk)
        fclose(f)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _read_tev_parallel_specialized_unsafe_pointers(char* filename,
                                                         ndarray[i8, ndim=2] _grouped,
                                                         Py_ssize_t blocksize,
                                                         ndarray[f4, ndim=2] _spikes):

    cdef:
        ip c, b, k, r
        ip nchannels = _grouped.shape[1], nblocks = _grouped.shape[0]

        Py_ssize_t f_bytes = sizeof(f4)
        Py_ssize_t num_bytes = f_bytes * blocksize

        f4* spikes = NULL, *chunk = NULL
        i8* grouped = NULL

        FILE* f = NULL

    with nogil, parallel():
        spikes = <f4*> _spikes.data
        grouped = <i8*> _grouped.data

        chunk = <f4*> malloc(num_bytes)

        f = fopen(filename, "rb")

        for c in prange(nchannels, schedule='static'):
            for b in range(nblocks):

                fseek(f, grouped[b * nchannels + c], SEEK_SET)
                fread(chunk, num_bytes, 1, f)

                for k, r in enumerate(range(b * blocksize,
                                            (b + 1) * blocksize)):
                    spikes[r * nchannels + c] = chunk[k]

        free(chunk)
        fclose(f)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef _read_tev_parallel_specialized_unsafe_pointers_python(char* filename,
                                                            ndarray[i8, ndim=2] grouped,
                                                            Py_ssize_t blocksize,
                                                            ndarray[f4, ndim=2] spikes):
    _read_tev_parallel_specialized_unsafe_pointers(filename, grouped,
                                                   blocksize, spikes)
