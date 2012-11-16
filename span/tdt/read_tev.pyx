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

cimport numpy as np
from numpy cimport (npy_intp as ip, float32_t as f4, float64_t as f8,
                    int64_t as i8, int32_t as i4, int16_t as i2, int8_t as i1)
from numpy cimport (uint8_t as u1, uint16_t as u2, uint32_t as u4,
                    uint64_t as u8)
from libc.stdio cimport fopen, fclose, fread, fseek, SEEK_SET, FILE
from libc.stdlib cimport malloc, free

from cython.parallel cimport parallel, prange

cimport cython

ctypedef fused floating:
    f4
    f8

ctypedef fused integer:
    i1
    i2
    i4
    i8
    ip
    ssize_t

    u1
    u2
    u4
    u8
    size_t


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _read_tev(char* filename, integer nsamples, integer[:] fp_locs,
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
            with gil:
                raise MemoryError('Out of memory when allocating chunk')

        f = fopen(filename, 'rb')

        if not f:
            free(chunk)
            chunk = NULL

            with gil:
                raise IOError('Unable to open file %s' % filename)

        for i in prange(n, schedule='guided'):
            # go to the ith file pointer location
            fseek(f, fp_locs[i], SEEK_SET)

            # read floating_bytes * nsamples bytes into chunk_data
            fread(chunk, f_bytes, nsamples, f)

            # assign the chunk data to the spikes array
            for j in xrange(nsamples):
                spikes[i, j] = chunk[j]

        # get rid of the chunk data
        free(chunk)
        chunk = NULL

        fclose(f)
        f = NULL


@cython.wraparound(False)
@cython.boundscheck(False)
def read_tev(char* filename, ip nsamples, integer[:] fp_locs not None,
             floating[:, :] spikes not None):
    """Read a TDT tev file in. Slightly faster than the pure Python version.

    Parameters
    ----------
    filename : char *
        Name of the TDT file to load.

    nsamples : i8
        The number of samples per chunk of data.

    fp_locs : i8[:]
        The array of locations of each chunk in the TEV file.

    spikes : floating[:, :]
        Output array
    """
    assert filename, 'filename (1st argument) cannot be empty'
    assert nsamples > 0, '"nsamples" must be greater than 0'
    _read_tev(filename, nsamples, fp_locs, spikes)
