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
from libc.stdio cimport fopen, fclose, fread, fseek, SEEK_SET, FILE
from libc.stdlib cimport malloc, free

from cython.parallel cimport parallel, prange

cimport cython

from cython cimport floating

ctypedef fused integral:
    np.int8_t
    np.int16_t
    np.int32_t
    np.int64_t
    np.npy_intp


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _read_tev(char* filename, integral nsamples, integral[:] fp_locs,
                    floating[:, :] spikes):
    cdef:
        np.npy_intp i, j, n
        size_t f_bytes

        floating* chunk = NULL

        FILE* f = NULL

    n = fp_locs.shape[0]
    f_bytes = sizeof(floating)

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
def read_tev(char* filename, integral nsamples, integral[:] fp_locs not None,
             floating[:, :] spikes not None):
    """Read a TDT tev file in. Slightly faster than the pure Python version.

    Parameters
    ----------
    filename : char *
        Name of the TDT file to load.

    nsamples : integral
        The number of samples per chunk of data.

    fp_locs : integral[:]
        The array of locations of each chunk in the TEV file.

    spikes : floating[:, :]
        Output array
    """
    assert filename is not NULL, 'filename (1st argument) cannot be empty'
    assert nsamples > 0, '"nsamples" must be greater than 0'
    _read_tev(filename, nsamples, fp_locs, spikes)
