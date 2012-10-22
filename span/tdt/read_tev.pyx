import os

from numpy cimport float32_t as float32, int64_t as int64

from libc.stdio cimport fopen, fclose, fread, fseek, SEEK_SET, FILE
from libc.stdlib cimport malloc, free
from libc.string cimport strlen

from cython.parallel cimport prange, parallel
cimport cython


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef read_tev(char* filename, int64 nsamples, int64[:] fp_locs,
               float32[:, :] spikes):
    """

    Parameters
    ----------
    filename : char *
    nsamples : int64
    fp_locs : int64[:]
    spikes : float32[:, :]

    Raises
    ------
    AssertionError
        If filename is a NULL pointer
    """
    assert filename is not NULL, 'filename (1st argument) cannot be empty'
    
    # _, ext = os.path.splitext(filename)
    # assert ext == 'tev', 'extension must be "tev"'
    # assert os.path.exists(filename), '%s does not exist' % filename

    cdef:
        int64 i, j, n = fp_locs.shape[0], nbytes = sizeof(float32)

        float32* chunk = NULL

        FILE* f = NULL

    with nogil, parallel():
        chunk = <float32*> malloc(nbytes * nsamples)

        f = fopen(filename, 'rb')

        if not f:
            if chunk:
                free(chunk)
                chunk = NULL

            with gil:
                raise IOError('Unable to open file %s' % filename)

        for i in prange(n, schedule='guided'):
            # go to the ith file pointer location
            fseek(f, fp_locs[i], SEEK_SET)

            # read nbytes * nsamples bytes into chunk_data
            fread(chunk, nbytes, nsamples, f)

            # assign the chunk data to the spikes array
            for j in prange(nsamples, schedule='guided'):
                spikes[i, j] = chunk[j]

        # get rid of the chunk data
        if chunk:
            free(chunk)
            chunk = NULL

        fclose(f)
        f = NULL
