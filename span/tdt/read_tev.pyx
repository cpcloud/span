import os

from numpy cimport float32_t as float32, int64_t as int64

from libc.stdio cimport fopen, fclose, fread, fseek, SEEK_SET, FILE
from libc.stdlib cimport malloc, free

cimport cython


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _read_tev(char* filename, int64 nsamples, int64[:] fp_locs,
                    float32[:, :] spikes) nogil:
    """

    Parameters
    ----------
    filename : char *
    nsamples : int64
    fp_locs : int64[:]
    spikes : float32[:, :]
    """
        
    cdef:
        int64 i, j, n = fp_locs.shape[0], nbytes = sizeof(float32)

        float32* chunk = NULL

        FILE* f = NULL

    chunk = <float32*> malloc(nbytes * nsamples)

    f = fopen(filename, 'rb')

    if not f:
        if chunk:
            free(chunk)
            chunk = NULL

        with gil:
            assert not chunk, 'memory leak when freeing chunk'
            raise IOError('Unable to open file %s' % filename)

    for i in xrange(n):
        # go to the ith file pointer location
        fseek(f, fp_locs[i], SEEK_SET)

        # read nbytes * nsamples bytes into chunk_data
        fread(chunk, nbytes, nsamples, f)

        # assign the chunk data to the spikes array
        for j in xrange(nsamples):
            spikes[i, j] = chunk[j]

    # get rid of the chunk data
    if chunk:
        free(chunk)
        chunk = NULL

        with gil:
            assert not chunk, 'memory leak when freeing chunk'

    fclose(f)
    f = NULL


@cython.wraparound(False)
@cython.boundscheck(False)
def read_tev(char* filename, int64 nsamples, int64[:] fp_locs not None,
             float32[:, :] spikes not None):
    assert filename is not NULL, 'filename (1st argument) cannot be empty'
    _read_tev(filename, nsamples, fp_locs, spikes)
