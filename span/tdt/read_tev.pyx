from numpy cimport float32_t as f4, npy_intp as i8

from libc.stdio cimport fopen, fclose, fread, fseek, SEEK_SET, FILE
from libc.stdlib cimport malloc, free

from cython.parallel cimport parallel, prange

cimport cython


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _read_tev(char* filename, i8 nsamples, i8[:] fp_locs,
                    f4[:, :] spikes) nogil:
    cdef:
        i8 i, j, n = fp_locs.shape[0], f4_bytes = sizeof(f4)

        f4* chunk = NULL

        FILE* f = NULL

    with parallel():
        chunk = <f4*> malloc(f4_bytes * nsamples)

        f = fopen(filename, 'rb')

        if not f:
            if chunk:
                free(chunk)
                chunk = NULL

            with gil:
                assert chunk is NULL, 'memory leak when freeing chunk'
                raise IOError('Unable to open file %s' % filename)

        for i in prange(n):
            # go to the ith file pointer location
            fseek(f, fp_locs[i], SEEK_SET)

            # read f4_bytes * nsamples bytes into chunk_data
            fread(chunk, f4_bytes, nsamples, f)

            # assign the chunk data to the spikes array
            for j in xrange(nsamples):
                spikes[i, j] = chunk[j]

        # get rid of the chunk data
        if chunk:
            free(chunk)
            chunk = NULL

            with gil:
                assert chunk is NULL, 'memory leak when freeing chunk'

        fclose(f)
        f = NULL


@cython.wraparound(False)
@cython.boundscheck(False)
def read_tev(char* filename, i8 nsamples, i8[:] fp_locs not None,
             f4[:, :] spikes not None):
    """Read a TDT tev file in. Slightly faster than the pure Python version.

    Parameters
    ----------
    filename : char *
        Name of the TDT file to load.

    nsamples : i8
        The number of samples per chunk of data.

    fp_locs : i8[:]
        The array of locations of each chunk in the TEV file.

    spikes : f4[:, :]
        Output array
    """
    assert filename is not NULL, 'filename (1st argument) cannot be empty'
    assert nsamples > 0, '"nsamples" must be greater than 0'
    _read_tev(filename, nsamples, fp_locs, spikes)
