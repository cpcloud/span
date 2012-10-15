from numpy cimport float32_t as float32, int64_t as int64, ndarray

from libc.stdio cimport fopen, fclose, fread, fseek, SEEK_SET, FILE
from libc.stdlib cimport malloc, free

from cython.parallel cimport prange, parallel
from cython cimport boundscheck, wraparound


@boundscheck(False)
@wraparound(False)
cpdef read_tev(char* filename, int64 nsamples, ndarray[int64] fp_locs,
               ndarray[float32, ndim=2] spikes):
    """
    """
    assert filename, 'filename (1st argument) cannot be empty'

    cdef:
        int64 i, j
        int64 n = fp_locs.shape[0], nbytes = sizeof(float32)

        float32* spikes_data = NULL
        float32* chunk_data = NULL

        int64* fp_locs_data = NULL

        FILE* f = NULL

    with nogil, parallel():
        spikes_data = <float32*> spikes.data
        chunk_data = <float32*> malloc(nbytes * nsamples)
        fp_locs_data = <int64*> fp_locs.data

        f = fopen(filename, 'rb')

        if not f:
            free(chunk_data)
            chunk_data = NULL

            with gil:
                raise IOError('Unable to open file %s' % filename)

        for i in prange(n, schedule='guided'):
            # go to the ith file pointer location
            fseek(f, fp_locs_data[i], SEEK_SET)

            # read nbytes * nsamples bytes into chunk_data
            fread(chunk_data, nbytes, nsamples, f)

            # assign the chunk data to the spikes array
            for j in prange(nsamples, schedule='guided'):
                spikes_data[i * nsamples + j] = chunk_data[j]

        # get rid of the chunk data
        free(chunk_data)
        chunk_data = NULL

        fclose(f)
        f = NULL
