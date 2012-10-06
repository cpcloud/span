from numpy cimport float32_t as float32, int64_t as int64, ndarray

from libc.stdio cimport fopen, fclose, fread, fseek, SEEK_SET, FILE
from libc.stdlib cimport malloc, free

from cython.parallel cimport prange, parallel
cimport cython


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef read_tev(char* filename, int64 nsamples, ndarray[int64] fp_locs,
               ndarray[float32, ndim=2] spikes):
    """
    """
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
            free(<void*> chunk_data)
            chunk_data = NULL
            with gil:
                return -1

        for i in prange(n):
            fseek(f, fp_locs_data[i], SEEK_SET)

            fread(<void*> chunk_data, nbytes, nsamples, f)
        
            for j in prange(nsamples):
                spikes_data[i * nsamples + j] = chunk_data[j]

        free(<void*> chunk_data)
        chunk_data = NULL

        fclose(f)
