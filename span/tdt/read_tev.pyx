cimport numpy as np

from libc.stdio cimport fopen, fclose, fread, fseek, SEEK_SET, FILE
from libc.stdlib cimport malloc, free

ctypedef np.float32_t float32
ctypedef np.int64_t int64

cpdef read_tev(char* filename, int64 nsamples, np.ndarray[int64] fp_locs,
               np.ndarray[float32, ndim=2] spikes):
    """
    """
    cdef:
        int64 i, j, r
        int64 n = fp_locs.shape[0], nbytes = sizeof(float32)

        float32* spikes_data = <float32*> spikes.data
        float32* chunk_data = <float32*> malloc(nbytes * nsamples)

        int64* fp_locs_data = <int64*> fp_locs.data

        FILE* f = fopen(filename, 'rb')

    if not f:
        free(<void*> chunk_data)
        chunk_data = NULL
        return -1

    for i in xrange(n):
        fseek(f, fp_locs_data[i], SEEK_SET)

        fread(<void*> chunk_data, nbytes, nsamples, f)
        
        for j in xrange(nsamples):
            spikes_data[i * nsamples + j] = chunk_data[j]

    free(<void*> chunk_data)
    chunk_data = NULL

    return fclose(f)
