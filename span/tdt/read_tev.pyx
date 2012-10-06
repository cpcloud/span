import numpy as np
cimport numpy as np

from libc.stdio cimport fopen, fclose, fread, fseek, SEEK_SET, FILE
from libc.stdlib cimport malloc, free

ctypedef np.float32_t float32

cpdef read_tev(char* filename, long nsamples, np.ndarray[long] fp_locs,
               np.ndarray[float32, ndim=2] spikes):
    cdef:
        long i, j
        long n = fp_locs.shape[0], nbytes = sizeof(float32)

        float32* spikes_data = <float32*> spikes.data
        float32* chunk_data = <float32*> malloc(nbytes * nsamples)

        long* fp_locs_data = <long*> fp_locs.data

        FILE* f = fopen(filename, 'rb')

    if not f:
        return -1

    for i in xrange(n):
        fseek(f, fp_locs_data[i], SEEK_SET)

        fread(<void*> chunk_data, nbytes, nsamples, f)
        
        for j in xrange(nsamples):
            spikes_data[i * nsamples + j] = chunk_data[j]

    free(<void*> chunk_data)
    chunk_data = NULL
    fclose(f)
