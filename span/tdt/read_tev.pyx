import numpy as np
cimport numpy as np

from libc.stdio cimport fopen, fclose, fread, fseek, SEEK_SET, FILE


cpdef read_tev(char* filename, long nsamples, np.ndarray[long] fp_locs,
               np.ndarray[np.float32_t, ndim=2] spikes):
    cdef:
        long i, j, n = fp_locs.shape[0]
        
        FILE* f = fopen(filename, 'rb')

        np.ndarray[np.float32_t] chunk = np.empty(nsamples, dtype=np.float32)

        np.float32_t* spikes_data = <np.float32_t*> spikes.data
        np.float32_t* chunk_data = <np.float32_t*> chunk.data
        long* fp_locs_data = <long*> fp_locs.data

    for i in xrange(n):
        fseek(f, fp_locs_data[i], SEEK_SET)

        fread(<void*> chunk_data, sizeof(np.float32_t), nsamples, f)
        
        for j in xrange(nsamples):
            spikes_data[i * nsamples + j] = chunk_data[j]

    fclose(f)
    
    
