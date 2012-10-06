import numpy as np
cimport numpy as np

from libc.stdio import fopen, fclose, FILE


cpdef void read_tev(char* filename, long nsamples,
                    np.ndarray[np.float32_t, ndim=2] spikes):
    cdef long i, j, n = fp_locs.shape[0]
    cdef np.ndarray[np.float32_t] chunk = np.empty(nsamples, dtype=np.float32)
    cdef FILE* f = fopen(filename, 'rb')

    for i in xrange(n):
        fread(<void*> chunk.data, sizeof(np.float32_t), nsamples, f)

        for j in xrange(nsamples): 
            spikes[i, j] = chunk[j]

    fclose(f)
    
    
