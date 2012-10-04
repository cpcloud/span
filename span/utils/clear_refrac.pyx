import numpy as np
cimport numpy as np
cimport cython

from cython.parallel cimport parallel, prange

ctypedef np.uint8_t uint8


@cython.wraparound(False)
@cython.boundscheck(False)
def clear_refrac(np.ndarray[uint8, ndim=2, cast=True] a not None, long window):
    cdef:
        long channel, i, sample
        long nsamples = a.shape[0], nchannels = a.shape[1]
        uint8* a_data = <uint8*> a.data

    for channel in xrange(nchannels):
        sample = 0

        while sample + window < nsamples:
            if a_data[sample * nchannels + channel]:
                with nogil, parallel():
                    for i in prange(sample + 1, sample + window + 1,
                                    schedule='guided'):
                        a_data[i * nchannels + channel] = 0
                sample += window
            sample += 1

