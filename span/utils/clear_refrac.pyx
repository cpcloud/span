from numpy cimport uint8_t as uint8, int64_t as int64, ndarray
from cython cimport boundscheck, wraparound


@wraparound(False)
@boundscheck(False)
cpdef clear_refrac(ndarray[uint8, ndim=2, cast=True] a, int64 window):
    cdef:
        int64 channel, i, sample
        int64 nsamples = a.shape[0], nchannels = a.shape[1]
        uint8* a_data = <uint8*> a.data

    for channel in xrange(nchannels):
        sample = 0

        while sample + window < nsamples:
            if a_data[sample * nchannels + channel]:
                for i in xrange(sample + 1, sample + window + 1):
                    a_data[i * nchannels + channel] = 0
                sample += window
            sample += 1
