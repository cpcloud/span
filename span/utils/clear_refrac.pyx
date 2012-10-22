from numpy cimport uint8_t as uint8, int64_t as int64
cimport cython


@cython.wraparound(False)
@cython.boundscheck(False)
cdef void _clear_refrac(uint8[:, :] a, int64 window) nogil:
    cdef:
        int64 channel, i, sample
        int64 nsamples = a.shape[0], nchannels = a.shape[1]

    for channel in xrange(nchannels):
        sample = 0

        while sample + window < nsamples:
            if a[sample, channel]:
                for i in xrange(sample + 1, sample + window + 1):
                    a[i, channel] = 0
                sample += window
            sample += 1


@cython.wraparound(False)
@cython.boundscheck(False)
cpdef clear_refrac(uint8[:, :] a, int64 window):
    _clear_refrac(a, window)
