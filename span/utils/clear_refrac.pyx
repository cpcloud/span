from numpy cimport uint8_t as u1, int64_t as i8
cimport cython


@cython.wraparound(False)
@cython.boundscheck(False)
cdef void _clear_refrac(u1[:, :] a, i8 window):
    cdef i8 channel, i, sample, nsamples = a.shape[0], nchannels = a.shape[1]

    with nogil:
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
def clear_refrac(u1[:, :] a not None, i8 window):
    _clear_refrac(a, window)
