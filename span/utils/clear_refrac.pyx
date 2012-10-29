from numpy cimport uint8_t as u1, npy_intp
cimport cython


@cython.wraparound(False)
@cython.boundscheck(False)
cdef void _clear_refrac(u1[:, :] a, npy_intp window):
    cdef npy_intp channel, i, sample, sp1
    cdef npy_intp nsamples = a.shape[0], nchannels = a.shape[1]

    with nogil:
        for channel in xrange(nchannels):
            sample = 0

            while sample + window < nsamples:
                if a[sample, channel]:
                    sp1 = sample + 1

                    for i in xrange(sp1, sp1 + window):
                        a[i, channel] = 0

                    sample += window
                sample += 1


@cython.wraparound(False)
@cython.boundscheck(False)
def clear_refrac(u1[:, :] a not None, npy_intp window):
    assert window > 0, '"window" must be greater than 0'
    _clear_refrac(a, window)
