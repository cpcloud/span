import numpy as np
cimport numpy as np
cimport cython

ctypedef np.uint8_t uint8


@cython.wraparound(False)
@cython.boundscheck(False)
cdef void _clear_refrac_out(np.ndarray[uint8, ndim=2, cast=True] a, long window):
    cdef:
        long channel, i, sample, sample_plus_one, loc
        long nchannels = a.shape[1], nsamples = a.shape[0]
        uint8* a_data = <uint8*> a.data

    for channel in xrange(nchannels):
        sample = 0
        while sample < nsamples:
            if a_data[sample * nchannels + channel]:
                sample_plus_one = sample + 1
                for i in xrange(sample_plus_one, sample_plus_one + window):
                    a_data[i * nchannels + channel] = 0
                sample += window
            sample += 1


@cython.wraparound(False)
@cython.boundscheck(False)
def clear_refrac_out(np.ndarray[uint8, ndim=2, cast=True] a not None,
                     long window):
    _clear_refrac_out(a, window)


@cython.wraparound(False)
@cython.boundscheck(False)
cdef void _thresh_out(np.ndarray[double, ndim=2] a,
                      np.ndarray[double, ndim=1] thresh,
                      np.ndarray[uint8, ndim=2] o):
    cdef:
        double thr
        long loc, channel, sample
        long nchannels = a.shape[1], nsamples = a.shape[0]

        uint8* o_data = <uint8*> o.data
        double* a_data = <double*> a.data
        double* thr_data = <double*> thresh.data

    for channel in xrange(nchannels):
        thr = thr_data[channel]

        for sample in xrange(nsamples):
            loc = sample * nchannels + channel
            o_data[loc] = a_data[loc] > thr


@cython.wraparound(False)
@cython.boundscheck(False)
def thresh_out(np.ndarray[double, ndim=2] a not None,
               np.ndarray[double, ndim=1] thresh not None,
               np.ndarray[uint8, ndim=2] o=None):
    if o is None:
        o = np.empty((a.shape[0], a.shape[1]), np.uint8)
    _thresh_out(a, thresh, o)
    return o


@cython.wraparound(False)
@cython.boundscheck(False)
def thresh(np.ndarray[double, ndim=2] a not None,
           np.ndarray[double, ndim=1] thr not None):
    cdef long nsamples = a.shape[0], nchannels = a.shape[1]
    cdef np.ndarray[uint8, ndim=2] o = np.empty((a.shape[0], a.shape[1]),
                                                np.uint8)
    _thresh_out(a, thr, o)
    return o


@cython.wraparound(False)
@cython.boundscheck(False)
def clear_refrac(np.ndarray[uint8, ndim=2, cast=True] a not None, long window):
    cdef np.ndarray[uint8, ndim=2] cleared = a.copy()
    _clear_refrac_out(cleared, window)
    return cleared


@cython.wraparound(False)
@cython.boundscheck(False)
def thresh_and_clear(np.ndarray[double, ndim=2] a, np.ndarray[double, ndim=2] thr,
                     long window):
    cdef np.ndarray[uint8, ndim=2] threshed = thresh(a, thr)
    cdef np.ndarray[uint8, ndim=2] cleared = clear_refrac(threshed, window)
    return cleared
