import numpy as np
cimport numpy as np
cimport cython

ctypedef np.uint8_t uint8
ctypedef np.int64_t int64
ctypedef np.float64_t float64


@cython.wraparound(False)
@cython.boundscheck(False)
cdef _clear_refrac_out(np.ndarray[uint8, ndim=2, cast=True] a, int64 spike_window):
    cdef:
        int64 nchannels = a.shape[1], nsamples = a.shape[0], channel, i, sample
        int64 sample_plus_one

    for channel in xrange(nchannels):
        sample = 0
        while sample < nsamples:
            if a[sample, channel]:
                sample_plus_one = sample + 1
                for i in xrange(sample_plus_one,
                                sample_plus_one + spike_window):
                    a[i, channel] = False
                sample += spike_window
            sample += 1


@cython.wraparound(False)
@cython.boundscheck(False)
def clear_refrac_out(np.ndarray[uint8, ndim=2, cast=True] a, int64 spike_window):
    _clear_refrac_out(a, spike_window)


@cython.wraparound(False)
@cython.boundscheck(False)
cdef _thresh_out(np.ndarray[float64, ndim=2] a, np.ndarray[float64, ndim=1] thresh,
               np.ndarray[uint8, ndim=2] o):
    cdef:
        int64 channel, nchannels = a.shape[1]
        int64 sample, nsamples = a.shape[0]
        float64 thr

    for channel in xrange(nchannels):
        thr = thresh[channel]
        for sample in xrange(nsamples):
            o[sample, channel] = a[sample, channel] > thr


@cython.wraparound(False)
@cython.boundscheck(False)            
def thresh_out(np.ndarray[float64, ndim=2] a, np.ndarray[float64, ndim=1] thresh,
               np.ndarray[uint8, ndim=2, cast=True] o):
    _thresh_out(a, thresh, o)


@cython.wraparound(False)
@cython.boundscheck(False)
def thresh(np.ndarray[float64, ndim=2] a, np.ndarray[float64, ndim=1] thr):
    cdef:
        int64 nsamples = a.shape[0], nchannels = a.shape[1]
        np.ndarray[uint8, ndim=2, cast=True] o = np.zeros((nsamples, nchannels),
                                                          dtype=np.bool_)
    thresh_out(a, thr, o)
    return o


@cython.wraparound(False)
@cython.boundscheck(False)
def clear_refrac(np.ndarray[uint8, ndim=2, cast=True] a, int64 spike_window):
    cdef np.ndarray[uint8, ndim=2, cast=True] cleared = a.copy()
    clear_refrac_out(cleared, spike_window)
    return cleared


@cython.wraparound(False)
@cython.boundscheck(False)
def thresh_and_clear(np.ndarray[float64, ndim=2] a,
                     np.ndarray[float64, ndim=2] thr,
                     int64 spike_window):
    cdef:
        np.ndarray[uint8, ndim=2] threshed = thresh(a, thr)
        np.ndarray[uint8, ndim=2] cleared = clear_refrac(threshed, spike_window)
    return cleared
