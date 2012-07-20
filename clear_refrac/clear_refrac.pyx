import numpy as np
cimport numpy as np
cimport cython


ctypedef np.uint8_t uint8
ctypedef np.int64_t int64
ctypedef np.float64_t float64


@cython.wraparound(False)
@cython.boundscheck(False)
cdef void _clear_refrac(np.ndarray[uint8, ndim=2] a, int64 spike_window_obs):
    cdef int64 nchannels = a.shape[1], nsamples = a.shape[0]
    cdef int64 channel, sample, i, chnsamps
    cdef uint8* a_data = <uint8*> a.data

    for channel in xrange(nchannels):
        sample = 0
        while sample < nsamples:
            chnsamps = channel * nsamples
            if a_data[chnsamps + sample]:
                for i in xrange(sample + 1, sample + spike_window_obs + 1):
                    a_data[chnsamps + i] = 0
                sample += spike_window_obs
            sample += 1


@cython.wraparound(False)
@cython.boundscheck(False)
cdef void _thresh(np.ndarray[float64, ndim=2] a,
                  np.ndarray[float64, ndim=1] thresh,
                  np.ndarray[uint8, ndim=2] o):
    cdef int64 channel, nchannels = a.shape[1]
    cdef int64 sample, nsamples = a.shape[0]
    cdef int64 offset
    cdef float64 thr

    cdef float64* a_data = <float64*> a.data
    cdef uint8* o_data = <uint8*> o.data
    cdef float64* thresh_data = <float64*> thresh.data

    for channel in xrange(nchannels):
        thr = thresh_data[channel]
        for sample in xrange(nsamples):
            offset = channel * nsamples + sample
            o_data[offset] = a_data[offset] > thr


@cython.wraparound(False)
@cython.boundscheck(False)
def thresh(np.ndarray[float64, ndim=2] a, np.ndarray[float64, ndim=1] thr):
    cdef int64 nchannels = a.shape[1]
    cdef int64 nsamples = a.shape[0]
    cdef np.ndarray[uint8, ndim=2] o = np.zeros((nsamples, nchannels),
                                               dtype=np.uint8)
    _thresh(a, thr, o)
    return o


@cython.wraparound(False)
@cython.boundscheck(False)
def clear_refrac(np.ndarray[uint8, ndim=2] a, int64 spike_window_obs):
    cdef np.ndarray[uint8, ndim=2] cleared = a.copy()
    _clear_refrac(cleared, spike_window_obs)
    return cleared


@cython.wraparound(False)
@cython.boundscheck(False)
def thresh_and_clear(np.ndarray[float64, ndim=2] a,
                     np.ndarray[float64, ndim=1] thr,
                     int64 spike_window_obs):
    cdef np.ndarray[uint8, ndim=2] threshed = thresh(a, thr)
    cdef np.ndarray[uint8, ndim=2] cleared = clear_refrac(threshed,
                                                         spike_window_obs)
    return cleared
