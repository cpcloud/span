from numpy cimport uint8_t as u1, npy_intp as i8

cimport cython


@cython.wraparound(False)
@cython.boundscheck(False)
cdef void _clear_refrac(u1[:, :] a, i8 window) nogil:
    cdef i8 channel, i, sample, sp1, nsamples, nchannels

    nsamples = a.shape[0], nchannels = a.shape[1]

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
def clear_refrac(u1[:, :] a not None, i8 window):
    """Clear the refractory period of a boolean array.

    Parameters
    ----------
    a : array_like
    window : npy_intp

    Notes
    -----
    If ``a.dtype == np.bool_`` in Python then this function will not work
    unless ``a.view(uint8)`` is passed.

    Raises
    ------
    AssertionError
        If `window` is less than or equal to 0
    """
    assert window > 0, '"window" must be greater than 0'
    _clear_refrac(a, window)
