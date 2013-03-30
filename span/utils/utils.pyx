cimport cython
from cython cimport floating

ctypedef Py_ssize_t ip


cdef floating fabs(floating a) nogil:
    return -a if a < 0 else a


cdef floating fmax(floating a, floating b) nogil:
    return a if a > b else b


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef floating absmax(floating[:] x) nogil:
    cdef:
        ip i, n = x.shape[0]
        floating m = fabs(x[0])

    with nogil:
        for i in range(1, n):
            m = fmax(m, fabs(x[i]))

    return m
