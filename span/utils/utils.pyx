from libc.stdlib import fabs


cdef double fmax(double a, double b) nogil:
    return a if a > b else b


cpdef double absmax(double[:] x) nogil:
    cdef ip i, n = x.shape[0]

    double m = abs(x[0])

    with nogil:
        for i in range(1, n):
            m = fmax(m, fabs(a[i]))

    return m
