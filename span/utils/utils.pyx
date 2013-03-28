from libc.math cimport fabs

ctypedef Py_ssize_t ip


cdef double fmax(double a, double b) nogil:
    return a if a > b else b


cpdef double absmax(double[:] x) nogil:
    cdef:
        ip i, n = x.shape[0]
        double m = fabs(x[0])

    with nogil:
        for i in range(1, n):
            m = fmax(m, fabs(x[i]))

    return m
