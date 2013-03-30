from libc.math cimport fabs
from cython.parallel cimport prange, parallel
ctypedef Py_ssize_t ip

import numpy as np
cimport cython
from numpy cimport ndarray

cdef double fmax(double a, double b) nogil:
    return a if a > b else b


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double absmax(double[:] x) nogil:
    cdef:
        ip i, n = x.shape[0]
        double m = fabs(x[0])

    with nogil:
        for i in range(1, n):
            m = fmax(m, fabs(x[i]))

    return m


cdef ip eq(ip[:] a, ip[:] b):
    cdef:
        ip i, na = a.shape[0], nb = b.shape[0]
        ip n = na

    for i in range(n):
        if a[i] != b[i]:
            return False

    return True


cdef ip ne(ip[:] a, ip[:] b):
    return not eq(a, b)


cdef void addc(ip[:] a, ip v, ip[:] out):
    cdef ip i, n = a.shape[0]
    for i in range(n):
        out[i] = a[i] + v


cdef class CartesianIndex:
    cdef:
        ip[:] lengths, indices
        ip n, has_next

    def __cinit__(self, ip[:] lengths):
        self.lengths = lengths
        self.indices = np.zeros(lengths.shape[0], dtype=np.int64)
        self.n = self.indices.shape[0]
        self.has_next = True

    def __iter__(self):
        while self.has_next:
            yield np.asarray(self.__next__())

    def __next__(self):
        cdef:
            ip[:] result = self.indices.copy()
            ip i

        for i in range(self.n - 1, -1, -1):
            if self.indices[i] == self.lengths[i] - 1:
                self.indices[i] = 0
                self.has_next = i
            else:
                self.indices[i] += 1
                break

        return result


cpdef CartesianIndex make_cartesian_index(list ns):
    return CartesianIndex(np.asanyarray(ns))


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef int _num2name(ip num, ip[:] rhs, ip[:] letters, ip[:] out) nogil except -1:
    cdef:
        ip i, li, lj, lk, ll, n = letters.shape[0]
        ip r0 = rhs[0], r1 = rhs[1], r2 = rhs[2], r3 = rhs[3]

    with nogil, parallel():
        for i in prange(n):
            li = letters[i]

            for j in range(n):
                lj = letters[j]

                for k in range(n):
                    lk = letters[k]

                    for l in range(n):
                        ll = letters[l]

                        if li * r0 + lj * r1 + r2 * lk + r3 * ll == num:
                            out[0] = li
                            out[1] = lj
                            out[2] = lk
                            out[3] = ll
                            return 0
