# clear_refrac.pyx ---

# Copyright (C) 2012 Copyright (C) 2012 Phillip Cloud <cpcloud@gmail.com>

# Author: Phillip Cloud <cpcloud@gmail.com>

# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.


cimport cython
cimport numpy as np
from numpy cimport uint8_t as u1, npy_intp as ip


@cython.wraparound(False)
@cython.boundscheck(False)
cdef void _clear_refrac(u1[:, :] a, ip window) nogil:
    cdef ip channel, i, sample, sp1, nsamples, nchannels

    nsamples = a.shape[0]
    nchannels = a.shape[1]

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
def clear_refrac(u1[:, :] a not None, ip window):
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
