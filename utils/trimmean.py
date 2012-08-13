#!/usr/bin/env python

import warnings
from itertools import imap
import numpy as np


class TrimmerBase(object):
    def __init__(self, percent, n, sz=1):
        super(TrimmerBase, self).__init__()
        self.percent, self.n, self.sz = imap(np.asanyarray, (percent, n, sz))
    
    def trim(self, x):
        if not self.alltrimmed:
            x = np.ma.masked_array(x, np.isnan(x))
            m = self.trim_impl(x)
            try:
                m = m.filled(np.nan)
            except AttributeError:
                pass
        else:
            _m = nans(self.sz, dtype=x.dtype)
            m = np.ma.masked_array(_m, np.isnan(_m))
        return m, self.alltrimmed

    def trim_impl(self, x):
        return x[self.k0 + 1:self.n - self.k0].mean(axis=0)

    @property
    def k0(self):
        return NotImplemented

    @property
    def alltrimmed(self):
        return not (self.n and self.n > 0 and self.k0 < self.n / 2.0)

    @property
    def n(self):
        return self.__n

    @n.setter
    def n(self, value):
        self.__n = value
        self.k = np.asanyarray(value * self.percent / 200.0)


class RoundTrimmer(TrimmerBase):
    def __init__(self, *args, **kwargs):
        super(RoundTrimmer, self).__init__(*args, **kwargs)

    @property
    def k0(self):
        return np.round(self.k - np.finfo(self.k.dtype).eps)


class UnweightedTrimmer(TrimmerBase):
    def __init__(self, *args, **kwargs):
        super(UnweightedTrimmer, self).__init__(*args, **kwargs)

    @property
    def k0(self):
        return np.floor(self.k)


class WeightedTrimmer(UnweightedTrimmer):
    def __init__(self, *args, **kwargs):
        super(WeightedTrimmer, self).__init__(*args, **kwargs)
        self.f = 1 + self.k0 - self.k

    @property
    def alltrimmed(self):
        return self.n and self.n > 0 and (self.k0 < self.n / 2.0 or self.f > 0)

    def trim_impl(self, x):
        xsum = x[self.k0 + 2:self.n - self.k0 - 1].sum(axis=0)
        xfsum = self.f * x[self.k0 + 1] + self.f * x[self.n - self.k0]
        maxsize = max(0, self.n - 2 * self.k0 - 2)
        return (xsum + xfsum) / (maxsize + 2.0 * self.f)


def ipermute(b, order):
    norder = order.size
    iorder = np.zeros(norder, dtype=int)
    iorder[order] = np.arange(norder)
    return b.transpose(inverseorder)


def mad(x, flag=1, axis=None):
    x = np.ma.masked_array(x, np.isnan(x))
    f = np.median if flag else np.mean
    return f(np.abs(x - f(x, axis=axis)), axis=axis).filled(np.nan)


def nans(shape, dtype=type(np.nan)):
    a = np.empty(shape, dtype=dtype)
    a.fill(np.nan)
    return a


def trimmean(x, percent, flag='round', axis=None):
    trimmers = {'round': RoundTrimmer, 'weighted': WeightedTrimmer,
                'floor': UnweightedTrimmer}
    if not x.size:
        return np.nan
    
    if axis is None:
        axis = np.where(np.array(x.shape) != 1)[0][0]
        if not axis.size:
            axis = 0

    allmissing = np.isnan(x).all(axis)
    xdims = x.ndim
    perm = []
    if axis > 0:
        perm = np.hstack((np.arange(axis, max(xdims, axis)),
                          np.arange(axis - 1)))
        x = x.transpose(perm)
    x.sort(axis=0)
    sz = list(x.shape)
    sz[0] = 1
    Trimmer = trimmers[flag]
    if not np.isnan(x).any():
        n = x.shape[0]
        trimmer = Trimmer(percent, n, sz=sz)
        m, alltrimmed = trimmer.trim(x)
    else:
        m = nans(sz, dtype=x.dtype)
        alltrimmed = np.zeros(sz, dtype=bool)
        trimmer = Trimmer(percent, n=None)
        for j in xrange(np.prod(sz[1:])):
            trimmer.n = np.where(np.logical_not(np.isnan(x[:, j])))[0][-1]
            m[j], alltrimmed[j] = trimmer.trim(x[:, j])
    m = m.reshape(sz)
    if perm:
        m = ipermute(m, perm)
        alltrimmed = ipermute(alltrimmed, perm)
    trimmed_data = alltrimmed.copy()
    alltrimmed = np.logical_and(trimmed_data, np.logical_not(allmissing))
    if alltrimmed.any():
        if alltrimmed.all():
            warnings.warn('No data remain after trimming')
        else:
            warnings.warn('No data remain in some columns after trimming')
    return m.squeeze(), trimmed_data


if __name__ == '__main__':
    n = 100
    # x = np.arange(1, n + 1).astype(float, copy=False)
    x = np.random.randn(1000, 1000)
    # print x
    pct = 99.1
    trimmed = trimmean(x, pct)
    print trimmed
