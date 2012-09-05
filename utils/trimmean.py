#!/usr/bin/env python

import warnings

import numpy as np
import pylab as pl


class TrimmerBase(object):
    def __init__(self, percent, n, sz=1):
        super(TrimmerBase, self).__init__()
        self.percent, self.n, self.sz = map(np.asanyarray, (percent, n, sz))

    def trim(self, x, axis=None):
        if not self.alltrimmed:
            x = np.ma.masked_array(x, np.isnan(x))
            m = self.trim_impl(x, axis=axis)
            try:
                m = m.filled(np.nan)
            except AttributeError:
                pass
        else:
            _m = nans(self.sz, dtype=x.dtype)
            m = np.ma.masked_array(_m, np.isnan(_m))
        return m, self.alltrimmed

    def trim_impl(self, x, axis=None):
        self.data = x[self.k0 + 1:self.n - self.k0]
        return self.data.mean(axis=axis)

    @property
    def k0(self): return NotImplemented

    @property
    def alltrimmed(self):
        return not (self.n and self.n > 0 and self.k0 < self.n / 2.0)

    @property
    def n(self): return self.__n

    @n.setter
    def n(self, value):
        self.__n = value
        self.k = np.asanyarray(value * self.percent / 200.0)


class RoundTrimmer(TrimmerBase):
    def __init__(self, *args, **kwargs):
        super(RoundTrimmer, self).__init__(*args, **kwargs)

    @property
    def k0(self): return np.round(self.k - np.finfo(self.k.dtype).eps)


class UnweightedTrimmer(TrimmerBase):
    def __init__(self, *args, **kwargs):
        super(UnweightedTrimmer, self).__init__(*args, **kwargs)

    @property
    def k0(self): return np.floor(self.k)


class WeightedTrimmer(UnweightedTrimmer):
    def __init__(self, *args, **kwargs):
        super(WeightedTrimmer, self).__init__(*args, **kwargs)
        self.f = 1 + self.k0 - self.k

    @property
    def alltrimmed(self):
        return self.n and self.n > 0 and (self.k0 < self.n / 2.0 or self.f > 0)

    def trim_impl(self, x, axis=None):
        xsum = x[self.k0 + 2:self.n - self.k0 - 1].sum(axis=axis)
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


def nans(shape, dtype=np.float64):
    a = np.empty(shape, dtype=dtype)
    a.fill(np.nan)
    return a

def rounded(x, n, percent, size):
    x, n, percent, size = map(np.asanyarray, (x, n, percent, size))
    k = n * percent / 200.0
    k0 = np.round(k - np.finfo(k.dtype).eps).astype(int)
    if n and n > 0 and k0 < n / 2.0:
        m = x[k0:n - k0].mean(axis=0)
        alltrimmed = False
    else:
        m = nans(size, dtype=x.dtype)
        alltrimmed = True
    return m, alltrimmed

def unweighted(x, n, percent, size):
    x, n, percent, size = map(np.asanyarray, (x, n, percent, size))
    k0 = np.floor(n * percent / 200.0)
    
    if n and n > 0 and k0 < n / 2.0:
        m = x[k0:n - k0].mean(axis=0)
        alltrimmed = False
    else:
        m = nans(size, dtype=x.dtype)
        alltrimmed = True
    return m, alltrimmed

def weighted(x, n, percent, size):
    x, n, percent, size = map(np.asanyarray, (x, n, percent, size))
    k = n * percent / 200.0
    k0 = np.floor(k)
    f = 1 + k0 - k
    if n and n > 0 and (k0 < n / 2.0 or f > 0):
        numer = x[k0 + 2:n - k0 - 1].sum() + f * x[k0 + 1] + f * x[n - k0]
        denom = np.max(0, n - 2 * k0 - 2) + 2.0 * f
        m = numer / denom
        alltrimmed = False
    else:
        m = nans(size, dtype=x.dtype)
        alltrimmed = True
    return m, alltrimmed

TRIMMERS = {'rounded': rounded, 'weighted': weighted, 'floor': unweighted}

def trimmean(x, percent=5, trim_type='rounded', axis=None):
    """ """
    x, percent = np.asanyarray(x), np.asanyarray(percent)
    allmissing = np.isnan(x).all(axis=axis)
    xdims = x.ndim
    perm = []
    if axis is not None and axis > 1:
        perm = np.hstack((np.r_[axis:np.max(xdims, axis)], np.r_[axis - 1]))
        x = x.transpose(perm)
    x = np.ma.masked_equal(np.sort(x, axis=0), np.nan)
    size = list(x.shape)
    size[0] = 1
    trimmer = TRIMMERS[trim_type]
    if not np.isnan(x.ravel()).any():
        n = x.shape[0]
        m, alltrimmed = trimmer(x, n, percent, size)
    else:
        m = np.ma.masked_equal(nans(size, dtype=x.dtype), np.nan)
        alltrimmed = np.zeros(size, dtype=bool)
        for j in range(np.prod(size[1:])):
            xj = x.T[j]
            n, = np.where(np.logical_not(np.isnan(xj)))
            n = n[-1]
            m[j], alltrimmed[j] = trimmer(xj, n, percent, (1,))
    m = m.reshape(size)
    if perm:
        m = ipermute(m, perm)
        alltrimmed = ipermute(alltrimmed, perm)
    alltrimmed = np.logical_and(alltrimmed, np.logical_not(allmissing)).ravel()
    if alltrimmed.any():
        if alltrimmed.all():
            warnings.warn('No data remain after trimming')
        else:
            warnings.warn('No data remain in some columns after trimming')
    return m.squeeze()


if __name__ == '__main__':
    m, n = 1, 10
    # x = np.arange(1, n + 1).astype(float, copy=False)
    x = np.sin(np.linspace(0, 2 * np.pi, n)) + np.random.rand(n)
    percent = 50
    regular = x.mean(0)
    trimmed = trimmean(x, percent=percent)
    print(regular, trimmed)
    # fig, axs = pl.subplots(2, 1, sharex=True, sharey=True)
    # for ax, m, t in zip(axs.flat, [trimmed, regular], ['trimmed', 'regular']):
        # ax.plot(m)
        # ax.set_title(t)
        # pl.axis('tight')
    # pl.show()
