from pandas import Panel, Series, DataFrame
import numpy as np

try:
    import matplotlib.pyplot as plt
except (RuntimeError, ImportError):
    pass


from numba import jit


def _permute_axis(values, axis=0):
    return values.take(np.random.permutation(values.shape[axis]),
                       axis=axis)


def _shuffle_frame(self, m=1, axis=0):
    values = self.values

    if m == 1:
        return self._constructor(_permute_axis(values, axis),
                                 index=self.index, columns=self.columns)

    res = {}

    for i in xrange(m):
        res[i] = _permute_axis(values, axis)

    return Panel(res, major_axis=self.index, minor_axis=self.columns)

DataFrame.shuffle = _shuffle_frame


def _series_shuffle(self, m=1):
    construct = self._constructor

    if m == 1:
        inds = np.random.permutation(self.size)
        return construct(self.values.take(inds), self.index,
                         self.dtype, self.name)

    res = dict((i, construct(
                self.values.take(np.random.permutation(self.size)),
                self.index, self.dtype, self.name))
               for i in xrange(m))
    return DataFrame(res)


Series.shuffle = _series_shuffle


def pointwise_acceptance_cch(xc, M=1000, alpha=0.05, plot=False, ax=None):
    # lower and upper alphas and N's
    a_lower = alpha / 2
    a_upper = 1 - a_lower
    n_lower, n_upper = M * a_lower, M * a_upper

    # create M surrogates
    xcs = xc.shuffle(m=M)

    # empirical mean
    xcm = xcs.mean(1)

    # add in the original for sorting
    xcs.columns = np.arange(1, M + 1)
    xcs = xcs.join(Series(xc.values, xc.index, name=0)).sort_index(axis=1)

    # sort the surrogates
    srt_xcs = xcs.copy()
    srt_xcs.values.sort(axis=1)

    # define a and b for pointwise acceptance bands
    a = srt_xcs[n_lower]
    b = srt_xcs[n_upper]

    # compute the 'trimmed' mean and std
    rsm = srt_xcs.ix[:, :M - 1]
    nu = rsm.mean(1)
    s = rsm.std(1)

    # some sort of t-like statistic
    c_star = xcs.sub(nu, axis=0).div(s, axis=0)

    # compute the max/min and sort them
    c_max = c_star.max()
    c_max.values.sort()
    c_min = c_star.min()
    c_min.values.sort()

    # a_star and b_star for simultaneous acceptance bands
    at = c_min[n_lower]
    bt = c_max[n_upper]

    # put them back in original units
    a_star = at * s + nu
    b_star = bt * s + nu

    # plotting crap
    try:
        if plot:
            if ax is None:
                fig, ax = plt.subplots(1, 1, sharex=True, sharey=True)

            ind = a.index.values
            # ax.fill_between(ind, a_star.values - xcm.values,
            #                 b_star.values - xcm.values, alpha=0.6, color='k')
            ax.fill_between(ind, a.values - xcm.values,
                            b.values - xcm.values, alpha=0.3, color='k')

            bigxc = xc[(xc <= a) | (xc >= b)]
            smallxc = xc[(xc > a) & (xc < b)]
            lw = 2
            ax.vlines(bigxc.index.values, 0, bigxc.values, color='r', lw=lw)
            ax.vlines(smallxc.index.values, 0, smallxc.values, lw=lw)
            # ax.legend([r'Sig at $\alpha=%g$' % alpha])
            ax.set_xlabel(r'$\ell$')
            ax.set_ylabel(r'$\gamma(\ell)$')
    except (RuntimeError, NameError):
        pass

    d = {'a': a, 'mu': xcm, 'b': b, 'nu': nu, 's': s, 'a_star': a_star,
         'b_star': b_star}
    return DataFrame(d)
