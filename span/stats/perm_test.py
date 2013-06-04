from pandas import Panel, Series, DataFrame
import numpy as np
import numpy.random as npr

from six.moves import xrange

#try:
    #import matplotlib.pyplot as plt
#except (RuntimeError, ImportError):
    #pass


def _permute_axis(values, axis=0):
    if not axis:
        return np.random.permutation(values)

    return values.take(np.random.permutation(values.shape[axis]), axis)


def _shuffle_frame(self, n=1, axis=0):
    values = self.values.copy()

    if n == 1:
        return self._constructor(_permute_axis(values, axis), self.index,
                                 self.columns)

    res = dict((i, _permute_axis(values, axis)) for i in xrange(n))
    return Panel(res, major_axis=self.index, minor_axis=self.columns)


DataFrame.shuffle = _shuffle_frame


def _shuffle_series(self, n=1):
    values = self.values.copy()

    if n == 1:
        return self._constructor(_permute_axis(values), self.index, self.dtype,
                                 self.name)

    res = dict((i, _permute_axis(values)) for i in xrange(n))
    return DataFrame(res, self.index)


Series.shuffle = _shuffle_series


def cch_perm(xci, M=1000, alpha=0.05, plot=False, ax=None):
    # lower and upper alphas and N's
    a_lower = alpha / 2
    a_upper = 1 - a_lower
    n_lower = M * a_lower
    n_upper = M * a_upper

    # create M surrogates by shuffling the lags
    xcs = xci.shuffle(n=M)

    # empirical mean across shuffles
    xcm = xcs.mean(1)

    # add in the original for sorting
    xcs.columns = np.arange(1, M + 1)
    xcs[0] = xci.values
    xcs.sort_index(axis=1, inplace=True)

    # compute a p-value for the lag 0 distribution
    p = np.mean(xcs.ix[0] >= xci.ix[0])

    # sort the surrogates (remember this includes the ORIGINAL ccg)
    srt_xcs = xcs.copy()
    srt_xcs.values.sort(axis=1)

    # define a and b for pointwise acceptance bands
    a = srt_xcs[n_lower]
    b = srt_xcs[n_upper]

    # compute the 'trimmed' mean and std
    rsm = srt_xcs.ix[:, 1:M - 1]  # inclusive indices here
    nu = rsm.mean(1)
    s = rsm.std(1)

    # a t-like statistic
    c_star = xcs.sub(nu, axis=0).div(s, axis=0)

    # compute the max/min and sort them
    c_max = c_star.max()
    c_max.values.sort()

    c_min = c_star.min()
    c_min.values.sort()

    # a_star and b_star for simultaneous acceptance bands
    at = c_min[n_lower]
    bt = c_max[n_upper]

    c_star_0 = c_star[0]
    sig = (c_star_0 < at) | (c_star_0 > bt)

    # put them back in original units
    a_star = at * s + nu
    b_star = bt * s + nu

    # plotting crap
    #try:
        #if plot:
            #if ax is None:
                #fig, (ax1, ax2) = plt.subplots(1, 2)

            #lw = 2
            #xcv = xci - xcm
            #lag0 = xcv.ix[0]

            #ind = a.index.values
            #lower, upper = a_star - xcm, b_star - xcm
            #ax1.fill_between(ind, lower, upper, alpha=0.3, color='k')
            #ax1.fill_between(ind, a - xcm, b - xcm, alpha=0.4, color='k')

            #ax1.vlines(ind, 0, xcv, lw=lw)
            #ax1.set_xlabel(r'$\ell$', fontsize=15)
            #ax1.set_ylabel(r'$\gamma(\ell)$', fontsize=15)
            #ax1.set_ylim((lower.min(), upper.max()))

            #xcs.ix[0].hist(ax=ax2, bins=20)
            #ax2.axvline(lag0, c='r', lw=lw)
            #ax2.set_xlabel(r'$\gamma(0)$', fontsize=15)
            #ax2.set_ylabel('Count', fontsize=15)
            #fig.tight_layout()
    #except (RuntimeError, NameError):
        #pass

    d = {'a': a, 'mu': xcm, 'b': b, 'nu': nu, 's': s, 'a_star': a_star,
         'b_star': b_star}
    return DataFrame(d), c_star, sig, p
