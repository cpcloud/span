#!/usr/bin/env python

# spikedataframe.py ---

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


"""
Examples
--------
>>> import span
>>> tank = span.tdt.PandasTank('some/tank/file/folder')
>>> sp = tank.spikes
>>> assert isinstance(sp, span.tdt.SpikeDataFrame)
"""

import numbers
import types
import abc

from functools import partial
from itertools import product as iproduct

import numpy as np

from pandas import Series, DataFrame, MultiIndex

import span
from span.xcorr import xcorr
from span.utils import sem, samples_per_ms, clear_refrac


class SpikeDataFrameBase(DataFrame):
    __metaclass__ = abc.ABCMeta

    def __init__(self, *args, **kwargs):
        super(SpikeDataFrameBase, self).__init__(*args, **kwargs)

    @abc.abstractproperty
    def nchannels(self):
        pass

    @abc.abstractproperty
    def nsamples(self):
        pass

    @abc.abstractproperty
    def fs(self):
        pass

    @abc.abstractmethod
    def threshold(self, thresh):
        pass

    @abc.abstractmethod
    def clear_refrac(self, threshed, ms=2):
        pass

    @abc.abstractmethod
    def xcorr(self, *args, **kwargs):
        pass


class SpikeDataFrame(SpikeDataFrameBase):
    """Class encapsulting a Pandas DataFrame with extensions for analyzing
    spike train data.

    See the :class:`SpikeDataFrameBase` documentation for constructor details.
    """
    def __init__(self, *args, **kwargs):
        super(SpikeDataFrame, self).__init__(*args, **kwargs)

    def sem(self, *args, **kwargs):
        r"""Return the standard error of the mean of array along `axis`.

        Parameters
        ----------
        axis : int, optional
            The axis along which to compute the standard error of the mean.

        ddof : int, optional
            Delta degrees of freedom. 0 computes the sem using the population
            standard deviation, 1 computes the sem using the sample standard
            deviation.

        Returns
        -------
        sem : Series
            The standard error of the mean of the object along `axis`.

        Notes
        -----
        The standard error of the mean is defined as:

        .. math::

           \operatorname{sem}\left(\mathbf{x}\right)=\sqrt{\frac{\frac{1}{n -
           \textrm{ddof}}\sum_{i=1}^{n}\left(x_{i} -
           \bar{\mathbf{x}}\right)^{2}}{n}}

        where :math:`n` is the number of elements along the axis `axis`.
        """
        return self.apply(sem, *args, **kwargs)

    @property
    def nchannels(self):
        return self.shape[1]

    @property
    def nsamples(self):
        return self.shape[0]

    @property
    def fs(self):
        return 1e9 / self.index.freq.n

    def threshold(self, threshes):
        """Threshold spikes.

        Parameters
        ----------
        threshes : array_like

        Raises
        ------
        AssertionError
            * If `threshes` is not a scalar or a vector of length equal to the
              number of channels.

        Returns
        -------
        threshed : array_like
        """
        if np.isscalar(threshes):
            threshes = np.repeat(threshes, self.nchannels)

        assert threshes.size == 1 or threshes.size == self.nchannels, \
            'number of threshold values must be 1 (same for all channels) or '\
            '{0}, different threshold for each channel'.format(self.nchannels)

        is_neg = np.all(threshes < 0)
        cmpf = self.lt if is_neg else self.gt

        thr = threshes.item() if threshes.size == 1 else threshes
        threshes = Series(thr, index=self.columns)

        f = partial(cmpf, axis=1)

        return f(threshes)

    @property
    def _constructor(self):
        return type(self)

    def clear_refrac(self, threshed, ms=2):
        """Remove spikes from the refractory period of all channels.

        Parameters
        ----------
        threshed : array_like
            Array of ones and zeros.

        ms : float, optional, default 2
            The length of the refractory period in milliseconds.

        Raises
        ------
        AssertionError
            * If `ms` is not an instance of the ADT ``numbers.Integral``.
            * If `ms` is less than 0 or is not ``None``.

        Returns
        -------
        r : SpikeDataFrame
            The thresholded and refractory-period-cleared array of booleans
            indicating the sample point at which a spike was above threshold.
        """
        assert isinstance(ms, (numbers.Integral, types.NoneType)), \
            '"ms" must be an integer or None'
        assert ms >= 0 or ms is None, \
            'refractory period must be a nonnegative integer or None'

        clr = threshed

        if ms:
            # copy so we don't write over the values of threshed
            clr = clr.copy()

            # get the number of samples in ms milliseconds
            ms_fs = samples_per_ms(self.fs, ms)
            clear_refrac(clr.values, ms_fs)

        return clr

    def xcorr(self, binned, maxlags=None, detrend=span.utils.detrend_mean,
              scale_type='normalize', sortlevel='shank i', nan_auto=False,
              lag_name='lag'):
        """Compute the cross correlation of binned data.

        Parameters
        ----------
        binned : array_like
            Data of which to compute the cross-correlation.

        maxlags : int, optional
            Maximum number of lags to return from the cross correlation.
            Defaults to None and computes the full cross correlation.

        detrend : callable, optional
            Callable used to detrend. Defaults to
            :func:`span.utils.detrend_mean`.

        scale_type : str, optional
            Method of scaling. Defaults to ``'normalize'``.

        sortlevel : str, optional
            How to sort the index of the returned cross-correlation.
            Defaults to "shank i" so the the xcorrs are ordered by their
            physical ordering.

        nan_auto : bool, optional
            If ``True`` then the autocorrelation values will be ``NaN``.
            Defaults to ``False``.

        lag_name : str, optional
            Name to give to the lag index for plotting. Defaults to
            ``r'$\ell$'``.

        Raises
        ------
        AssertionError
           * If detrend is not a callable object
           * If scale_type is not a string or is not None

        ValueError
           * If sortlevel is not ``None`` and is not a string or a number in
             the list of level names or level indices.

        Returns
        -------
        xc : DataFrame or Series
            The cross correlation of all the columns of the data.

        See Also
        --------
        span.xcorr.xcorr
            General cross correlation function.

        SpikeDataFrame.bin
            Binning function.

        SpikeDataFrame.clear_refrac
            Clear the refractory period of a channel or array of channels.
        """
        assert callable(detrend) or detrend is None, ('detrend must be a '
                                                      'callable class or '
                                                      'function or None')
        assert isinstance(scale_type, basestring) or scale_type is None, \
            'scale_type must be a string or None'

        xc = xcorr(binned, maxlags=maxlags, detrend=detrend,
                   scale_type=scale_type)

        xc.columns = _create_xcorr_inds(binned.columns)

        if nan_auto:
            # slight hack for channel names
            xc0 = xc.ix[0]
            names = xc0.index.names
            chi_ind = names.index('channel i')
            chj_ind = names.index('channel j')

            select_func = lambda x: x[chi_ind] == x[chj_ind]
            xc.ix[0, xc0.select(select_func).index] = np.nan

        xc = xc.sortlevel(level=sortlevel, axis=1)
        xc.index.name = lag_name

        return xc

    def jitter(self, window=100, unit='ms'):
        index = self.index.values
        dt = index.dtype
        beg = np.floor(index.astype(int) / window)
        start = (window * beg).astype(dt, copy=False)
        td_unit = 'timedelta64[%s]' % unit
        shifted = start + (np.random.rand(self.nsamples) *
                           window).astype(td_unit, copy=False)
        return self._constructor(self.values, shifted,
                                 self.columns).sort_index()


# TODO: hack to make it so nans are allowed when creating indices
def _create_xcorr_inds(columns):
    ncols = len(columns)
    xr = xrange(ncols)
    prod = iproduct(xr, xr)
    inds = [columns[i] + columns[j] for i, j in prod]
    names = 'shank i', 'channel i', 'shank j', 'channel j'
    return MultiIndex.from_tuples(inds, names=names)
