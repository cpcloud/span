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
Example
-------
>>> import span
>>> tank = span.TdtTank('basename/of/some/tank/file')
>>> sp = tank.spik
>>> assert isinstance(sp, span.SpikeDataFrame)
"""
import abc
import functools
import numbers
import types
import warnings

import numpy as np
from pandas import Series, DataFrame, DatetimeIndex
from span.utils import samples_per_ms, clear_refrac, LOCAL_TZ
from span.xcorr import xcorr as _xcorr
import six


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
    def threshold(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def clear_refrac(self, *args, **kwargs):
        pass


class SpikeDataFrame(SpikeDataFrameBase):
    """Class encapsulting a Pandas DataFrame with extensions for analyzing
    spike train data.

    See the pandas DataFrame documentation for constructor details.
    """
    def __init__(self, *args, **kwargs):
        self.super.__init__(*args, **kwargs)
        self.isclean = False

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

        assert threshes.size == self.nchannels, \
            'number of threshold values must be 1 (same for all channels) or '\
            '{0}, different threshold for each channel'.format(self.nchannels)

        is_neg = np.all(threshes < 0)
        cmpf = self.lt if is_neg else self.gt

        thr = threshes.item() if threshes.size == 1 else threshes
        threshes = Series(thr, index=self.columns)

        f = functools.partial(cmpf, axis=1)

        return f(threshes)

    @property
    def _constructor(self):
        return type(self)

    def clear_refrac(self, ms=2, inplace=False):
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

        Notes
        -----
        This method DOES NOT modify the object inplace by default.
        """
        assert isinstance(ms, (np.integer, numbers.Integral,
                               types.NoneType)), \
            '"ms" must be an integer or None'
        assert ms >= 0 or ms is None, \
            'refractory period must be a nonnegative integer or None'

        if not ms:
            return None if inplace else self
        else:
            ms_fs = samples_per_ms(self.fs, ms)

            if inplace:
                clear_refrac(self.values, ms_fs)
            else:
                v = self.copy()
                clear_refrac(v.values, ms_fs)
                return v

    @classmethod
    def xcorr(cls, binned, maxlags=None, detrend=None, scale_type=None,
              sortlevel='shank i', nan_auto=False):
        """Compute the cross correlation of binned data.

        Parameters
        ----------
        binned : array_like
            Data of which to compute the cross-correlation.

        maxlags : int, optional
            Maximum number of lags to return from the cross correlation.
            Defaults to None and computes the full cross correlation.

        detrend : callable or None, optional
            Callable used to detrend. Defaults to ``None``

        scale_type : str, optional
            Method of scaling. Defaults to ``None``.

        sortlevel : str, optional
            How to sort the index of the returned cross-correlation.
            Defaults to "shank i" so the the xcorrs are ordered by their
            physical ordering.

        nan_auto : bool, optional
            If ``True`` then the autocorrelation values will be ``NaN``.
            Defaults to ``False``.

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
        xc : DataFrame
            The cross correlation of all the columns of the data, indexed by
            lags and columned by channel pair.

        See Also
        --------
        span.xcorr.xcorr
            General cross correlation function.

        SpikeDataFrame.clear_refrac
            Clear the refractory period of a channel or array of channels.
        """
        assert callable(detrend) or detrend is None, ('detrend must be a '
                                                      'callable class or '
                                                      'function or None')
        assert isinstance(scale_type, six.string_types + (types.NoneType,)), \
            'scale_type must be a string or None'

        xc = _xcorr(binned, maxlags=maxlags, detrend=detrend,
                    scale_type=scale_type)

        if nan_auto:
            # HACK for channel names
            xc0 = xc.ix[0]
            names = xc0.index.names
            chi_ind = names.index('channel i')
            chj_ind = names.index('channel j')

            selector = lambda x: x[chi_ind] == x[chj_ind]
            xc.ix[0, xc0.select(selector).index] = np.nan

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', FutureWarning)
            xc.sortlevel(level=sortlevel, axis=1, inplace=True)

        return xc

    def basic_jitter(self, window=100, unit='ms'):
        """Basic jitter samples by some window in units of `unit`.

        Parameters
        ----------
        window : int, optional
            The size of the jitter window.
        unit : str, optional
            The time units of the jitter window.

        Returns
        -------
        df : SpikeDataFrame
        """
        new_index = self._basic_jitter_reindex(window, unit)
        df = self._constructor(self.values, new_index, self.columns)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', FutureWarning)
            df.sort_index(inplace=True)

        return df

    def _basic_jitter_reindex(self, window, unit):
        # raw datetime ndarray
        index = self.index.values

        # datetime units
        dt = index.dtype

        # start of the window-length window
        beg = np.floor(index.astype(int, copy=False) / window)
        start = (window * beg).astype(dt, copy=False)

        # timedelta unit
        td_unit = 'timedelta64[%s]' % unit

        # shift from beginning of jitter window by U * window
        rt = np.random.rand(self.nsamples) * window
        rand_time = rt.astype(td_unit, copy=False)
        shifted = start + rand_time

        return DatetimeIndex(shifted, tz=LOCAL_TZ)

    ## reimplement methods that pandas dataframe doesn't correctly construct
    #  after calling
    @property
    def super(self):
        return super(SpikeDataFrame, self)

    def _call_super_method(self, method_name, *args, **kwargs):
        method = getattr(self.super, method_name)
        return self._constructor(method(*args, **kwargs))

    def dot(self, *args, **kwargs):
        return self._call_super_method('dot', *args, **kwargs)

    def sort_index(self, *args, **kwargs):
        return self._call_super_method('sort_index', *args, **kwargs)


spike_xcorr = SpikeDataFrame.xcorr
