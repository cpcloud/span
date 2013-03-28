#!/usr/bin/env python

# recording.py ---

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
A module for encapsulating information about the electrode array used
in an experiment.
"""

import numbers

import numpy as np
from scipy.spatial.distance import squareform, pdist

from pandas import DataFrame, MultiIndex, Series, Index

from span.utils import ndtuples, create_repeating_multi_index


def distance_map(nshanks, electrodes_per_shank, within_shank, between_shank,
                 metric='wminkowski', p=2.0):
    """Create an electrode distance map.

    Parameters
    ----------
    nshanks, electrodes_per_shank : int

    within_shank, between_shank : float

    metric : str or callable, optional
        The distance measure to use to compute the distance between
        electrodes.

    p : number, optional
        See :py:mod:`scipy.spatial.distance` for more details.

    Raises
    ------
    AssertionError
        * If `nshanks` < 1 or `nshanks` is not an integer
        * If neither of those same conditions holds for `electrodes_per_shank`
        * If `metric` is not a string or callable or
        * If `p` is not an instance of ``numbers.Real`` and ``<= 0``

    Returns
    -------
    dists : DataFrame
        DataFrame of pairwise distances between electrodes.
    """
    assert nshanks >= 1, 'must have at least one shank'
    assert isinstance(nshanks, (numbers.Integral, np.integer)), \
        'nshanks must be an integer'
    assert electrodes_per_shank >= 1, \
        'must have at least one electrode per shank'
    assert isinstance(electrodes_per_shank, (numbers.Integral, np.integer)), \
        '"electrodes_per_shank" must be an integer'
    assert isinstance(metric, basestring) or callable(metric), \
        '"metric" must be a string or callable'
    assert isinstance(p, (numbers.Real, np.floating)), \
        '"p" must be a real number'
    assert p > 0, '"p" must be a positive number'

    args = [electrodes_per_shank]

    if nshanks > 1:
        args += [nshanks]

    locs = ndtuples(*args)

    if nshanks == 1:
        locs = np.column_stack((locs, locs))

    w = np.asanyarray((between_shank, within_shank), dtype=np.float64)
    return squareform(pdist(locs, metric=metric, p=p, w=w))


class ElectrodeMap(object):
    """Encapsulate the geometry of an electrode map."""

    __slots__ = ('__original', '__nshank', '__channel', '__shank',
                 'within_shank', 'between_shank', '__nchannel')

    def __init__(self, map_, within_shank=None, between_shank=None):
        super(ElectrodeMap, self).__init__()

        try:
            channels_per_shank, self.__nshank = map_.shape
        except ValueError:  # passed a vector
            channels_per_shank, = map_.shape
            self.__nshank = 1

            assert between_shank is None or between_shank == 0, \
                ('between_shank must be 0 or None for single shank electrode '
                 'maps')

            if between_shank is None:
                between_shank = 0

        self.__nchannel = map_.size

        self.__channel = Index(map_.ravel(order='K'), name='channel')
        self.__shank = Index(np.repeat(np.arange(self.nshank),
                                       channels_per_shank), name='shank')
        self.within_shank = within_shank
        self.between_shank = between_shank
        self.__original = np.column_stack([np.squeeze(map_.copy())])

    def _get_labels(self, name):
        return self.index.labels[self.index.names.index(name)]

    @property
    def channel(self):
        return self._get_labels(self.__channel.name)

    @property
    def shank(self):
        return self._get_labels(self.__shank.name)

    @property
    def nshank(self):
        return self.__nshank

    @property
    def nchannel(self):
        return self.__nchannel

    @property
    def raw(self):
        return DataFrame(self.shank, index=self.channel, columns=['shank'])

    @property
    def index(self):
        shank, channel = self.__shank, self.__channel
        names = [shank.name, channel.name]
        inds = zip(shank, channel)
        inds.sort()
        return MultiIndex.from_tuples(inds, names=names)

    @property
    def original(self):
        df = DataFrame(self.__original, copy=True)
        df.index.name = 'channel'
        df.columns.name = 'shank'
        return df

    def _build_unicode_map(self):
        channels_per_shank = self.nchannel // self.nshank
        _bars = [u'\u2502'] * channels_per_shank
        _top = [u' '] * channels_per_shank
        _mid = [u'\u2500'] * channels_per_shank
        bars = [_top * 2] + ([_bars * 2] * self.nshank)

        joiner = lambda x, y: [u'{0}{1:>2}{2}'.format(xi, ch, xj)
                               for xi, xj, ch in zip(x[::2], x[1::2], y)]
        joiner_nopad = lambda x, y: [u'{0}{1:\u2500>2}{2}'.format(xi, ch, xj)
                                     for xi, xj, ch in zip(x[::2], x[1::2],
                                                           y)]
        _bars = map(joiner, bars[1:], self.original.values)
        bars = map(joiner, bars[:1], [xrange(self.nshank)])
        bars += map(joiner_nopad, [_mid * 2],
                    [[u'\u2500'] * self.nshank]) + _bars
        s = u'\n'.join(map(lambda x: u' '.join(x), bars))
        bottom = (u' ' * 3).join([u'\u2572\u2571'] * self.nshank)
        btop = u' '.join([u'\u2570\u2500\u2500\u256f'] * self.nshank)
        s += u'\n' + btop + u'\n ' + bottom
        mu = u'\u03bc'
        s += u'\n\nwthn: {0} {2}m\nbtwn: {1} {2}m'.format(self.within_shank,
                                                          self.between_shank,
                                                          mu)
        return s

    def __unicode__(self):
        try:
            return self._build_unicode_map()
        except:
            return unicode(self.original)

    def __bytes__(self):
        return unicode(self).encode('utf8', 'replace')

    __repr__ = __str__ = __bytes__

    @property
    def _electrodes_per_shank(self):
        return self.nchannel // self.nshank

    def distance_map(self, metric='wminkowski', p=2.0):
        r"""Create a distance map from the current electrode configuration.

        Parameters
        ----------
        metric : str or callable, optional
            Metric to use to calculate the distance between electrodes/shanks.
            Defaults to a weighted Minkowski distance

        p : numbers.Real, optional
            The :math:`p` of the norm to use. Defaults to 2.0 for weighted
            Euclidean distance.

        Notes
        -----
        This method performs some type checking on its arguments.

        The default `metric` of ``'wminkowski'`` and the default `p` of ``2.0``
        combine to give a weighted Euclidean distance metric. The weighted
        Minkowski distance between two points
        :math:`\mathbf{x},\mathbf{y}\in\mathbb{R}^{n}`, and a weight vector
        :math:`\mathbf{w}\in\mathbb{R}^{n}` is given by

            .. math::
               \left(\sum_{i=1}^{n}w_i\left|x_i-y_i\right|^{p}\right)^{1/p}

        Raises
        ------
        AssertionError
            * If `p` is not an instance of ``numbers.Real`` or
              ``numpy.floating``
            * If metric is not an instance of ``basestring`` or a callable

        Returns
        -------
        df : DataFrame
            A dataframe with pairwise distances between electrodes, indexed by
            channel, shank.
        """
        assert isinstance(metric, basestring) or callable(metric), \
            '"metric" must be a callable object or a string'
        assert isinstance(p, (numbers.Real, np.floating)) and p > 0, \
            'p must be a real number greater than 0'

        # import ipdb; ipdb.set_trace()
        dm = distance_map(self.nshank, self._electrodes_per_shank,
                          self.within_shank, self.between_shank,
                          metric=metric, p=p)
        mi = MultiIndex.from_arrays(np.flipud(self.raw.reset_index().values.T),
                                    names=('shank', 'channel'))
        rmi = create_repeating_multi_index(mi)

        return Series(dm.ravel(), index=rmi, name='distance').sort_index()
