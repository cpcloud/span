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
``recording.py`` is a module for encapsulating information about the
electrode array(s) used in an experiment.
"""

from six.moves import zip, map

import numbers
import operator

import numpy as np
from pandas import DataFrame,  MultiIndex

from span.utils import ndtuples, compose
from scipy.spatial.distance import squareform, pdist


def distance_map(nshanks, electrodes_per_shank, within_shank, between_shank,
                 metric='wminkowski', p=2.0):
    """Create an electrode distance map.

    Parameters
    ----------
    nshanks, electrodes_per_shank : int

    within_shank, between_shank : float

    metric : str or callable, optional
        The distance measure to use to compute the distance between electrodes.

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
    assert nshanks >= 1, 'must have at least one shank, gave %i' % nshanks
    assert isinstance(nshanks, numbers.Integral), 'nshanks must be an integer'
    assert electrodes_per_shank >= 1, \
        'must have at least one electrode per shank'
    assert isinstance(electrodes_per_shank, numbers.Integral), \
        '"electrodes_per_shank" must be an integer'
    assert isinstance(metric, basestring) or callable(metric), \
        '"metric" must be a string or callable'
    assert isinstance(p, numbers.Real) and p > 0, \
        '"p" must be a positive real number'

    locs = ndtuples(electrodes_per_shank, nshanks)

    if locs.ndim == 1:
        locs = locs[:, np.newaxis]

    w = np.asanyarray((between_shank, within_shank), dtype=float)

    return squareform(pdist(locs, metric=metric, p=p, w=w))


class ElectrodeMap1D(Series):
    """
    """
    def __init__(self, *args, **kwargs):
        super(ElectrodeMap1D, self).__init__(*args, **kwargs)

    @property
    def _constructor(self):
        return ElectrodeMap1D

    @property
    def nshanks(self):
        return 1

    @property
    def nchans(self):
        return self.size

    def distance_map(self, within, between=0, metric='wminkowski', p=2.0):
        pass


class ElectrodeMap2D(DataFrame):
    """Encapsulate the geometry of the electrode map used in a recording.

    Parameters
    ----------
    map_ : array_like
        The electrode configuration. Can have an arbitrary integer base.

    Attributes
    ----------
    nshanks : int
        Number of shanks.

    nchans : int
        Number of channels.
    """
    def __init__(self, map_):
        map_ = np.atleast_1d(np.asanyarray(map_).squeeze())

        try:
            m, n = map_.shape
            s = np.repeat(np.arange(n), m)
        except ValueError:
            m, = map_.shape
            s = np.ones(m, dtype=int)

        data = {'channel': map_.ravel(), 'shank': s}
        df = DataFrame(data).sort('shank').reset_index(drop=True)
        df.index = df.pop('shank')

        super(ElectrodeMap, self).__init__(df.sort())

    @property
    def nshanks(self):
        return self.index.unique().size

    @property
    def nchans(self):
        return self.channel.unique().size

    def distance_map(self, within, between, metric='wminkowski', p=2.0):
        r"""Create a distance map from the current electrode configuration.

        This method performs some type checking on its arguments.

        Parameters
        ----------
        within, between : number
            `between` is the distance between shanks and `within` is the
            distance between electrodes on any given shank.

        metric : str or callable, optional
            Metric to use to calculate the distance between electrodes/shanks.
            Defaults to a weighted Minkowski distance

        p : numbers.Real, optional
            The :math:`p` of the norm to use. Defaults to 2.0 for weighted
            Euclidean distance.

        Notes
        -----
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
            * If `within` is not an instance of ``numbers.Real``
            * If `between` is not an instance of ``numbers.Real``
            * If `p` is not an instance of ``numbers.Real``
            * If metric is not an instance of ``basestring`` or a callable

        Returns
        -------
        df : DataFrame
            A dataframe with pairwise distances between electrodes, indexed by
            channel, shank.
        """
        assert isinstance(within, numbers.Real) and within > 0, \
            '"within" must be a positive real number'
        assert isinstance(between, numbers.Real) and between > 0, \
            '"between" must be a positive real number'
        assert isinstance(metric, basestring) or callable(metric), \
            '"metric" must be a callable object or a string'
        assert isinstance(p, numbers.Real) and p > 0, \
            'p must be a real number greater than 0'

        dm = distance_map(self.nshanks, self.nchans / self.nshanks, within,
                          between, metric=metric, p=p)
        s = self.sort()
        cols = s.index, s.channel

        values_getter = operator.attrgetter('values')
        cols = tuple(map(values_getter, cols))
        names = 'shank', 'channel'

        # index = _label_maker('i', names)
        # columns = _label_maker('j', names)
        # df = DataFrame(dm, index=index, columns=columns)

        columns = _label_maker('i', names)
        index = _label_maker('j', names)
        df = DataFrame(dm, index=index, columns=columns)

        nnames = len(names)
        ninds = len(index)
        nlevels = nnames * ninds

        zipped = zip(xrange(nnames), xrange(nnames, nlevels))
        reordering = tuple(reduce(operator.add, zipped))

        # for _ in xrange(nnames - 1):
        #     s = s.stack(0)

        # s.name = r'$d\left(i, j\right)$'

        # return s.reorder_levels(reordering)
        return dm

        return s.reorder_levels(reordering)
        # return df
