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
:mod:`span.tdt.recording` is a module for encapsulating information about the
electrode array used in an experiment.

The single class :class:`ElectrodeMap` and the module level function
:func:`distance_map` are exported to allow the user to easily deal with the
computation of pairwise distance between electrodes given an electrode map.
:func:`distance_map` is more of a low-level function used by the
:class:`ElectrodeMap` class, but I chose to export it anyway for exploration
purposes. A few string constants are also exported but can be ignored.

The main features of this module are:
   * Encapsulation and visualization of electrode map structure
   * Ability to easily compute pairwise distance between electrodes

One thing that I think might be a useful generalization is to add a database
of electrode map structures from various vendors (NeuroNexus comes to mind).

An issue that might be considered a bug at worst or an inconvenience that
needs to be addressed is that if your electrodes are not numbered from 1 to
:math:`n`, the labels when computing the pairwise distance will be those of an
array that *is* numbered that way.
"""

import numbers

import numpy as np
from scipy.spatial.distance import squareform, pdist
from pandas import DataFrame, MultiIndex, Series, Index

from span.utils import ndtuples, create_repeating_multi_index


V_BAR = u'\u2502'
H_BAR = u'\u2500'
DOWN_AND_RIGHT_ARC = u'\u2570'
DOWN_AND_LEFT_ARC = u'\u256f'
MU = u'\u03bc'
F_SLASH = u'\u2572'
B_SLASH = u'\u2571'
DWARD_POINT = u'%s%s' % (F_SLASH, B_SLASH)
EMPTY_STRING = u''
SPACE = u' '
NEWLINE = u'\n'
TRI_DOWN = u'\u25bc'


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

            if np.logical_xor(within_shank, between_shank):
                raise ValueError('Shank measurements must be all or none if a '
                                 '2D electrode array is given')

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
        names = shank.name, channel.name
        inds = zip(shank, channel)
        inds.sort()
        return MultiIndex.from_tuples(inds, names=names)

    @property
    def original(self):
        df = DataFrame(self.__original, copy=True)
        df.index.name = 'channel'
        df.columns.name = 'shank'
        return df

    @property
    def _repr_pad(self):
        # force padding to be at least 2
        p = len(str(self.nchannel))

        if p == 1:
            p += 1

        return p

    def _build_unicode_sharps(self):
        nshank = self.nshank
        repr_pad = self._repr_pad

        spaces = SPACE * 2
        numspaces = repr_pad - 2

        single_sharp_base = [spaces.join([SPACE + F_SLASH + SPACE * numspaces +
                                         B_SLASH] * nshank)]

        if repr_pad == 3:
            tri_downs = [TRI_DOWN] * nshank
            joiner = SPACE * (repr_pad + 2)
            single_sharp_base.append(SPACE * 2 + joiner.join(tri_downs))
        elif repr_pad != 2:
            i = 2

            while numspaces > 0:
                numspaces -= 2
                spaces += SPACE
                s = SPACE * i + F_SLASH + SPACE * numspaces + B_SLASH
                single_sharp_base.append(spaces.join([s] * nshank))
                i += 1

                if numspaces == 1:
                    spaces += SPACE
                    s = (len(single_sharp_base) + 1) * SPACE + TRI_DOWN
                    single_sharp_base.append(spaces.join([s] * nshank))
                    break

        return NEWLINE.join(single_sharp_base)

    def _build_unicode_repr(self):
        orig = self.original
        columns = orig.columns
        nshank = self.nshank
        repr_pad = self._repr_pad

        # build the shank labels that sit on top of the array
        fmt_s = u'{0}{1:>{pad}}{2}'
        shank_row = SPACE.join(fmt_s.format(SPACE, shank, SPACE, pad=repr_pad)
                               for shank in columns)

        # now the horizontal bars underneath the shank labels
        fmt_s = u'{0}{1:{2}>{pad}}{3}'
        horz_bar_row = SPACE.join(fmt_s.format(*([H_BAR] * 4), pad=repr_pad)
                                  for shank in columns)

        # build the cells with just the vertical bars
        fmt_s = u'{0}{1:>{pad}}{2}'
        cells = []

        # for each row of channels fill in the channel number
        for _, row in orig.iterrows():
            cells.append(SPACE.join(fmt_s.format(V_BAR, channel, V_BAR,
                                                 pad=repr_pad)
                                    for channel in row))

        cell_s = NEWLINE.join(cells)

        # build the rounded out bottom
        _sb = ([DOWN_AND_RIGHT_ARC] + ([H_BAR] * repr_pad) +
               [DOWN_AND_LEFT_ARC])
        sb = EMPTY_STRING.join(_sb)
        shank_bottom = SPACE.join([sb] * nshank)

        # build the sharps
        sharps = self._build_unicode_sharps()
        #sharps = left_pad + (SPACE * numspaces).join([DWARD_POINT] * nshank)

        # make the within and between shank measurement strings
        ws, bs = self.within_shank, self.between_shank
        meas = ws, bs
        ses = map(str, meas)
        lengths = map(len, ses)
        max_s, min_s = max(lengths), min(lengths)
        s = (u'{newline} within shank: {ws}{mum:>{pad}}{newline}'
             u'between shank: {bs}{mum:>{pad}}')
        meas_s = s.format(newline=NEWLINE, ws=ws, mum=MU + 'm', bs=bs,
                          pad=max_s)

        # the final string
        components = (shank_row, horz_bar_row, cell_s, shank_bottom, sharps,
                      meas_s)
        return NEWLINE.join(components)

    def __unicode__(self):
        try:
            return self._build_unicode_repr()
        except:
            return unicode(self.original)

    def __bytes__(self):
        return unicode(self).encode('utf8', 'replace')

    __repr__ = __bytes__

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
