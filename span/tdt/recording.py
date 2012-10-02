"""Module for meta data about the recording."""

from future_builtins import map

import operator

import numpy as np
from pandas import Series, DataFrame, Int64Index, MultiIndex

from span.utils import ndtuples, fractional
from scipy.spatial.distance import squareform, pdist


def distance_map(nshanks, electrodes_per_shank, between_shank, within_shank,
                 metric='wminkowski', p=2.0):
    """Create an electrode distance map.

    Parameters
    ----------
    nshanks, electrodes_per_shank : int

    between_shank, within_shank : float

    metric : str, optional
        The distance measure to use to compute the distance between electrodes.

    p : number, optional
        See scipy.spatial.distance for more details here.

    Returns
    -------
    dists : DataFrame
        DataFrame of pairwise distances between electrodes.
    """
    assert nshanks >= 1 and not fractional(nshanks), \
        'must have at least one shank'
    assert electrodes_per_shank >= 1 and not fractional(electrodes_per_shank), \
        'must have at least one electrode per shank'
    locations = ndtuples(electrodes_per_shank, nshanks)
    weights = np.asanyarray((between_shank, within_shank), dtype=float)
    return squareform(pdist(locations, metric=metric, p=p, w=weights))


class ElectrodeMap(DataFrame):
    """Encapsulate the geometry of the electrode map used in a recording.

    Parameters
    ----------
    map_ : array_like
        The electrode configuration.

    order : None or str, optional
        If there is a topography to the are that was recorded from, indicate here
        by "lm". Defaults to None.

    base_index : int, optional
        The number to start the channel indexing from. Defaults to 0 for ease of
        use in Python.

    Attributes
    ----------
    nshanks : int
        Number of shanks.

    nchans : int
        Total number of channels.
    """
    def __init__(self, map_, order=None, base_index=0):
        map_ = np.asanyarray(map_).squeeze()
        mm = map_.min()

        if mm != base_index:
            v = -1 if mm > base_index else 1
            while mm != base_index:
                map_ += v
                mm = map_.min()

        try:
            m, n = map_.shape
            s = np.repeat(np.arange(n), m)
        except ValueError:
            m, = map_.shape
            s = np.ones(m, dtype=int)

        data = {'channel': map_.ravel(), 'shank': s}

        if order is not None:
            assert map_.ndim == 2, 'map_ must be 2D if there is a shank order'
            assert order is not None, 'if "side" given "order" cannot be None'
            assert order in ('lm', 'ml'), \
                'order must be "lm" (lateral to medial) or "ml" (medial to ' \
                'lateral)'
            data['side'] = np.repeat(tuple(order), map_.size / 2)

        df = DataFrame(data).sort('channel').reset_index(drop=True)
        df.index = df.pop('channel')

        super(ElectrodeMap, self).__init__(df)
            
    @property
    def nshanks(self): return self.shank.nunique()

    @property
    def nchans(self): return Series(self.index).nunique()

    def distance_map(self, between_shank, within_shank):
        """Create a distance map from the current electrode configuration.

        Parameters
        ----------
        between_shank, within_shank : number
            `between_shank` is the distance between shanks and `within_shank` is
            the distance between electrodes on any given shank.

        Returns
        -------
        df : DataFrame
            A dataframe with pairwise distances between electrodes.
        """
        dm = distance_map(self.nshanks, self.shank.nunique(), between_shank,
                          within_shank)
        s = self.sort('shank')
        cols = s.shank, s.index

        if hasattr(self, 'side'):
            cols += s.side,

        cols = tuple(map(operator.attrgetter('values'), cols))
        index = MultiIndex.from_arrays(cols, names=('shank', 'channel', 'side'))
        return DataFrame(dm, index=index, columns=index)

    def show(self):
        """Print out the electrode configuration with 1 based indexing."""
        values = self.values.copy().T
        values[0] += 1
        index = Int64Index(self.index.values + 1, name='channel')
        df = DataFrame({'shank': values[0], 'side': values[1]}, index=index)
        print df
