from nose.tools import nottest

import numpy as np
from numpy.random import rand, randint

from span.utils import bin_data, cast


def test_bin_data_none():
    x = rand(randint(25, 200), randint(10, 20)) > 0.5
    binsize = randint(10, max(x.shape))
    bins = cast(np.r_[:binsize:x.shape[0]], long)
    binned = bin_data(x, bins)
    assert binned.shape == (bins.shape[0] - 1, x.shape[1])
    assert binned.dtype == np.int64
