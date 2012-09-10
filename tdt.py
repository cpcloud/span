#!/usr/bin/env python
# encoding: utf-8
"""
Module for reading TDT (Tucker-Davis Technologies) Tank files.
"""

import os
import sys
import glob

import numpy as np
import pandas as pd

from .spikeglobals import *
from . import tank


def get_tank_names(path=os.path.expanduser(os.path.join('~', 'xcorr_data'))):
    """Get the names of the tanks on the current machine.

    Parameters
    ----------
    path : str, optional
    """
    if not os.path.exists(path):
        raise OSError('{} does not exist'.format(path))
    fns = glob.glob(os.path.join(path, '*'))
    fns = np.fromiter(filter(os.path.isdir, fns), dtype=str)
    tevs = glob.glob(os.path.join(globs, '**', '*%stev' % os.extsep))
    tevsize = np.fromiter(map(os.path.getsize, tevs), dtype=int)
    inds = np.argsort(tevsize)
    return fns[::-1].tolist()


def profile_spikes(pct_stats=0.05, sortby='time'):
    """Profile the reading of TEV files.
    """
    import cProfile as profile
    import pstats
    import tempfile

    fns = get_tank_names()

    # sort by size
    fns.sort(key=lambda x: os.path.getsize(x))

    for f in fns[-1:]:
        fn = os.path.join(f, os.path.basename(f))
        with tempfile.NamedTemporaryFile(mode='w+') as stats_file:
            stats_fn = stats_file.name
            profile.run('sp = tank.PandasTank("%s").spikes' % fn, stats_fn)
            p = pstats.Stats(stats_fn)
        p.strip_dirs().sort_stats(sortby).print_stats(pct_stats)
    return fns


if __name__ == '__main__':
    # fns = profile_spikes()
    fns = get_tank_names()
    fns.sort(key=lambda x: os.path.getsize(x))
    ind = -2
    fn = fns[ind]
    fn_small = os.path.join(fn, os.path.basename(fn))
    t = tank.PandasTank(fn_small)
    sp = t.spikes
    raw = sp.raw
    # thr = spikes.threshold(3e-5).astype(float)
    # thr.values[thr.values == 0] = np.nan
