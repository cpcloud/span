#!/usr/bin/env python
# encoding: utf-8
"""
Module for reading TDT (Tucker-Davis Technologies) Tank files.
"""

from future_builtins import filter

import os
import glob

import span.tdt.tank
import span.utils


def get_tank_names(path=os.path.expanduser(os.path.join('~', 'xcorr_data')),
                   key=span.utils.dirsize):
    """Get the names of the tanks on the current machine.

    Parameters
    ----------
    path : str, optional
    key : callable, optional

    Returns
    -------
    filtered : list
    """
    if not os.path.exists(path):
        raise OSError('{} does not exist'.format(path))
    filtered = list(filter(os.path.isdir, glob.glob(os.path.join(path, '*'))))
    filtered.sort(key=key)
    return filtered


def profile_spikes(pct_stats=0.05, sortby='time'):
    """Profile the reading of TEV files.

    Parameters
    ----------
    pct_stats : float, optional
    sortby : str, optional


    """
    import cProfile as profile
    import pstats
    import tempfile

    fns = get_tank_names()

    for f in fns[-1:]:
        fn = os.path.join(f, os.path.basename(f))
        with tempfile.NamedTemporaryFile(mode='w+') as stats_file:
            stats_fn = stats_file.name
            profile.run('sp = tank.PandasTank("%s").spikes' % fn, stats_fn)
            p = pstats.Stats(stats_fn)
        p.strip_dirs().sort_stats(sortby).print_stats(pct_stats)




if __name__ == '__main__':
    fnames = get_tank_names()
    ind = 0
    fname = fnames[ind]
    fn_small = os.path.join(fname, os.path.basename(fname))
    t = span.tdt.tank.PandasTank(fn_small)
    sp = t.spikes
