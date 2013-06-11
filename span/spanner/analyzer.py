import re
import warnings
import os

import numpy as np
import pandas as pd

import span
from span import ElectrodeMap, NeuroNexusMap, TdtTank

from span.spanner.command import SpanCommand
from span.spanner.utils import error


def _colon_to_slice(c):
    splitter = re.compile(r'\s*:\s*')
    start, stop = splitter.split(c)
    return slice(int(start), int(stop))


def _parse_artifact_ranges(s):
    splitter = re.compile(r'\s*,\s*')
    split = splitter.split(s)
    return [_colon_to_slice(spl) for spl in split]


def _get_xcorr(sp, threshold, sd, binsize, how, firing_rate_threshold, maxlags,
               which_lag, refractory_period, nan_auto, detrend, scale_type):
    thr = sp.threshold(threshold * sd)
    thr.clear_refrac(refractory_period, inplace=True)

    binned = thr.resample(binsize, how=how)
    binned.ix[:, binned.mean() < firing_rate_threshold] = np.nan

    xc = thr.xcorr(binned, maxlags=maxlags, detrend=getattr(span, 'detrend_' +
                                                            detrend),
                   scale_type=scale_type, nan_auto=nan_auto).ix[which_lag]
    xc.name = threshold
    return xc


def _get_xcorr_many_threshes(sp, threshes, sd, binsize, how,
                             firing_rate_threshold, maxlags, which_lag,
                             refractory_period, nan_auto, detrend, scale_type,
                             distance_map):
    xcs = pd.concat([_get_xcorr(sp, thresh, sd, binsize, how,
                                firing_rate_threshold, maxlags, which_lag,
                                refractory_period, nan_auto, detrend,
                                scale_type) for thresh in threshes], axis=1)
    xcs['distance'] = distance_map
    xcs.sort('distance', inplace=True)
    xcs = xcs.dropna(axis=(0, 1), how='all')
    xcs.drop_duplicates(inplace=True)
    return xcs


def _get_cutoff(df, thresh, dist_key='distance'):
    f = lambda x: pd.isnull(x).sum()
    cols = df.columns
    null_count_diff = df[cols[cols != dist_key]].apply(f).diff()
    cutoff = np.nan
    for ind, el in null_count_diff.iteritems():
        if np.floor(np.log10(el)) >= thresh:
            cutoff = ind
            break

    if pd.isnull(cutoff):
        return null_count_diff[null_count_diff > 0].index[0]
    return cutoff


def _trim_concatted(xcs_df, ord_mag_thresh=2, dist_key='distance'):
    cutoff = _get_cutoff(xcs_df, ord_mag_thresh)
    df = xcs_df.ix[:, xcs_df.columns >= cutoff].sort_index(axis=1)
    cols = df.columns
    subset = cols[cols != dist_key]
    return df.dropna(axis=0, how='all', subset=subset).dropna(axis=1,
                                                              how='all')


def _concat(xcs, threshes):
    xc_all = xcs.copy()
    xc_all.distance /= xc_all.distance.max()
    dist_key = 'distance'
    subset = xc_all.columns[xc_all.columns != dist_key]
    xc_df = xc_all.dropna(axis=0, how='all', subset=subset)
    return xc_df


def compute_xcorr(args):
    filename = args.filename

    # make a tank
    em = ElectrodeMap(NeuroNexusMap.values, args.within_shank,
                      args.between_shank)
    tank = TdtTank(_maybe_get_common_data_path(filename, COMMON_DATA_PATH), em)

    # get the raw data
    spikes = tank.spik

    if args.remove_first_pc:
        span.remove_first_pc(spikes)

    # get the threshes
    threshes = np.linspace(args.min_threshold, args.max_threshold,
                           args.num_thresholds)

    # compute a few different thresholds
    sd = spikes.std()

    # compute the cross correlation at each threshold
    xcs = _get_xcorr_many_threshes(spikes, threshes, sd, args.bin_size,
                                   args.bin_method, args.firing_rate_threshold,
                                   args.max_lags, args.which_lag,
                                   args.refractory_period, not args.keep_auto,
                                   args.detrend, args.scale_type,
                                   em.distance_map())
    # concat all xcorrs
    xcs_df = _concat(xcs, threshes)

    # remove the data that is useless
    trimmed = _trim_concatted(xcs_df)

    return trimmed, tank.age, tank.site


def show_xcorr(args):
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import ImageGrid
    try:
        from bottleneck import nanmax, nanmin
    except ImportError:
        from numpy import nanmax, nanmin
    trimmed, age, site = compute_xcorr(args)

    vmax = nanmax(trimmed.values)
    vmin = nanmin(trimmed.values)

    fig = plt.figure(figsize=(16, 9))
    grid = ImageGrid(fig, 111, nrows_ncols=(1, 1), direction='row',
                     axes_pad=0.05, add_all=True, label_mode='1',
                     share_all=False, cbar_location='right',
                     cbar_mode='single', cbar_size='10%', cbar_pad=0.05)
    if args.sort_by == 'shank':
        trimmed.sortlevel('shank i', inplace=True)
        trimmed.sortlevel('shank j', inplace=True)
    elif args.sort_by == 'distance':
        trimmed.sortlevel('distance', inplace=True)

    ax = grid[0]
    trimmed.set_index(['distance'], append=True, inplace=True, drop=True)
    im = ax.imshow(trimmed.values, interpolation='none', aspect='auto',
                   vmax=vmax, vmin=vmin)
    m, n = trimmed.shape

    ax.set_xticks(np.arange(n))
    ax.set_xticklabels(map('{0:.1f}'.format,
                           trimmed.columns.values.astype(float)))
    ax.tick_params(labelsize=3, left=False, top=False, right=False,
                   bottom=False)
    ax.cax.colorbar(im)
    ax.cax.tick_params(labelsize=5, right=False)
    ax.set_yticks(np.arange(m))
    ax.set_xlabel('Threshold (multiples of standard deviation)', fontsize=6)
    mpl.rcParams['text.usetex'] = True
    f = lambda x: r'\textbf{{0}}, {1}, \textbf{{2}}, {3}, {4:.1f}'.format(*x)
    ax.set_yticklabels(map(f, trimmed.index))
    ax.set_ylabel('shank i, channel i, shank j, channel j % of max distance',
                  fontsize=6)
    ax.set_title('Age: {0}, Site: {1}'.format(age, site))
    if args.plot_filename is None:
        plot_filename = os.path.splitext(args.filename)[0]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        fig.tight_layout()
        fig.savefig('{0}{1}{2}'.format(plot_filename, os.extsep,
                                       args.plot_format), fmt=args.plot_format,
                    bbox_inches='tight')


class Analyzer(SpanCommand):
    pass


class CorrelationAnalyzer(Analyzer):
    def _run(self, args):
        show_xcorr(args)


class IPythonAnalyzer(Analyzer):
    """Drop into an IPython shell given a filename or database id number"""
    def _run(self, args):
        try:
            from IPython import embed
        except ImportError:
            return error('ipython not installed, please install it with pip '
                         'install ipython')
        else:
            tank, spikes = self._load_data(return_tank=True)
            embed()
        return 0
