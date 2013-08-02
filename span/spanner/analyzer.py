import re
import warnings
import os

import numpy as np
import pandas as pd

import span
from span import ElectrodeMap, NeuroNexusMap, TdtTank, SpikeDataFrame

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


def get_xcorr(sp, threshold, sd, binsize='S', how='sum',
              firing_rate_threshold=1.0, refractory_period=2, nan_auto=True,
              detrend='mean', scale_type='normalize', which_lag=0):
    thr = sp.threshold(threshold * sd)
    thr.clear_refrac(refractory_period, inplace=True)

    binned = thr.resample(binsize, how=how)
    binned.loc[:, binned.mean() < firing_rate_threshold] = np.nan

    xc = thr.xcorr(binned, detrend=getattr(span, 'detrend_' + detrend),
                   scale_type=scale_type, nan_auto=nan_auto)
    s = xc.loc[which_lag]
    s.name = threshold
    return s


def get_xcorr_multi_thresh(sp, threshes, sd, distance_map, binsize='S',
                           how='sum', firing_rate_threshold=1.0,
                           refractory_period=2, nan_auto=True, detrend='mean',
                           scale_type='normalize', which_lag=0):
    xcs = pd.concat([get_xcorr(sp, threshold=thresh, sd=sd, binsize=binsize,
                               how=how,
                               firing_rate_threshold=firing_rate_threshold,
                               refractory_period=refractory_period,
                               nan_auto=nan_auto, detrend=detrend,
                               scale_type=scale_type, which_lag=0) for thresh
                     in threshes], axis=1)
    dname = distance_map.name
    xcs[dname] = distance_map
    xcs.sort(dname, inplace=True)
    subset = xcs.columns[xcs.columns != dname]
    xcs = xcs.dropna(axis=0, how='all', subset=subset)
    xcs = xcs.dropna(axis=1, how='all')
    xcs.drop_duplicates(inplace=True)
    return xcs


def trim_sans_distance(xcs_df, dist_key='distance'):
    cols = xcs_df.columns
    subset = cols[cols != dist_key]
    return xcs_df.dropna(axis=0, how='all',
                         subset=subset).dropna(axis=1, how='all')


def concat_xcorrs(xcs, scale_max_dist):
    xc_all = xcs.copy()
    if scale_max_dist:
        xc_all.distance /= xc_all.distance.max()
    dist_key = 'distance'
    cols = xc_all.columns
    subset = cols[cols != dist_key]
    return xc_all.dropna(axis=0, how='all', subset=subset)


def tank_to_prec(tank, **fields):
    from pandas import Series
    d = {'date': tank.date,
         'filename': tank.path,
         'age': tank.age,
         'weight': fields.get('weight', None),
         'probe': fields.get('probe', None),
         'within_shank': tank.electrode_map.within_shank,
         'between_shank': tank.electrode_map.between_shank,
         'order': fields.get('order', None),
         'site': tank.site,
         'condition': fields.get('condition', None),
         'animal_type': fields.get('animal_type', None),
         'filebase': tank.path,
         'basename': tank.name,
         'duration': tank.duration,
         'start': tank.start,
         'end': tank.end}
    d.update(fields)
    prec = Series(d, name='prec')
    return prec


def compute_xcorr_with_args(args):
    filename = args.filename
    h5name = '{0}{1}h5'.format(filename, os.extsep)
    mn, mx, n = args.min_threshold, args.max_threshold, args.num_thresholds
    threshes = np.linspace(mn, mx, n)

    name = '_'.join(map(str, (mn, mx, n, args.bin_size, args.bin_method,
                              args.firing_rate_threshold,
                              args.refractory_period, args.detrend,
                              args.scale_type)))
    xcs_name = 'xcs_{name}'.format(name=name)
    em = ElectrodeMap(NeuroNexusMap.values, args.within_shank,
                      args.between_shank)
    # make a tank
    tank = TdtTank(filename, em, clean=args.remove_first_pc)

    try:
        xcs = pd.read_hdf(h5name, xcs_name)
    except KeyError:
        try:
            spikes = SpikeDataFrame(pd.read_hdf(h5name, 'sp'))
            spikes.columns = pd.MultiIndex.from_tuples(spikes.columns,
                                                    names=['shank', 'channel'])
        except KeyError:

            # get the raw data
            spikes = tank.spik

            if args.store_h5:
                pd.DataFrame(spikes).to_hdf(h5name, 'sp', append=True)

        # get the threshes
        try:
            sd = pd.read_hdf(h5name, 'sd')
        except KeyError:
            sd = spikes.std()
            sd.to_hdf(h5name, 'sd')

        # compute the cross correlation at each threshold
        xcs = get_xcorr_multi_thresh(spikes, threshes, sd, em.distance_map(),
                                     binsize=args.bin_size,
                                     how=args.bin_method,
                                     firing_rate_threshold=args.firing_rate_threshold,
                                     refractory_period=args.refractory_period,
                                     nan_auto=not args.keep_auto,
                                     detrend=args.detrend,
                                     scale_type=args.scale_type)
        xcs.to_hdf(h5name, xcs_name)
    try:
        prec = pd.read_hdf(h5name, 'prec')
    except KeyError:
        prec = tank_to_prec(tank)
        prec.to_hdf(h5name, 'prec')

    # concat all xcorrs
    xcs_df = concat_xcorrs(xcs, args.scale_max_dist)

    # remove the data that is useless
    trimmed = trim_sans_distance(xcs_df)
    return trimmed, tank.age, tank.site


try:
    from bottleneck import nanmax, nanmin
except ImportError:
    from numpy import nanmax, nanmin


def frame_image(df, figsize=(16, 9), interpolation='none', cbar_mode='single',
                cbar_size='10%', cbar_pad=0.05, cbar_location='right',
                aspect='auto', vmax=None, vmin=None):
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import ImageGrid

    values = df.values
    vmax = vmax or nanmax(values)
    vmin = vmin or nanmin(values)
    fig = plt.figure(figsize=figsize)
    grid = ImageGrid(fig, 111, nrows_ncols=(1, 1), direction='row',
                     axes_pad=0.05, add_all=True, label_mode='1',
                     share_all=False, cbar_location=cbar_location,
                     cbar_mode=cbar_mode, cbar_size=cbar_size,
                     cbar_pad=cbar_pad)
    ax = grid[0]
    im = ax.imshow(values, interpolation=interpolation, aspect=aspect,
                   vmax=vmax, vmin=vmin)
    return ax, im


def plot_xcorrs(trimmed, tick_labelsize=8, titlesize=15, xlabelsize=10,
                ylabelsize=10, cbar_labelsize=8, title='',
                xlabel='Threshold (multiples of standard deviation)',
                ylabel='', figsize=(16, 9), usetex=True):
    import matplotlib as mpl
    mpl.rc('text', usetex=usetex)

    if 'distance' not in trimmed.index.names:
        trimmed = trimmed.set_index(['distance'], drop=True, append=True)

    ax, im = frame_image(trimmed, figsize=figsize)

    m, n = trimmed.shape

    ax.set_xticks(np.arange(n))
    ax.set_xticklabels(map('{0:.1f}'.format,
                           trimmed.columns.values.astype(float)))
    ax.tick_params(labelsize=tick_labelsize, left=False, top=False, right=False,
                   bottom=False)
    ax.cax.colorbar(im)
    ax.cax.tick_params(labelsize=cbar_labelsize, right=False)
    ax.set_yticks(np.arange(m))
    ax.set_xlabel(xlabel, fontsize=xlabelsize)
    f = lambda x: r'\textbf{{{0}}}, {1}, \textbf{{{2}}}, {3}, {4:.1f}'.format(*x)

    try:
        ax.set_yticklabels(map(f, trimmed.index))
    except TypeError:
        ax.set_yticklabels(map('{0:.1f}'.format, trimmed.index))
    ax.set_ylabel(ylabel, fontsize=ylabelsize)
    ax.set_title(title, fontsize=titlesize)


def show_xcorr(args):
    import matplotlib as mpl
    mpl.use('Agg')
    trimmed, age, site = compute_xcorr_with_args(args)

    ax, im = frame_image(trimmed)
    fig = ax.get_figure()

    if args.sort_by == 'shank':
        trimmed.sortlevel('shank i', inplace=True)
        trimmed.sortlevel('shank j', inplace=True)
    elif args.sort_by == 'distance':
        trimmed.sortlevel('distance', inplace=True)

    trimmed = trimmed.set_index(['distance'], append=True, drop=True)
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
    f = lambda x: r'\textbf{{{0}}}, {1}, \textbf{{{2}}}, {3}, {4:.1f}'.format(*x)
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
