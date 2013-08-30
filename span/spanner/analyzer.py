import re
import warnings
import os

import numpy as np
import pandas as pd

import span
from span import ElectrodeMap, NeuroNexusMap, TdtTank, SpikeDataFrame

from span.spanner.command import SpanCommand
from span.spanner.utils import error
from span.utils import bold, green, puts


def _is_proper_spike_frame(df):
    return (isinstance(df, SpikeDataFrame) and
            isinstance(df.columns, pd.MultiIndex) and
            isinstance(df.index, pd.DatetimeIndex) and
            df.columns.names == ['shank', 'channel'])


def _frame_to_spike_frame(df, names=None):
    if _is_proper_spike_frame(df):
        return df
    columns = df.columns.copy()
    names = names if names is not None else ['shank', 'channel']

    if not isinstance(columns, pd.MultiIndex):
        columns = pd.MultiIndex.from_tuples(columns, names=names)
    else:
        columns.names = names

    return SpikeDataFrame(df.values, index=df.index, columns=columns)


def _get_max_within_shank(xcs):
    shank_i, shank_j = map(xcs.index.get_level_values, ('shank i', 'shank j'))
    eq_shanks = shank_i == shank_j
    return xcs.distance[eq_shanks].max(), eq_shanks


def split_xcorrs(xcs):
    max_within_shank, eq_shanks = _get_max_within_shank(xcs)
    dist_index = xcs.distance <= max_within_shank
    shanks_equal = xcs[dist_index & eq_shanks]
    shanks_not_equal = xcs[~eq_shanks]
    return shanks_equal, shanks_not_equal


def split_xcorrs_index(xcs):
    w, b = split_xcorrs(xcs)
    w.set_index(['distance'], append=True, inplace=True)
    b.set_index(['distance'], append=True, inplace=True)
    return w.index.append(b.index)


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


def concat_xcorrs(xcs, scale_max_dist=False):
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
        puts(bold(green('loaded cleaned xcs data')))
    except KeyError:
        try:
            spikes = _frame_to_spike_frame(pd.read_hdf(h5name, 'sp'))
            puts(bold(green('read spikes from h5 file, shape: '
                            '{0}'.format(spikes.shape))))
        except KeyError:
            # get the raw data
            spikes = _frame_to_spike_frame(tank.spik)

            if args.store_h5:
                pd.DataFrame(spikes).to_hdf(h5name, 'sp', append=True)
                puts(bold(green('wrote h5 file')))

        # get the threshes
        try:
            sd = pd.read_hdf(h5name, 'sd')
            puts(bold(green('loaded sd from h5')))
        except KeyError:
            sd = spikes.std()
            sd.to_hdf(h5name, 'sd')
            puts(bold(green('wrote sd to h5')))

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
    return trimmed, tank.age, tank.site, tank.date


try:
    from bottleneck import nanmax, nanmin
except ImportError:
    from numpy import nanmax, nanmin

def _try_1f(value):
    try:
        return '{0:.1f}'.format(float(value))
    except ValueError:
        return value


def _get_xticklabels(threshes, formatter=_try_1f):
    strs = pd.Series(map(str, threshes))
    radix = strs.str.split('.').str.get(0).duplicated()
    strs[radix] = ''
    return map(formatter, strs)


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
    return fig, ax, im


def _color_yticks(xcs, color):
    colored = r'\textbf{{{0}}}, {1}, \textbf{{{2}}}, {3}, {4:.1f}'
    try:
        index = split_xcorrs_index(xcs.reset_index('distance'))
        res = map(lambda x: colored.format(*x), index.values)
        return res
    except IndexError:
        v = xcs.index.values

        try:
            return map(lambda x: '{0:.1f}'.format(float(x)), v)
        except TypeError:
            return map('{0}'.format, v)


def plot_xcorrs(trimmed, tick_labelsize=8, titlesize=15, xlabelsize=10,
                ylabelsize=10, cbar_labelsize=8, title='',
                xlabel='Threshold (multiples of standard deviation)',
                ylabel='', figsize=(16, 9), usetex=True,
                split_color='blue'):
    import matplotlib as mpl
    mpl.rc('text', usetex=usetex)

    if 'distance' not in trimmed.index.names:
        trimmed = trimmed.set_index(['distance'], drop=True, append=True)

    fig, ax, im = frame_image(trimmed, figsize=figsize)

    m, n = trimmed.shape

    ax.set_xticks(np.arange(n))
    tlbls = _get_xticklabels(trimmed.columns.values.astype(float))
    ax.set_xticklabels(tlbls)
    ax.tick_params(labelsize=tick_labelsize, left=False, top=False,
                   right=False, bottom=False)
    ax.cax.colorbar(im)
    ax.cax.tick_params(labelsize=cbar_labelsize, right=False)
    ax.set_yticks(np.arange(m))
    ax.set_xlabel(xlabel, fontsize=xlabelsize)

    ax.set_yticklabels(_color_yticks(trimmed, split_color))
    ax.set_ylabel(ylabel, fontsize=ylabelsize)
    ax.set_title(title, fontsize=titlesize)
    return fig, ax


def analyze(xcs, scale_dist=False):
    xcs_concat = concat_xcorrs(xcs, scale_dist)
    xcs_ind = xcs_concat.set_index(['distance'], drop=True, append=True)
    gb_dist = xcs_ind.groupby(level='distance', axis=0)
    mdist = gb_dist.mean()
    return mdist, xcs_concat


def show_xcorr(args):
    import matplotlib as mpl
    mpl.use('Agg')
    trimmed, age, site, date = compute_xcorr_with_args(args)
    _, xcs_concat = analyze(trimmed)
    w, b = split_xcorrs(xcs_concat)
    agg = pd.concat([w, b])
    cols = agg.columns
    agg = agg.drop_duplicates().dropna(how='all', axis=0,
                                       subset=cols[cols != 'distance'])
    fig, _ = plot_xcorrs(agg, title='Age: P{0}, Site: {1}, Date: '
                         '{2}'.format(age, site, date), tick_labelsize=5)

    plot_filename = args.plot_filename or os.path.splitext(args.filename)[0]

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', Warning)
        fig.tight_layout()
        fig.autofmt_xdate()
        fmt = args.plot_format
        fn = '{0}{1}{2}'.format(plot_filename, os.extsep, fmt)
        fig.savefig(fn, fmt=fmt, bbox_inches='tight')


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
