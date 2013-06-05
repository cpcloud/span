#!/usr/bin/env python

import sys
import re
import argparse
import os
import warnings

import numpy as np
import pandas as pd

from dateutil.parser import parse as _parse_date

from span.spanner.db import Db, DbCreator, DbReader, DbUpdater, DbDeleter
from span.spanner.analyzer import CorrelationAnalyzer, IPythonAnalyzer
from span.spanner.converters import Converter
from span.spanner.viewer import Viewer
from span.spanner.utils import _init_db
from span.spanner.defaults import SPAN_DB

import span
from span import TdtTank, NeuroNexusMap, ElectrodeMap


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
    tank = TdtTank(filename, em)

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
    mpl.use('pdf')
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import ImageGrid
    from bottleneck import nanmax, nanmin
    trimmed, age, site = compute_xcorr(args)

    vmax = nanmax(trimmed.values)
    vmin = nanmin(trimmed.values)

    fig = plt.figure(figsize=(16, 9))
    grid = ImageGrid(fig, 111, nrows_ncols=(1, 1), direction='row',
                     axes_pad=0.05, add_all=True, label_mode='1',
                     share_all=False, cbar_location='right',
                     cbar_mode='single', cbar_size='10%', cbar_pad=0.05)
    ax = grid[0]
    trimmed.sortlevel('shank i', inplace=True)
    trimmed.sortlevel('shank j', inplace=True)
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
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        fig.tight_layout()
        fig.savefig('{0}{1}pdf'.format(os.path.splitext(args.filename)[0],
                                       os.extsep), bbox_inches='tight')


def build_analyze_parser(subparsers):
    def build_correlation_parser(subparsers):
        parser = subparsers.add_parser('correlation', help='perform cross '
                                       'correlation analysis on a recording',
                                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument('-f', '--filename', help='filename')
        parser.add_argument('-i', '--id', help='id')
        cleaning = parser.add_argument_group('cleaning')
        display = parser.add_argument_group('display')
        thresholding = parser.add_argument_group('thresholding')
        binning = parser.add_argument_group('binning')
        xcorr = parser.add_argument_group('cross correlation')
        cleaning.add_argument('-c', '--remove-first-pc', action='store_true',
                              help='remove the first principal component of the data. warning: this drastically slows down the analysis')
        display.add_argument('-d', '--display', action='store_true',
                             help='display the resulting cross correlation analysis')
        thresholding.add_argument(
            '-T', '--max-threshold', type=float, default=4.0,
            help='maximum threshold in multiples of the standard deviation of the voltage data')
        thresholding.add_argument(
            '-t', '--min-threshold', type=float, default=3.0,
            help='minimum threshold in multiples of the standard deviation of the voltage data')
        thresholding.add_argument('-n', '--num-thresholds', type=int, default=50)
        thresholding.add_argument(
            '-r', '--refractory-period', type=int, default=2, help='refractory period in milliseconds')
        binning.add_argument(
            '-b', '--bin-size', default='S', help='bin size in some time unit')
        binning.add_argument(
            '-p', '--bin-method', default='sum', help='function to use for binning spikes')
        binning.add_argument(
            '-R', '--firing-rate-threshold', type=float, default=1.0)
        xcorr.add_argument('-w', '--within-shank', type=float, default=50.0)
        xcorr.add_argument('-W', '--between-shank', type=float, default=125.0)
        xcorr.add_argument(
            '-s', '--scale-type', choices=('normalize', 'none', 'biased',
                                           'unbiased'), default='normalize',
            help='type of scaling to use on the raw cross correlation')
        xcorr.add_argument(
            '-m', '--detrend', choices=('mean', 'linear', 'none'),
            default='mean', help='function to use to detrend the raw cross correlation')
        xcorr.add_argument('-l', '--max-lags', type=int, default=1,
                           help='maximum number of lags of the cross correlation to return')
        xcorr.add_argument('-L', '--which-lag', type=int, default=0)
        xcorr.add_argument(
            '-k', '--keep-auto', action='store_true', help='keep the autocorrelation values')
        parser.set_defaults(run=show_xcorr)

    parser = subparsers.add_parser('analyze', help='perform an analysis on a '
                                   'TDT tank file')
    subparsers = parser.add_subparsers()
    build_correlation_parser(subparsers)


class DateParseAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, _parse_date(values))


class ArtifactRangesAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, _parse_artifact_ranges(values))


def build_convert_parser(subparsers):
    parser = subparsers.add_parser('convert', help='convert a TDT tank file '
                                   'into a different format')
    add_filename_and_id_to_parser(parser)
    parser.add_argument('-t', '--type',
                        help='the type of conversion you want to '
                        'perform', choices=('neuroscope', 'matlab', 'numpy',
                                            'h5'), required=True)
    parser.add_argument('-d', '--base-type',
                        help='the base numeric type to convert to',
                        default='float', choices=('float', 'int', 'uint', 'f',
                                                  'i', 'ui'), required=True)
    parser.add_argument('-p', '--precision', help='the number of bits '
                        'to use for conversion', type=int, default=64,
                        choices=(8, 16, 32, 64), required=True)
    parser.set_defaults(run=Converter().run)


def build_view_parser(subparsers):
    parser = subparsers.add_parser('view', help='display the raw traces of a '
                                   'TDT tank file in Neuroscope')
    add_filename_and_id_to_parser(parser)
    parser.add_argument('-s', '--start-time', type=int,
                        help='where to place you in the recording when showing'
                        ' the data')
    parser.add_argument('-w', '--window-size', type=int,
                        help='the number of milliseconds to show in the full '
                        'window')
    parser.add_argument('-r', '--voltage-range', type=int, default=10,
                        help='a magical parameter needed by neuroscope')
    parser.add_argument('-a', '--amplification', type=int, default=1000,
                        help='another magical parameter needed by neuroscope')
    parser.add_argument('-t', '--format', default='gz', help='the type of '
                        'archive in which to output a neuroscope-ready data '
                        'set, default: gz', choices=('gz', 'bz2'))
    parser.set_defaults(run=Viewer().run)


def build_db_parser(subparsers):
    def _add_args_to_parser(parser):
        parser.add_argument(
            '-a', '--age', type=int, help='the age of the animal')
        parser.add_argument(
            '-t', '--animal-type', help='the kind of animal, e.g., rat, mouse, etc.')
        parser.add_argument(
            '-r', '--artifact-ranges', action=ArtifactRangesAction, help='the ranges of the artifacts')
        parser.add_argument(
            '-s', '--between-shank', type=float, help='the distance between the shanks')
        parser.add_argument(
            '-c', '--condition', help='the experimental condition, if any')
        parser.add_argument(
            '-d', '--date', action=DateParseAction, help='the date of the recording')
        parser.add_argument(
            '-f', '--filename', help='name of the file to store')
        parser.add_argument(
            '-i', '--id', type=int, help='force a particular id number. WARNING: this is not recommended')
        parser.add_argument('-o', '--shank-order', choices=(
            'lm', 'ml'), help='the ordering of the shanks relative to the MNTB')
        parser.add_argument('-p', '--probe', help='the probe number')
        parser.add_argument(
            '-l', '--site', type=int, help='the site of the recording')
        parser.add_argument('-v', '--invalid-recording', action='store_true',
                            help='pass this argument if the recording is invalid')
        parser.add_argument(
            '-w', '--weight', type=float, help='the weight of the animal')
        parser.add_argument('-e', '--within-shank', type=float,
                            help='the distance between the channels on each shank')

    def build_db_create_parser(subparsers):

        parser = subparsers.add_parser('create', help='put a new recording in '
                                       'the database')
        _add_args_to_parser(parser)
        parser.set_defaults(run=DbCreator().run)

    def build_db_read_parser(subparsers):
        parser = subparsers.add_parser('read', help='query the properties of a'
                                       ' recording')
        _add_args_to_parser(parser)
        parser.set_defaults(run=DbReader().run)

    def build_db_update_parser(subparsers):
        parser = subparsers.add_parser('update', help='update the properties '
                                       'of an existing recording')
        _add_args_to_parser(parser)
        parser.set_defaults(run=DbUpdater().run)

    def build_db_delete_parser(subparsers):
        parser = subparsers.add_parser('delete', help='delete a recording or '
                                       'recordings matching certain '
                                       'conditions')
        _add_args_to_parser(parser)
        parser.set_defaults(run=DbDeleter().run)

    parent_parser = subparsers.add_parser('db', help='Operate on the database '
                                          'of recordings', add_help=False)
    subparsers = parent_parser.add_subparsers(description='use the following '
                                              'subcomands to perform specific '
                                              'operations on the database of '
                                              'recordings')
    build_db_create_parser(subparsers)
    build_db_read_parser(subparsers)
    build_db_update_parser(subparsers)
    build_db_delete_parser(subparsers)


def add_filename_and_id_to_parser(parser):
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-f', '--filename',
                       help='The name of the file to read from')
    group.add_argument('-i', '--id', type=int, help='alternatively you can use'
                       ' a database id number of a recording if you know it '
                       '(you can query for these using spanner db read')


def main():
    parser = argparse.ArgumentParser(description='Analyze TDT tank files')
    subparsers = parser.add_subparsers(help='Subcommands for analying TDT '
                                       'tank files')
    build_analyze_parser(subparsers)
    build_convert_parser(subparsers)
    #build_view_parser(subparsers)
    #build_db_parser(subparsers)
    args = parser.parse_args()
    return args.run(args)


if __name__ == '__main__':
    try:
        _init_db(SPAN_DB, Db)
    except:
        pass
    sys.exit(main())
