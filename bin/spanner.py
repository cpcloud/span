#!/usr/bin/env python

import os
import sys
import argparse

from dateutil.parser import parse as _parse_date

from span.spanner.db import Db, DbCreator, DbReader, DbUpdater, DbDeleter
from span.spanner.analyzer import (CorrelationAnalyzer, IPythonAnalyzer,
                                   _parse_artifact_ranges)
from span.spanner.converters import Converter
from span.spanner.utils import _init_db
from span.spanner.defaults import SPAN_DB


def _find_below_common_data_path(path, common_data_path):
    pass


def build_analyze_parser(subparsers):
    def build_correlation_parser(subparsers):
        parser = subparsers.add_parser('correlation', help='perform cross '
                                       'correlation analysis on a recording',
                                       formatter_class=
                                       argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument('-f', '--filename', help='filename')
        parser.add_argument('-i', '--id', help='id')
        cleaning = parser.add_argument_group('cleaning')
        display = parser.add_argument_group('display')
        thresholding = parser.add_argument_group('thresholding')
        binning = parser.add_argument_group('binning')
        xcorr = parser.add_argument_group('cross correlation')
        cleaning.add_argument('-c', '--remove-first-pc', action='store_true',
                              help='remove the first principal component of '
                              'the data. warning: this drastically slows down '
                              'the analysis')
        display.add_argument('-o', '--plot-filename', help='the name of the '
                             'file to which the plot is output')
        display.add_argument('-F', '--plot-format', help='the output format of'
                             ' the plot', default='pdf')
        display.add_argument('-B', '--sort-by', help='how to sort the '
                             'resulting y-axis', choices=('shank', 'distance'))
        thresholding.add_argument(
            '-T', '--max-threshold', type=float, default=4.0,
            help='maximum threshold in multiples of the standard deviation of '
            'the voltage data')
        thresholding.add_argument(
            '-t', '--min-threshold', type=float, default=3.0,
            help='minimum threshold in multiples of the standard deviation of '
            'the voltage data')
        thresholding.add_argument('-n', '--num-thresholds', type=int,
                                  default=50)
        thresholding.add_argument(
            '-r', '--refractory-period', type=int, default=2, help='refractory'
            ' period in milliseconds')
        binning.add_argument(
            '-b', '--bin-size', default='S', help='bin size in some time unit')
        binning.add_argument(
            '-p', '--bin-method', default='sum', help='function to use for '
            'binning spikes')
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
            default='mean', help='function to use to detrend the raw cross '
            'correlation')
        xcorr.add_argument('-l', '--max-lags', type=int, default=1,
                           help='maximum number of lags of the cross '
                           'correlation to return')
        xcorr.add_argument('-L', '--which-lag', type=int, default=0)
        xcorr.add_argument(
            '-k', '--keep-auto', action='store_true', help='keep the '
            'autocorrelation values')
        parser.set_defaults(run=CorrelationAnalyzer().run)

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
    parser.add_argument('-t', '--format', help='the type of conversion '
                        'you want to perform', choices=('neuroscope', 'igor',
                        'matlab', 'numpy', 'h5'), required=True)
    parser.add_argument('-T', '--numeric-type',
                        help='the base numeric type to convert to',
                        default='float', choices=('float', 'int', 'uint'))
    parser.add_argument('-p', '--precision', help='the number of bits '
                        'to use for conversion', type=int, default=64,
                        choices=(8, 16, 32, 64))
    parser.add_argument('-s', '--start-time', type=int,
                        help='where to place you in the recording when showing'
                        ' the data', default=1)
    parser.add_argument('-w', '--window-size', type=int,
                        help='the number of milliseconds to show in the full '
                        'window', default=1000)
    parser.add_argument('-r', '--voltage-range', type=int, default=10,
                        help='a magical parameter needed by neuroscope')
    parser.add_argument('-a', '--amplification', type=int, default=1000,
                        help='another magical parameter needed by neuroscope')
    parser.add_argument('-c', '--compression-format', default='gz', help='the '
                        'type of archive in which to output a neuroscope-ready'
                        ' data set, default: gz', choices=('gz', 'bz2'))
    parser.set_defaults(run=Converter().run)


def build_db_parser(subparsers):
    def _add_args_to_parser(parser):
        parser.add_argument(
            '-a', '--age', type=int, help='the age of the animal')
        parser.add_argument(
            '-t', '--animal-type', help='the kind of animal, e.g., rat, mouse,'
            ' etc.')
        parser.add_argument(
            '-r', '--artifact-ranges', action=ArtifactRangesAction,
            help='the ranges of the artifacts')
        parser.add_argument(
            '-s', '--between-shank', type=float,
            help='the distance between the shanks')
        parser.add_argument(
            '-c', '--condition', help='the experimental condition, if any')
        parser.add_argument(
            '-d', '--date', action=DateParseAction,
            help='the date of the recording')
        parser.add_argument(
            '-f', '--filename', help='name of the file to store')
        parser.add_argument(
            '-i', '--id', type=int,
            help='force a particular id number. WARNING: this is not '
            'recommended')
        parser.add_argument('-o', '--shank-order', choices=('lm', 'ml'),
                            help='the ordering of the shanks relative to the '
                            'MNTB')
        parser.add_argument('-p', '--probe', help='the probe number')
        parser.add_argument(
            '-l', '--site', type=int, help='the site of the recording')
        parser.add_argument('-v', '--invalid-recording', action='store_true',
                            help='pass this argument if the recording is'
                            'invalid')
        parser.add_argument(
            '-w', '--weight', type=float, help='the weight of the animal')
        parser.add_argument('-e', '--within-shank', type=float,
                            help='the distance between the channels on each'
                            'shank')

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
    parser.add_argument('-P', '--common-data-path',
                        default=os.path.expanduser(os.path.join('~Adrian',
                                                                'Data',
                                                                'Multielectrode')),
                        help='search in this directory for the path provided')


def main():
    parser = argparse.ArgumentParser(description='Analyze TDT tank files')
    subparsers = parser.add_subparsers(help='Subcommands for analying TDT '
                                       'tank files')
    build_analyze_parser(subparsers)
    build_convert_parser(subparsers)
    #build_db_parser(subparsers)
    args = parser.parse_args()
    return args.run(args)


if __name__ == '__main__':
    try:
        _init_db(SPAN_DB, Db)
    except:
        pass
    sys.exit(main())
