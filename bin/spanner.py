#!/usr/bin/env python

import sys
import os
import argparse
import numbers

import pandas as pd
import IPython as ipy
from clint.textui import puts
from clint.textui.colored import red

from span import TdtTank, NeuroNexusMap, spike_xcorr


def error(msg):
    errmsg = os.path.basename(__file__)
    errmsg += ': error: {0}'.format(msg)
    puts(red(errmsg))
    return sys.exit(2)


def _get_fn(id_num_or_filename, db_path=os.environ.get('SPAN_DB_PATH', None)):
    if db_path is None:
        error('SPAN_DB_PATH environment variable not set, please set via '
              '"export SPAN_DB_PATH=\'path_to_the_span_database\'"')
    db_path = os.path.abspath(db_path)
    db = pd.read_hdf(db_path, 'db')

    if isinstance(id_num_or_filename, numbers.Integral):
        if id_num_or_filename not in db.index:
            error('{0} is not a valid id number'.format(id_num_or_filename))
    elif isinstance(id_num_or_filename, basestring):
        if id_num_or_filename not in db.filename.values:
            error('{0} is not a valid filename'.format(id_num_or_filename))
        return id_num_or_filename
    return db.filename.ix[id_num_or_filename]


class SpanCommand(object):
    def _parse_filename_and_id(self, args):
        if args.filename is None and args.id is not None:
            self.filename = _get_fn(args.id)
        elif args.filename is not None and args.id is None:
            self.filename = _get_fn(args.filename)
        else:
            return error('Must pass a valid id number or filename')

        if not os.path.exists(self.filename):
            return error('"{0}" is a nonexistent path'.format(self.filename))

    def run(self, args):
        self._parse_filename_and_id(args)
        return self._run(args)


class Analyzer(SpanCommand):
    def _load_data(self, return_tank=False):
        tank = TdtTank(self.filename, NeuroNexusMap)
        spikes = tank.spikes

        if return_tank:
            return tank, spikes
        else:
            return spikes

    def _run(self, args):
        pass


class CorrelationAnalyzer(Analyzer):
    def _compute_xcorr(self, args):
        threshold = args.threshold
        bin_size = args.bin_size
        scale_type = args.scale_type
        detrend = args.detrend
        max_lags = args.max_lags
        nan_auto = args.nan_auto
        ms = args.refractory_period
        bin_method = args.bin_method

        spikes = self._load_data()
        thr = spikes.threshold(threshold)
        thr.clear_refrac(ms=ms, inplace=True)
        binned = thr.bin(bin_size, how=bin_method, reduced=False)
        xc = spike_xcorr(binned, max_lags=max_lags, scale_type=scale_type,
                         detrend=detrend, nan_auto=nan_auto)
        return xc

    def _run(self, args):
        display = args.display
        xc = self._compute_xcorr(args)

        if display:
            self._display_xcorr(xc)

    def _display_xcorr(self, xc):
        pass


class IPythonAnalyzer(Analyzer):
    """Drop into an IPython shell given a filename or database id number"""
    def _run(self, args):
        tank = self._load_data()
        ipy.embed_kernel(local_ns={'tank': tank, 'raw': tank.spikes})


class Converter(SpanCommand):
    def _run(self, args):
        pass


# get the filename/id, convert to neuroscope with int16 precision, zip into
# package, unzip and show in neuroscope
class Viewer(SpanCommand):
    def _run(self, args):
        pass


class Db(SpanCommand):
    pass


class DbCreator(Db):
    def _run(self, args):
        pass


class DbReader(Db):
    def _run(self, args):
        pass


class DbUpdater(Db):
    def _run(self, args):
        pass


class DbDeleter(Db):
    def _run(self, args):
        pass


def build_analyze_parser(subparsers, parent):
    def build_correlation_parser(subparsers):
        parser = subparsers.add_parser('correlation', help='perform cross '
                                       'correlation analysis on a recording')
        add_filename_and_id_to_parser(parser)
        parser.add_argument('-d', '--display', action='store_true')
        parser.add_argument('-t', '--threshold', type=float)
        parser.add_argument('-r', '--refractory-period', type=int)
        parser.add_argument('-b', '--bin-size', type=int)
        parser.add_argument('-p', '--bin-method')
        parser.add_argument('-s', '--scale-type', choices=('normalize', 'none',
                                                           'biased',
                                                           'unbiased'))
        parser.add_argument('-m', '--detrend')
        parser.add_argument('-l', '--max-lags', type=int)
        parser.add_argument('-n', '--nan-auto', action='store_true')
        parser.set_defaults(run=CorrelationAnalyzer().run)

    def build_ipython_parser(subparsers):
        parser = subparsers.add_parser('ipython', help='drop into an ipython '
                                       'shell')
        add_filename_and_id_to_parser(parser)
        parser.set_defaults(run=IPythonAnalyzer().run)

    parser = subparsers.add_parser('analyze', help='perform an analysis on a '
                                   'TDT tank file')
    subparsers = parser.add_subparsers()
    build_correlation_parser(subparsers)
    build_ipython_parser(subparsers)


def build_convert_parser(subparsers, parent):
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


def build_view_parser(subparsers, parent):
    parser = subparsers.add_parser('view', help='display the raw traces of a '
                                   'TDT tank file in Neuroscope',
                                   parents=[parent])
    parser.set_defaults(run=Viewer().run)


def build_db_parser(subparsers, parent):
    def build_db_create_parser(subparsers):
        parser = subparsers.add_parser('create', help='put a new recording in '
                                       'the database')
        add_filename_and_id_to_parser(parser)
        parser.set_defaults(run=DbCreator().run)

    def build_db_read_parser(subparsers):
        parser = subparsers.add_parser('read', help='query the properties of a'
                                       ' recording')
        add_filename_and_id_to_parser(parser)
        parser.set_defaults(run=DbReader().run)

    def build_db_update_parser(subparsers):
        parser = subparsers.add_parser('update', help='update the properties '
                                       'of an existing recording')
        add_filename_and_id_to_parser(parser)
        parser.set_defaults(run=DbUpdater().run)

    def build_db_delete_parser(subparsers):
        parser = subparsers.add_parser('delete', help='delete a recording or '
                                       'recordings matching certain '
                                       'conditions')
        add_filename_and_id_to_parser(parser)
        parser.set_defaults(run=DbDeleter().run)

    parent_parser = subparsers.add_parser('db', help='Operate on the database '
                                          'of recordings', add_help=False)
    parent_parser.add_argument('-a', '--age', type=int, help='the age of the '
                               'animal')
    parent_parser.add_argument('-c', '--condition', help='the experimental '
                               'condition, if any')
    parent_parser.add_argument('-w', '--weight', type=float, help='the weight '
                               'of the animal')
    parent_parser.add_argument('-b', '--bad', action='store_true', help='Mark '
                               'a recording as "good"')
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
    group.add_argument('-f', '--filename')
    group.add_argument('-i', '--id', type=int)


def main():
    parent_parser = argparse.ArgumentParser(description='analyze TDT tank '
                                            'files', add_help=False)
    subparsers = parent_parser.add_subparsers(help='subcommands for analying '
                                              'TDT tankfiles')
    build_analyze_parser(subparsers, parent_parser)
    build_convert_parser(subparsers, parent_parser)
    build_view_parser(subparsers, parent_parser)
    build_db_parser(subparsers, parent_parser)
    args = parent_parser.parse_args()
    return args.run(args)


if __name__ == '__main__':
    sys.exit(main())
