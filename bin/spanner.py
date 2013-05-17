#!/usr/bin/env python

import sys
import os
import argparse
import numbers

import numpy as np
import pandas as pd


CHAR_BIT = 8


def error(msg):
    from clint.textui import puts
    from clint.textui.colored import red
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
        from span import TdtTank, NeuroNexusMap
        tank = TdtTank(self.filename, NeuroNexusMap)
        spikes = tank.spikes

        if return_tank:
            return tank, spikes
        else:
            return spikes

    def _run(self, args):
        raise NotImplementedError()


def _compute_xcorr(spikes, args):
    import span
    from span import spike_xcorr
    detrend = 'detrend_' + args.detrend
    thr = spikes.threshold(args.threshold)
    thr.clear_refrac(ms=args.refractory_period, inplace=True)
    binned = thr.bin(args.bin_size, how=args.bin_method)
    xc = spike_xcorr(binned, max_lags=args.max_lags,
                     scale_type=args.scale_type,
                     detrend=getattr(span, detrend), nan_auto=args.nan_auto)
    return xc


def _build_plot_filename(tank):
    raise NotImplementedError()


class CorrelationAnalyzer(Analyzer):
    def _run(self, args):
        tank, spikes = self._load_data(return_tank=True)
        xc = self._compute_xcorr(spikes, args)

        if args.display:
            plot_filename = _build_plot_filename(tank)
            self._display_xcorr(xc, plot_filename)

    def _display_xcorr(self, xc, plot_filename):
        pass


class IPythonAnalyzer(Analyzer):
    """Drop into an IPython shell given a filename or database id number"""
    def _run(self, args):
        import IPython as ipy
        tank, spikes = self._load_data(return_tank=True)
        ipy.embed_kernel(local_ns={'tank': tank, 'raw': tank.spikes})


class BaseConverter(object):
    store_index = False
    store_fs = False

    def __init__(self, base_type, precision, date):
        self.base_type, self.precision = base_type, precision
        self.dtype = np.dtype(self.__base_type + str(self._precision))
        self.date = date

    def split_data(self, raw):
        shank = raw.columns.get_level_values('shank').values
        channels = raw.columns.get_level_values('channels').values
        index = raw.index.values
        values = raw.values
        fs = raw.fs
        date = self.date
        elapsed = (raw.index.freq.n +
                   np.zeros(raw.nsamples)).cumsum().astype('m8[ns]')
        return locals()

    def convert(self, raw, outfile):
        if not self.store_index:
            raw.sortlevel('channel', axis=1, inplace=True)

        if not self.store_fs:
            base, ext = os.path.splitext(outfile)
            outfile = '{base}{fs}{extsep}{ext}'.format(base=base, fs=raw.fs,
                                                       extsep=os.extsep,
                                                       ext=ext)
        self._convert(raw, outfile)


class NeuroscopeConverter(BaseConverter):
    def _convert(self, raw, outfile):
        from bottleneck import nanmax
        max_prec = float(2 ** (self.precision * CHAR_BIT - 1) - 1)
        const = max_prec / nanmax(np.abs(raw))
        xc = raw.values * const
        xc.astype(self.dtype).tofile(outfile)


class H5Converter(BaseConverter):
    store_index = True
    store_fs = True

    def _convert(self, raw, outfile):
        raw.to_hdf(outfile, 'raw')


class NumPyConverter(BaseConverter):
    store_index = True
    store_fs = True

    def _convert(self, raw, outfile):
        split = self.split_data(raw)
        values = split['values']

        if self.dtype != values.dtype:
            split['values'] = values.astype(self.dtype)

        np.savez(outfile, **split)


class MATLABConverter(BaseConverter):
    store_index = True
    store_fs = True

    def _convert(self, raw, outfile):
        from scipy.io import savemat
        savemat(outfile, self.split_data(raw))


_converters = {'neuroscope': NeuroscopeConverter, 'matlab': MATLABConverter,
               'h5': H5Converter, 'numpy': NumPyConverter}


class Converter(SpanCommand):
    def _run(self, args):
        spikes = self._load_data()
        converter = _converters[args.format](args.base_dtype, args.precision)
        converter.convert(spikes, args.outfile)


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


def build_analyze_parser(subparsers):
    def build_correlation_parser(subparsers):
        parser = subparsers.add_parser('correlation', help='perform cross '
                                       'correlation analysis on a recording')
        add_filename_and_id_to_parser(parser)
        parser.add_argument('-d', '--display', action='store_true')
        parser.add_argument('-t', '--threshold', type=float)
        parser.add_argument('-r', '--refractory-period', type=int, default=2)
        parser.add_argument('-b', '--bin-size', type=int)
        parser.add_argument('-p', '--bin-method', default='sum')
        parser.add_argument('-s', '--scale-type', choices=('normalize', 'none',
                                                           'biased',
                                                           'unbiased'),
                            default='normalize')
        parser.add_argument('-m', '--detrend', choices=('mean', 'linear',
                                                        'none'),
                            default='mean')
        parser.add_argument('-l', '--max-lags', type=int, default=1)
        parser.add_argument('-n', '--nan-auto', action='store_true',
                            default=True)
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
    parser.set_defaults(run=Viewer().run)


def build_db_parser(subparsers):
    def _add_args_to_parser(parser):
        parser.add_argument('-a', '--age', type=int, help='the age of the '
                            'animal')
        parser.add_argument('-c', '--condition', help='the experimental '
                            'condition, if any')
        parser.add_argument('-w', '--weight', type=float, help='the weight '
                            'of the animal')
        parser.add_argument('-b', '--bad', action='store_true', help='Mark '
                            'a recording as "good"')

    def build_db_create_parser(subparsers):
        parser = subparsers.add_parser('create', help='put a new recording in '
                                       'the database')
        add_filename_and_id_to_parser(parser)
        _add_args_to_parser(parser)
        parser.set_defaults(run=DbCreator().run)

    def build_db_read_parser(subparsers):
        parser = subparsers.add_parser('read', help='query the properties of a'
                                       ' recording')
        add_filename_and_id_to_parser(parser)
        _add_args_to_parser(parser)
        parser.set_defaults(run=DbReader().run)

    def build_db_update_parser(subparsers):
        parser = subparsers.add_parser('update', help='update the properties '
                                       'of an existing recording')
        add_filename_and_id_to_parser(parser)
        _add_args_to_parser(parser)
        parser.set_defaults(run=DbUpdater().run)

    def build_db_delete_parser(subparsers):
        parser = subparsers.add_parser('delete', help='delete a recording or '
                                       'recordings matching certain '
                                       'conditions')
        add_filename_and_id_to_parser(parser)
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
    group.add_argument('-f', '--filename')
    group.add_argument('-i', '--id', type=int)


def main():
    parser = argparse.ArgumentParser(description='analyze TDT tank files')
    subparsers = parser.add_subparsers(help='subcommands for analying TDT '
                                       'tank files')
    build_analyze_parser(subparsers)
    build_convert_parser(subparsers)
    build_view_parser(subparsers)
    build_db_parser(subparsers)
    args = parser.parse_args()
    return args.run(args)


if __name__ == '__main__':
    sys.exit(main())
