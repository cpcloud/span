#!/usr/bin/env python

import sys
import os
import argparse
import numbers
import subprocess
import collections
import glob
import tarfile
import re
import tempfile
from functools import partial

import numpy as np
from numpy.random import rand
import pandas as pd

from scipy.io import savemat
from scipy.constants import golden as golden_ratio

from IPython import embed

from bottleneck import nanmax

from clint.textui import puts
from clint.textui.colored import red, blue, green, magenta, white
from clint.packages.colorama import Style

from lxml.builder import ElementMaker
from lxml import etree

from dateutil.parser import parse as _parse_date

import sh

from span import TdtTank, NeuroNexusMap, ElectrodeMap

git = sh.git

CHAR_BIT = 8

HOME = os.environ.get('HOME', os.path.expanduser('~'))
SPAN_DB_PATH = os.environ.get('SPAN_DB_PATH', os.path.join(HOME, '.spandb'))
SPAN_DB_NAME = os.environ.get('SPAN_DB_NAME', 'db')
SPAN_DB_EXT = os.environ.get('SPAN_DB_EXT', 'csv')
SPAN_DB = os.path.join(SPAN_DB_PATH, '{0}{1}{2}'.format(SPAN_DB_NAME,
                                                        os.extsep,
                                                        SPAN_DB_EXT))


def hsv_to_rgb(h, s, v):
    hi = int(h * 6)
    f = h * 6 - hi
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    m = {0: (v, t, p), 1: (q, v, p), 2: (p, v, t), 3: (p, q, v), 4: (t, p, v),
         5: (v, p, q)}
    return '#{0:0>2x}{1:0>2x}{2:0>2x}'.format(*np.int64(256 * np.array(m[hi])))


def randcolor(h, s, v):
    if h is None:
        h = rand()
    h += golden_ratio - 1
    h %= 1
    return hsv_to_rgb(h, s, v)


def randcolors(ncolors, hue=None, saturation=0.99, value=0.99):
    colors = np.empty(ncolors, dtype=object)
    for i in xrange(ncolors):
        colors[i] = randcolor(hue, saturation, value)
    return colors


def error(msg):
    errmsg = green(os.path.basename(__file__))
    errmsg += '{0} {1}{0} {2}'.format(white(':'), red('error'), magenta(msg))
    puts(bold(errmsg))
    return sys.exit(2)


def _get_fn(id_num_or_filename, db_path=SPAN_DB):
    if db_path is None:
        error('SPAN_DB_PATH environment variable not set, please set via '
              '"export SPAN_DB_PATH=\'path_to_the_span_database\'"')
    db_path = os.path.abspath(db_path)
    db = pd.read_csv(db_path)
    _pop_column_to_name(db, 0)

    if isinstance(id_num_or_filename, numbers.Integral):
        if id_num_or_filename not in db.index:
            error(bold('{0} {1}'.format(blue('"' + str(id_num_or_filename) +
                                             '"'),
                                        red('is not a valid id number'))))
    elif isinstance(id_num_or_filename, basestring):
        if id_num_or_filename not in db.filename.values:
            error(bold('{0} {1}'.format(blue('"' + str(id_num_or_filename) +
                                             '"'),
                                        red('is not a valid filename'))))
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

        paths = glob.glob(self.filename + '*')
        common_prefix = os.path.commonprefix(paths)
        self.filename = common_prefix.strip(os.extsep)

        if not paths:
            return error('No paths match the expression '
                         '"{0}*"'.format(self.filename))

    def run(self, args):
        self._parse_filename_and_id(args)
        return self._run(args)

    def _run(self, args):
        raise NotImplementedError()

    def _load_data(self):
        full_path = os.path.join(SPAN_DB_PATH, 'h5', 'raw', str(self.id) +
                                 '.h5')
        with pd.get_store(full_path, mode='a') as raw_store:
            try:
                spikes = raw_store.get('raw')
                meta = self._get_meta(self.id)
            except KeyError:
                em = ElectrodeMap(NeuroNexusMap.values, 50, 125)
                tank = TdtTank(os.path.normpath(self.filename), em)
                spikes = tank.spik
                raw_store.put('raw', spikes)
                meta = self._get_meta(tank)

        return meta, spikes

    def _get_meta(self, obj):
        return {TdtTank: self._get_meta_from_tank,
                int: self._get_meta_from_id}[type(obj)](obj)

    def _get_meta_from_id(self, id_num):
        return self.db.iloc[id_num]

    def _get_meta_from_tank(self, tank):
        meta = self._append_to_db_and_commit(tank)
        return meta

    def _append_to_db_and_commit(self, tank):
        row = _row_from_tank(tank)
        self._append_to_db(row)
        self._commit_to_db()
        return row

    def _append_to_db(self, row):
        pass

    def _commit_to_db(self):
        pass


def _row_from_tank(tank):
    pass


class Analyzer(SpanCommand):
    pass


def _get_from_db(dirname, rec_num, method, *args):
    full_path = os.path.join(SPAN_DB_PATH, 'h5', dirname, str(rec_num) + '.h5')
    key = str(hash(args))

    with pd.get_store(full_path, mode='a') as raw_store:
        try:
            out = raw_store.get(key)
        except KeyError:
            out = method(*args)
            raw_store.put(key, out)
    return out


def _compute_xcorr(args):
    import span
    from span import spike_xcorr

    rec_num = args.id
    detrend = getattr(span, 'detrend_' + args.detrend)

    raw = _get_from_db('raw', rec_num)
    thr = _get_from_db('thr', rec_num, raw.threshold, args.threshold)
    cleared = _get_from_db('clr', rec_num, partial(thr.clear_refrac,
                                                   inplace=True),
                           args.refractory_period)
    binned = _get_from_db('binned', rec_num, cleared.bin, args.bin_size,
                          args.bin_method)
    xc = _get_from_db('xcorr', rec_num, partial(spike_xcorr, binned),
                      args.max_lags, args.scale_type, detrend, args.nan_auto)
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
        tank, spikes = self._load_data(return_tank=True)
        embed()
        return 0


class BaseConverter(object):
    store_index = False

    def __init__(self, base_type, precision, date):
        self.base_type, self.precision = base_type, precision
        self.dtype = np.dtype(self.base_type + str(self.precision))
        self.date = date

    def split_data(self, raw):
        shank = raw.columns.get_level_values('shank').values
        channels = raw.columns.get_level_values('channel').values
        index = raw.index.values
        values = raw.values
        fs = raw.fs
        date = self.date
        elapsed = (raw.index.freq.n *
                   np.zeros(raw.nsamples)).cumsum().astype('timedelta64[ns]')
        return locals()

    def convert(self, raw, outfile):
        if not self.store_index:
            raw.sortlevel('channel', axis=1, inplace=True)

        self._convert(raw, outfile)


class NeuroscopeConverter(BaseConverter):

    def _convert(self, raw, outfile):
        max_prec = float(2 ** (self.precision * CHAR_BIT - 1) - 1)
        const = max_prec / nanmax(np.abs(raw.values))
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
        savemat(outfile, self.split_data(raw))


_converters = {'neuroscope': NeuroscopeConverter, 'matlab': MATLABConverter,
               'h5': H5Converter, 'numpy': NumPyConverter}


class Converter(SpanCommand):

    def _run(self, args):
        spikes = self._load_data()
        converter = _converters[args.format](args.base_dtype, args.precision)
        converter.convert(spikes, args.outfile)


def _build_anatomical_description_element(index, E):
    anatomicalDescription = E.anatomicalDescription
    channelGroups = E.channelGroups
    group = E.group
    channel = E.channel
    groups = collections.defaultdict(list)
    for shank, channel in index:
        groups[shank].append(E.channel(str(channel)))
    items = groups.items()
    items.sort(key=lambda x: x[0])
    grouplist = []
    for gn, grp in items:
        grouplist.append(group(*grp))
    return anatomicalDescription(channelGroups(*grouplist))


def _build_spike_detection_element(index, E):
    spikeDetection = E.spikeDetection
    channelGroups = E.channelGroups
    group = E.group
    channel = E.channel
    groups = collections.defaultdict(list)
    for shank, channel in index:
        groups[shank].append(E.channel(str(channel), skip='0'))
    items = groups.items()
    items.sort(key=lambda x: x[0])
    grouplist = []
    for gn, grp in items:
        grouplist.append(group(*grp))
    return spikeDetection(channelGroups(*grouplist))


def _build_channels_element(index, E, colors):
    def _build_single_channel_color(channel, color):
        return E.channelColors(
            E.channel(channel),
            E.color(color),
            E.anatomyColor(color),
            E.spikeColor(color)
        )

    def _build_single_channel_offset(channel):
        return E.channelOffset(
            E.channel(channel),
            E.defaultOffset('0')
        )

    elements = []

    for shank, channel in index:
        c = str(channel)
        elements.append(_build_single_channel_color(c, colors[shank]))
        elements.append(_build_single_channel_offset(c))
    return E.channels(*elements)


def _make_neuroscope_xml(spikes, base, precision, voltage_range, amp, tarfile):
    E = ElementMaker()
    parameters = E.parameters
    acquisitionSystem = E.acquisitionSystem
    nBits = E.nBits
    nChannels = E.nChannels
    samplingRate = E.samplingRate
    voltageRange = E.voltageRange
    amplification = E.amplification
    offset = E.offset
    columns = spikes.columns
    colors = randcolors(columns.get_level_values('shank').unique().size)

    doc = parameters(
        acquisitionSystem(
            nBits(str(precision)),
            nChannels(str(spikes.nchannels)),
            samplingRate(str(int(spikes.fs))),
            voltageRange(str(voltage_range)),
            amplification(str(amp)),
            offset('0')
        ),

        E.fieldPotentials(
            E.lfpSamplingRate('1250')
        ),

        _build_anatomical_description_element(columns, E),
        _build_spike_detection_element(columns, E),

        E.neuroscope(
            E.miscellaneous(
                E.screenGain('0.2'),
                E.traceBackgroundImage()
            ),

            E.video(
                E.rotate('0'),
                E.flip('0'),
                E.videoImage(),
                E.positionsBackground('0')
            ),

            E.spikes(
                E.nsamples('72'),
                E.peakSampleIndex('36')
            ),

            _build_channels_element(columns, E, colors),
            version='1.3.3'
        ),
        creator='spanner.py',
        version='0.1'
    )

    filename = base + os.extsep + 'xml'

    with open(filename, 'w') as f:
        f.write(etree.tostring(doc, pretty_print=True))

    tarfile.add(filename)
    os.remove(filename)


def _make_neuroscope_nrs(spikes, base, start_time, window_size, tarfile):
    def _build_channel_positions():
        return (
            E.channelPosition(
                E.channel(
                    str(channel)
                ),
                E.gain('10'),
                E.offset('0')
            ) for channel in channels
        )
    E = ElementMaker()
    channels = spikes.columns.get_level_values('channel')

    doc = E.neuroscope(
        E.files(),
        E.displays(
            E.display(
                E.tabLabel('Field Potentials Display'),
                E.showLabels('0'),
                E.startTime(str(start_time)),
                E.duration(str(window_size)),
                E.multipleColumns('0'),
                E.greyScale('0'),
                E.positionView('0'),
                E.showEvents('0'),
                E.spikePresentation('0'),
                E.rasterHeight('33'),
                E.channelPositions(
                    *_build_channel_positions()
                ),
                E.channelsSelected(),
                E.channelsShown(
                    *(E.channel(str(channel)) for channel in channels)
                )
            )
        )
    )

    filename = base + os.extsep + 'nrs'

    with open(filename, 'w') as f:
        f.write(etree.tostring(doc, pretty_print=True))

    tarfile.add(filename)
    os.remove(filename)


def _build_neuroscope_package(spikes, converter, base, outfile, zipped_name,
                              args):
    tarfile_name = base + os.extsep + 'tar{0}{1}'.format(os.extsep,
                                                         args.format)
    with tarfile.open(tarfile_name, 'w:{0}'.format(args.format)) as f:
        converter.convert(spikes, outfile)
        f.add(outfile)
        os.remove(outfile)
        _make_neuroscope_xml(spikes, base, args.precision, args.voltage_range,
                             args.amplification, f)
        _make_neuroscope_nrs(spikes, base, args.start_time, args.window_size,
                             f)


def _get_dat_from_tarfile(tarfile):
    members = tarfile.getmembers()
    names = [member.name for member in members]
    for name, member in zip(names, members):
        if name.endswith('.dat'):
            return member
    else:
        return error('no DAT file found in neuroscope package. files found '
                     'were {1}'.format(names))


def _run_neuroscope(tarfile):
    member = _get_dat_from_tarfile(tarfile)
    tarfile.extractall()
    try:
        return subprocess.check_call(['neuroscope', os.path.join(os.curdir,
                                                                 member.name)])
    except OSError:
        return error('could not find neuroscope on the system path, it is '
                     'probably not '
                     'installed\n\nPATH={0}'.format(os.environ.get('PATH')))
    except subprocess.CalledProcessError as e:
        return error(e.msg)


# get the filename/id, convert to neuroscope with int16 precision, zip into
# package, unzip and show in neuroscope
class Viewer(SpanCommand):

    def _run(self, args):
        tank, spikes = self._load_data(return_tank=True)
        base, _ = os.path.splitext(self.filename)
        base = os.path.join(os.curdir, os.path.basename(base))
        outfile = '{base}{extsep}dat'.format(base=base, extsep=os.extsep)
        converter = _converters['neuroscope']('int', 16, tank.datetime)
        args.precision = converter.precision
        zipped_name = '{0}{1}tar{1}{2}'.format(base, os.extsep, args.format)
        _build_neuroscope_package(spikes, converter, base, outfile,
                                  zipped_name, args)
        with tarfile.open(zipped_name,
                          mode='r:{0}'.format(args.format)) as r_package:
            _run_neuroscope(r_package)


def _commit_if_changed(version):
    if git('diff-index', '--quiet', 'HEAD', '--').exit_code:
        git.commit(message="'version {0}'".format(version))


def _create_new_db(db, path):
    try:
        os.remove(path)
    except OSError:
        # path doesn't exist
        pass

    db.to_csv(path, index_label=db.columns.names)
    dbdir = os.path.dirname(path)
    curdir = os.getcwd()
    git.init(dbdir)
    os.chdir(dbdir)
    git.add(path)

    # something has changed
    _commit_if_changed(version=0)
    os.chdir(curdir)


def _init_db(path, dbcls):
    """initialize the database"""
    if not hasattr(dbcls, 'schema'):
        raise AttributeError('class {0} has no schema attribute'.format(dbcls))
    schema = sorted(dbcls.schema)
    name = 'field'
    columns = pd.Index(schema, name=name)
    empty_db = pd.DataFrame(columns=columns).sort_index(axis=1)

    try:
        current_db = pd.read_csv(path)
        current_db.pop(name)
        current_db.columns.name = name
    except IOError:
        # db doesn't exist create it
        _create_new_db(empty_db, path)
    else:
        if not (current_db.columns != columns).all():
            _create_new_db(empty_db, path)


def _build_query_index(db, query_dict):
    res = pd.Series(np.ones(db.shape[0], dtype=bool))

    try:
        it = query_dict.iteritems()
    except AttributeError:
        it = query_dict

    for column, value in it:
        if column in db and value is not None:
            res &= getattr(db, column) == value

    return res


def _query_dict_from_args(args):
    return args._get_kwargs()


def _pop_column_to_name(df, column):
    try:
        df.columns.name = df.pop(column).name
    except KeyError:
        df.columns.name = df.pop(df.columns[column]).name


def bold(s):
    return '{0}{1}{2}'.format(Style.BRIGHT, s, Style.RESET_ALL)


def _df_prettify(df):
    df = df.copy()
    s = bold(df.to_string())

    for colname in df.columns:
        s = s.replace(colname, str(blue(colname)))
    s = s.replace(df.index.name, str(red(df.index.name)))
    return df.to_string()


def _df_pager(df_string):
    with tempfile.NamedTemporaryFile() as tmpf:
        tmpf.write(df_string)

        try:
            return subprocess.check_call([os.environ.get('PAGER', 'less'),
                                          tmpf.name], stdin=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            return error(e.msg)


class Db(SpanCommand):
    schema = ('artifact_ranges', 'weight', 'between_shank', 'valid_recording',
              'age', 'probe_number', 'site', 'filename', 'within_shank',
              'date', 'animal_type', 'shank_order', 'condition')

    def _parse_filename_and_id(self, args):
        self.db = pd.read_csv(SPAN_DB)
        self.db.columns.name = self.db.pop(self.db.columns[0]).name


class DbCreator(Db):
    """Create a new entry in the recording database"""
    def _run(self, args):
        self.validate_args(args)
        self.append_new_row(self.make_new_row(args))
        self.commit_changes()

    def validate_args(self, args):
        pass

    def append_new_row(self, new_row):
        pass

    def commit_changes(self):
        pass


class DbReader(Db):
    """Read, retrieve, search, or view existing entries"""
    def _run(self, args):
        if self.db.empty:
            return error('no recordings in database named '
                         '"{0}"'.format(SPAN_DB))
        else:
            query_dict = _query_dict_from_args(args)
            indexer = _build_query_index(self.db, query_dict)
            return _df_pager(_df_prettify(self.db[indexer]))


class DbUpdater(Db):

    """Edit an existing entry or entries"""
    def _run(self, args):
        pass


class DbDeleter(Db):

    """Remove an existing entry or entries"""
    def _run(self, args):
        pass


def _colon_to_slice(c):
    splitter = re.compile(r'\s*:\s*')
    start, stop = splitter.split(c)
    return slice(int(start), int(stop))


def _parse_artifact_ranges(s):
    splitter = re.compile(r'\s*,\s*')
    split = splitter.split(s)
    return [_colon_to_slice(spl) for spl in split]


def build_analyze_parser(subparsers):
    def build_correlation_parser(subparsers):
        parser = subparsers.add_parser('correlation', help='perform cross '
                                       'correlation analysis on a recording',
                                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        add_filename_and_id_to_parser(parser)
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
            '-t', '--threshold', type=float, required=True,
            help='threshold in multiples of the standard deviation of the voltage data')
        thresholding.add_argument(
            '-r', '--refractory-period', type=int, default=2, help='refractory period in milliseconds')
        binning.add_argument(
            '-b', '--bin-size', type=int, default='1S', help='bin size in some time unit')
        binning.add_argument(
            '-p', '--bin-method', default='sum', help='function to use for binning spikes')
        xcorr.add_argument(
            '-s', '--scale-type', choices=('normalize', 'none', 'biased',
                                           'unbiased'), default='normalize', help='type of scaling to use on the raw cross correlation')
        xcorr.add_argument(
            '-m', '--detrend', choices=('mean', 'linear', 'none'),
            default='mean', help='function to use to detrend the raw cross correlation')
        xcorr.add_argument('-l', '--max-lags', type=int, default=1,
                           help='maximum number of lags of the cross correlation to return')
        xcorr.add_argument(
            '-k', '--keep-auto', action='store_true', help='keep the autocorrelation values')
        parser.set_defaults(run=CorrelationAnalyzer().run)

    def build_ipython_parser(subparsers):
        parser = subparsers.add_parser('ipython', help='drop into an ipython '
                                       'shell')
        add_filename_and_id_to_parser(parser)
        parser.add_argument('-c', '--remove-first-pc', action='store_true',
                            help='remove the first principal component of the data. warning: this drastically slows down the analysis')
        parser.set_defaults(run=IPythonAnalyzer().run)

    parser = subparsers.add_parser('analyze', help='perform an analysis on a '
                                   'TDT tank file')
    subparsers = parser.add_subparsers()
    build_correlation_parser(subparsers)
    build_ipython_parser(subparsers)


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
    build_view_parser(subparsers)
    build_db_parser(subparsers)
    args = parser.parse_args()
    return args.run(args)


if __name__ == '__main__':
    _init_db(SPAN_DB, Db)
    sys.exit(main())
