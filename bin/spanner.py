#!/usr/bin/env python

import sys
import argparse as argp


class SpanCommand(object):
    def __init__(self, **kwargs):
        self.id = kwargs.get('id', None) or kwargs.get('filename', None)
        self.kwargs = kwargs

    def run(self):
        raise NotImplementedError()


# perform some analyses
class Analyzer(SpanCommand):
    pass


# get the filename/id convert to some type with some precision
class Converter(SpanCommand):
    pass


# get the filename/id, convert to neuroscope with int16 precision, zip into
# package, unzip and show in neuroscope
class Viewer(SpanCommand):
    pass


# superclass for db commands
class Db(SpanCommand):
    pass


# query the database
class DbGet(Db):
    pass


# put something in the database
class DbPut(Db):
    pass


# update the database
class DbUpdate(Db):
    pass


def build_analyze_parser(subparsers):
    parser = subparsers.add_parser('analyze', help='Perform an analysis on a '
                                   'TDT tank file')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-f', '--filename', help='The name of the '
                       'folder with tev/tsq files or either a tev or '
                       'tsq file WITHOUT the extension')
    group.add_argument('-n', '--id-number', type=int, help='alternati'
                       'vely use the database id number of the '
                       'recording')
    parser.add_argument('-t', '--type', help='The kind of analysis to perform',
                        choices=('correlation', 'ipython'))
    parser.add_argument('-r', '--threshold', type=float)
    parser.add_argument('-d', '--display', action='store_true')


def build_convert_parser(subparsers):
    convert_parser = subparsers.add_parser('convert',
                                           help='Convert a TDT tank file into '
                                           'a different format')
    convert_parser.add_argument('-f', '--filename', help='The name of the '
                                'folder with tev/tsq files or either a tev or '
                                'tsq file WITHOUT the extension',
                                required=True)
    convert_parser.add_argument('-t', '--type',
                                help='The type of conversion you want to '
                                'perform', choices=('neuroscope', 'matlab',
                                                    'numpy', 'h5'),
                                required=True)
    convert_parser.add_argument('-d', '--base-type',
                                help='The base numeric type to convert to',
                                default='float', choices=('float', 'int',
                                                          'uint'),
                                required=True)
    convert_parser.add_argument('-p', '--precision', help='The number of bits '
                                'to use for conversion', type=int, default=64,
                                choices=(8, 16, 32, 64), required=True)


def build_view_parser(subparsers):
    view_parser = subparsers.add_parser('view',
                                        help='Display the raw traces of a TDT '
                                        'tank file in Neuroscope')
    view_parser.add_argument('-f', '--filename', help='The name of the '
                             'folder with tev/tsq files or either a tev or '
                             'tsq file WITHOUT the extension',
                             required=True)


def build_db_parser(subparsers):
    query_parser = subparsers.add_parser('db',
                                         help='Operate on the database of '
                                         'recordings')
    query_subparsers = query_parser.add_subparsers()
    query_get_parser = query_subparsers.add_parser('get',
                                                   help='Query the properties '
                                                   'of a recording')
    query_put_parser = query_subparsers.add_parser('put',
                                                   help='Put a new recording '
                                                   'in the database')
    query_update_parser = query_subparsers.add_parser('update',
                                                      help='Update the '
                                                      'properties of an '
                                                      'existing recording')
    query_put_parser.add_argument('-f', '--filename', help='The file to add '
                                  'to the database of recordings. Note that '
                                  'this subcommand will attempt to infer the '
                                  'values of the other arguments. It will '
                                  'throw an error if not all of the other '
                                  'arguments can be inferred. If that happens '
                                  'you should provide the values.')
    query_update_parser.add_argument('-a', '--age', type=int,
                                     help='change the age of the animal')
    query_update_parser.add_argument('-c', '--condition',
                                     help='change the experimental condition')
    query_update_parser.add_argument('-d', '--date',
                                     help='change the date of the recording')
    query_update_parser.add_argument('-w', '--weight', type=float,
                                     help='change the weight of the animal')
    query_update_parser.add_argument('-b', '--bad', action='store_true',
                                     help='mark a recording as "good"')
    query_put_parser.add_argument('-a', '--age', type=int,
                                  help='the age of the animal')
    query_put_parser.add_argument('-c', '--condition',
                                  help='the experimental condition, if any')
    query_put_parser.add_argument('-d', '--date',
                                  help='the date of the recording')
    query_put_parser.add_argument('-w', '--weight', type=float,
                                  help='the weight of the animal')
    query_put_parser.add_argument('-b', '--bad', action='store_true',
                                  help='Mark a recording as "good"')
    query_get_parser.add_argument('-a', '--age', nargs='*', type=int,
                                  help='the age of the animal')
    query_get_parser.add_argument('-c', '--condition', nargs='*',
                                  help='The experimental condition, if any')
    query_get_parser.add_argument('-d', '--date', nargs='*',
                                  help='The date of the recording')
    query_get_parser.add_argument('-w', '--weight', nargs='*', type=float,
                                  help='The weight of the animal')
    query_get_parser.add_argument('-g', '--only-good', action='store_true',
                                  help='Only show recordings that have been '
                                  'marked as "good"')




def parse_args():
    parser = argp.ArgumentParser(description='Analyze TDT tank files')
    subparsers = parser.add_subparsers(help='Subcommands for analying TDT tank'
                                       'files')
    build_analyze_parser(subparsers)
    build_convert_parser(subparsers)
    build_view_parser(subparsers)
    build_db_parser(subparsers)
    return parser.parse_args()


def main(args):
    print args


if __name__ == '__main__':
    sys.exit(main(parse_args()))
