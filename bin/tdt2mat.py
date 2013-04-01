#!/usr/bin/env python

# serv2mat.py ---

# Copyright (C) 2012 Copyright (C) 2012 Phillip Cloud <cpcloud@gmail.com>

# Author: Phillip Cloud <cpcloud@gmail.com>

# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.


import os
import warnings

with warnings.catch_warnings():
    warnings.simplefilter('ignore', UserWarning)
    import argparse

import scipy.io
import numpy as np
import span


def serv2mat(df, fs, output_filename):
    """Wrapper for `scipy.io.savemat`.

    Parameters
    ----------
    df : DataFrame
    output_filename : str
    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FutureWarning)
        scipy.io.savemat(output_filename + '_' + str(fs), {'data': df})


def serv2bin(df, fs, filename, ext='dat'):
    kws = dict(filename=filename, fs=fs, extsep=os.extsep, ext=ext)
    np.asfortranarray(df).tofile('{filename}_{fs}{extsep}{ext}'.format(**kws))


def convert_and_save(filename, file_type, electrode_map, clean, ws, bs):
    base_filename, _ = os.path.splitext(filename)

    print '\nConverting TDT Tank to MATLAB: {0}'.format(base_filename)

    if electrode_map == 'nn4x4':
        electrode_map = span.ElectrodeMap(span.NeuroNexusMap.values - 1, ws,
                                          bs)

    tank = span.PandasTank(base_filename, electrode_map, clean)
    sp = tank.spik
    fs = sp.fs
    sp = sp.reorder_levels((1, 0), axis=1)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FutureWarning)
        sp.sort_index(axis=1, inplace=True)

    ft_funcs = {'m': serv2mat, 'b': serv2bin, 'matlab': serv2mat,
                'binary': serv2bin}
    ft_funcs[file_type](sp.values, fs, base_filename)
    print 'Done!'


def convert_and_save_multiple(filenames, dry_run, file_type, electrode_map,
                              clean, ws, bs):
    for filename in filenames:
        if dry_run:
            print filename
            assert os.path.exists(filename)
        else:
            convert_and_save(filename, file_type, electrode_map, clean, ws, bs)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Convert TDT tank files to MATLAB or binary arrays')

    parser.add_argument('filenames', nargs='+',
                        help='A file name or list of file names from the '
                        'server that contains the data of interest')
    parser.add_argument('-d', '--dry-run', action='store_true',
                        help='Perform a dry run to make sure arguments are '
                        'correct and that files exist')
    parser.add_argument('-c', '--clean', action='store_true',
                        help='Clean the data by removing the first '
                        'principal component, if this flag is given')
    parser.add_argument('-t', '--file-type', help='Output file type,'
                        ' defaults to matlab', choices=('b', 'm', 'binary',
                                                        'matlab'),
                        default='matlab')
    parser.add_argument('-m', '--electrode-map',
                        help='The name of an electrode map, defaults to nn4x4',
                        default='nn4x4')
    parser.add_argument('-w', '--within-shank', type=float,
                        help='Within shank distance')
    parser.add_argument('-b', '--between-shank', type=float,
                        help='Between shank distance')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    convert_and_save_multiple(args.filenames, args.dry_run, args.file_type,
                              args.electrode_map, args.clean, args.ws, args.bs)
