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
import argparse
import warnings

import scipy.io
import span


def serv2mat(df, fs, output_filename):
    """Wrapper for `scipy.io.savemat`.

    Parameters
    ----------
    df : DataFrame
    output_filename : str
    """
    scipy.io.savemat(output_filename, {'data': df, 'fs': fs})


def convert_and_save(filename):
    base_filename, _ = os.path.splitext(filename)

    print '\nConverting TDT Tank to MATLAB: {0}'.format(base_filename)
    tank = span.PandasTank(base_filename)
    sp = tank.spik
    fs = tank.fs['Spik']
    sp = sp.reorder_levels((1, 0), axis=1)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FutureWarning)
        sp.sort_index(axis=1, inplace=True)

    serv2mat(sp.values, fs, base_filename)
    print 'Done!'


def convert_and_save_multiple(filenames):
    for filename in filenames:
        convert_and_save(filename)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Convert TDT tank files to MATLAB arrays')
    parser.add_argument('filenames', nargs='*',
                        help='A file name or group of file names from the '
                        'server that contains the data of interest')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    convert_and_save_multiple(args.filenames)
