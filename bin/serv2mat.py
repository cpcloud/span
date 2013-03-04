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
import glob

try:
    from argparse import ArgumentParser
except ImportError:
    from optparse import OptionParser as ArgumentParser

import scipy.io
import span


def serv2mat(raw, output_filename, name='data'):
    """Wrapper for `scipy.io.savemat`.

    Parameters
    ----------
    raw : array_like
    output_filename : str
    name : str, optional
        Name of the array when loaded into MATLAB.
    """
    scipy.io.savemat(output_filename, {name: raw}, oned_as='row')


def parse_args():
    """Parse command line arguments."""
    parser = ArgumentParser(description='convert TDT to MATLAB')
    try:
        parser.add_argument('dirname', metavar='DIRNAME', type=str,
                            help='a directory name from the server')
    except AttributeError:
        pass

    try:
        _, args = parser.parse_args()
        args = args[0]
    except (TypeError, ValueError):
        args = parser.parse_args()
    return args


def main():
    dn = parse_args()

    try:
        dn = dn.dirname.rstrip(os.sep)
    except:
        pass

    if not os.path.exists(dn):
        raise OSError('%s does NOT exist, make sure you typed the name of '
                      'the directory correctly' % dn)
    tev, = glob.glob(os.path.join(dn, '*.tev'))
    tev_name, _ = os.path.splitext(tev)
    mat_filename = os.path.join(tev_name + os.extsep + 'mat')
    print '\nConverting TDT Tank to MATLAB: {0}'.format(mat_filename)
    serv2mat(span.tdt.PandasTank(tev_name).spik.values,
             mat_filename)
    print 'Done!'


if __name__ == '__main__':
    main()
