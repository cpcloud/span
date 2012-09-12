#!/usr/bin/env python

"""
"""

import os
import argparse

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
    parser = argparse.ArgumentParser(description='convert TDT to MATLAB')
    parser.add_argument('dirname', metavar='DIRNAME', type=str,
                        help='a directory name from the server')
    return parser.parse_args()

    
def main():
    # parse the arguments
    args = parse_args()
    dn = args.dirname
    
    mat_filename = dnbn + os.extsep + 'mat'

    print '\nConverting TDT Tank to MATLAB: {}'.format(mat_filename)
    
    # save to the current directory
    serv2mat(span.tdt.PandasTank(dn).spikes.raw, mat_filename)
    print 'Done!'

    # get rid of the extra mat file from the server
    os.remove(dn + os.extsep + 'mat')


if __name__ == '__main__':
    main()     
