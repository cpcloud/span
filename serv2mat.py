#!/usr/bin/env python

"""
"""

import os
import sys
import argparse

import numpy as np
import scipy
import scipy.io

import tdt
import server as serv


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
    d = args.dirname
    dn = d.rstrip(os.sep) if d[-1] == os.sep else d
    bn = 'Spont_Spikes_' + os.path.basename(dn)

    bn_base = bn + os.extsep
    tev_fn = os.path.join(dn, bn_base + 'tev')
    tsq_fn = os.path.join(dn, bn_base + 'tsq')

    # init the server
    server = serv.ArodServer()

    # download the files from the server
    local_tev, local_tsq = download_files(server, [tev_fn, tsq_fn])

    # make the file names
    tank_base, _ = os.path.splitext(local_tev)
    tank_base = os.path.join(os.getcwd(), os.path.basename(tank_base))
    
    mat_filename = os.path.join(os.getcwd(),
                                os.path.basename(dn) + os.extsep + 'mat')

    print('\nConverting TDT Tank to MATLAB...')

    # save to the current directory
    serv2mat(tdt.PandasTank(tank_base).spikes.raw, mat_filename)

    # get rid of the extra files
    os.remove(local_tsq)
    os.remove(local_tev)
    

if __name__ == '__main__':
    main()
