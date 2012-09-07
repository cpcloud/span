#!/usr/bin/env python

"""
"""

import os
import sys
import abc
import argparse
import getpass
import ftplib
import shutil
import multiprocessing
import concurrent
import concurrent.futures

import numpy as np
import scipy
import scipy.io

import tdt
import server


CPU_COUNT = multiprocessing.cpu_count()


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


def download_files(server, filenames):
    """Download files from the arod server using multithreading.

    Parameters
    ----------
    server : ArodServer
    filenames : sequence
        Files to download from the server

    Returns
    -------
    r : list
        Names of the downloaded files
    """
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=cpu_count) as e:
        r = [e.submit(server.download_file, filename) for filename in filenames]
    return map(lambda x: x.result(), r)
    

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
    server = server.ArodServer()

    # download the files from the server
    local_tev, local_tsq = download_files(server, (tev_fn, tsq_fn))

    # make the file names
    tank_base, _ = os.path.splitext(local_tev)
    tank_dir = os.path.join(os.getcwd(), os.path.basename(tank_base))
    
    mat_filename = os.path.join(os.getcwd(),
                                os.path.basename(dn) + os.extsep + 'mat')

    print('\nConverting TDT Tank to MATLAB...')

    # save to the current directory
    serv2mat(tdt.PandasTank(tank_dir).spikes.channels.values, mat_filename)

    # get rid of the extra files
    os.remove(local_tsq)
    os.remove(local_tev)
    

if __name__ == '__main__':
    main()
