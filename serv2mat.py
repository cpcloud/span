#!/usr/bin/env python

import os
import sys
import argparse
import warnings
import getpass
import contextlib
import ftplib
import shutil

sys.path.append(os.path.expanduser(os.path.join('~', 'code', 'py')))

import numpy as np

from scipy.io import savemat

import tdt

with warnings.catch_warnings():
    warnings.filterwarnings('ignore', category=UserWarning)
    import pandas as pd


class ProgressBar(object):
    def __init__(self, start=0, end=10, width=12, fill='#', blank='.',
                 format='[{fill}{blank}] {progress}%', incremental=True):
        super(ProgressBar, self).__init__()
        self.start = start
        self.end = end
        self.width = width
        self.fill = fill
        self.blank = blank
        self.format = format
        self.incremental = incremental
        self.step = 100.0 / float(width)
        self.reset()

    def __iadd__(self, increment):
        increment = self._get_progress(increment)
        if self.progress + increment < 100:
            self.progress += increment
        else:
            self.progress = 100.0
        return self

    def __str__(self):
        progressed = int(self.progress / self.step)
        fill = progressed * self.fill
        blank = (self.width - progressed) * self.blank
        return self.format.format(fill=fill, blank=blank,
                                  progress=int(self.progress))

    __repr__ = __str__

    def _get_progress(self, increment):
        return float(increment * 100.0) / self.end

    def reset(self):
        self.progress = self._get_progress(self.start)
        return self


class AnimatedProgressBar(ProgressBar):
    def __init__(self, *args, **kwargs):
        super(AnimatedProgressBar, self).__init__(*args, **kwargs)
        self.stdout = kwargs.get('stdout', sys.stdout)

    def show_progress(self):
        c = '\n'
        if hasattr(self.stdout, 'isatty') and self.stdout.isatty():
            c = '\r'
            
        self.stdout.write(c)
        self.stdout.write(str(self))
        self.stdout.flush()


class ArodServer(object):
    def __init__(self, username='Adrian'):
        super(ArodServer, self).__init__()
        self.username, self.password = username, getpass.getpass()
        self.ftp = ftplib.FTP('192.168.70.4', self.username, self.password)

    def download_file(self, filename, ipaddr='192.168.70.4', port=22,
                      username='Adrian', hostkey=None, verbose=True):
        filename = '/home/' + filename.lstrip('~')
        local_path = os.path.join(os.getcwd(), os.path.basename(filename))
        self.progress_bar = AnimatedProgressBar(end=self.ftp.size(filename),
                                                width=50)
        print '%s' % os.path.basename(local_path)
        
        with open(local_path, 'wb') as local_file:
            def callback(chunk):
                local_file.write(chunk)
                self.progress_bar += len(chunk)
                self.progress_bar.show_progress()
            self.ftp.retrbinary('RETR {0}'.format(filename), callback)
        return local_path

    def __str__(self):
        return '<ArodServer object at 0x%x>' % id(self)

    __repr__ = __str__


def serv2mat(raw, output_filename):
    savemat(output_filename, {'data': raw}, oned_as='row')


def parse_args():
    parser = argparse.ArgumentParser(description='convert TDT to MATLAB')
    parser.add_argument('dirname', metavar='DIRNAME', type=str,
                        help='a directory name from the server')
    return parser.parse_args()


def download_files(server, filenames):
    local_names = []
    
    for filename in filenames:
        print
        local_names.append(server.download_file(filename))
        print
        
    return local_names
    

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
    server = ArodServer()

    # download the files from the server
    local_tev, local_tsq = download_files(server, [tev_fn, tsq_fn])

    # make the file names
    tank_base, _ = os.path.splitext(local_tev)
    tank_dir = os.path.join(os.getcwd(), os.path.basename(tank_base))
    
    mat_filename = os.path.join(os.getcwd(),
                                os.path.basename(dn) + os.extsep + 'mat')

    print
    print 'Converting TDT Tank to MATLAB...'

    # save to the current directory
    serv2mat(tdt.PandasTank(tank_dir).spikes.channels.values, mat_filename)

    # get rid of the extra files
    os.remove(local_tsq)
    os.remove(local_tev)
    

if __name__ == '__main__':
    main()
    print
