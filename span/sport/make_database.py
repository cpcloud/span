#!/usr/bin/env python

import os
import re
from glob import glob
import itertools


import numpy as np
import tables


class Recording(tables.IsDescription):
    """Encapsulates a recording.

    Each field must be have a minimum length of 4 so that
    the 'None' string can be seen.
    """
    date = tables.atom.StringCol(10) # any iso date format is valid?
    age = tables.atom.StringCol(4) # e.g., p# where # is any number
    animal = tables.atom.StringCol(5) # 'rat' or 'mouse'
    site = tables.atom.StringCol(4)
    season_litter = tables.atom.StringCol(4)
    depth = tables.atom.StringCol(6)
    direction = tables.atom.StringCol(4)
    type = tables.atom.StringCol(5)
    nchannels = tables.atom.UInt8Col() # necessary
    is_tetrode = tables.atom.StringCol(4)
    fs = tables.atom.Float64Col() #
    id = tables.atom.StringCol(40) # git hash string length


def rotate1(s):
    return '%s%s' % (s[1:], s[:1])


def rotate(word, n=1):
    s = word
    for _ in range(n):
        s = rotate1(s)
    return s


def get_map():
    while True:
        try:
            ch_map = np.asarray(
                [x - 1 for x in
                 (int(y) for y in input('Enter a channel map: ').split())]
            )
        except ValueError:
            print('Invalid channel map, try again')
            continue
        else:
            if (ch_map < 0).any():
                print('You gave a negative-valued channel, try again')
                continue
            else:
                break
    return ch_map


types = (lambda s: str(np.datetime64(rotate(s, 4))).split(' ')[0],
         str, str, str, str, str, str, str, str, float)
date = r'(\d{6})'
age = r'([pP]\d{1,2})'
animal = r'(rat|mouse)'
site = r'(\d{1,2})?' # optional
season_litter = r'([sSfFwW][lL]\d{1,2})?' # optional
depth = r'(\d{3,4})?' # optional
direction = r'([hHvV])?' # optional
type = r'(spikes|Spik|lfp|LFP)'
is_tetrode = r'(True|False|[yY](?:[eE][sS])?|[nN][oO]?|[tTfF01])'
fs = r'([-+]?\d*\.?\d+([eE][-+]?\d+)?)'
dir_pattern = re.compile('_?'.join((date, age, animal, site, season_litter,
                                    depth, direction, type, is_tetrode, fs)))
keys = ('date', 'age', 'animal', 'site', 'season_litter', 'depth',
        'direction', 'type', 'is_tetrode', 'fs')



if __name__ == '__main__':
    # set a prompt
    prompt = 'Are you sure you want to recreate the entire database? [N/y] '

    h5file = tables.openFile('data.h5', mode='r+', title='ARC Spike Data')
    root = '/'
    group = h5file.createGroup(root, 'recordings', 'All Recordings')
    table = h5file.createTable(h5file.root.recordings, 'meta',
                           Recording, 'Recording Metadata')
    data_group = h5file.createGroup(h5file.root.recordings,
                                'data', 'Data')
    raw_group = h5file.createGroup(h5file.root.recordings.data,
                               'raw', 'Raw Data')
    row = table.row
    # make sure I actually want to do this (it takes a long time)
    if input(prompt).lower() == 'y':
        comp_prompt = 'Use default compression (2, zlib)? [Y/n] '
        use_default_compression = input(comp_prompt).lower()
        complevel = 2
        if use_default_compression == 'n':
            while complevel not in list(range(1, 10)):
                complevel = int(eval(input('Compression level [0-9]? ')))
        elif use_default_compression in 'y\n':
            complevel = 2
        else:
            raise ValueError('Invalid input to compression prompt')

        nshanks_string = input('How many shanks? [4] ').strip()
        nshanks = int(nshanks_string
                      if re.match(r'^\d+$', nshanks_string) is not None else 4)

        channel_map = get_map()

        nchannels = len(channel_map)
        nchannels_per_shank = nchannels / nshanks
        directories = [d for d in os.listdir(os.getcwd()) if os.path.isdir(d)]
        # directories = filter(lambda x: x.startswith('100511'), dirs)
        # directories = ['100511_p7_rat_2_spikes_no_24414.1']

        # loop over the directories
        for i, directory in enumerate(directories):
            # while map_changed not in 'yn':
            #     map_changed = raw_input(
            #         'Map changed for experiment: %s? [N/y] ' % directory
            #     ).lower()

            # did_map_change = map_changed == 'y'

            # if did_map_change:
                # channel_map = get_map()

            fname = directory.split(os.sep)[-1]
            print(('starting build of database with: %s' % fname))

            # get the experiment metadata from the folder name
            metadata = dir_pattern.match(directory).groups()

            # open the data into a single array
            npy_files = glob('{d}{sep}{npy}{sep}*.{npy}'.format(d=directory,
                                                                sep=os.sep,
                                                                npy='npy'))

            if nchannels != len(npy_files):
                raise ValueError(
                    'channel map has more channels than the number of files'
                )
            sort_func = lambda x: int(re.match(r'.+Ch(\d{1,2}).+', x).groups()[0])
            npy_files.sort(key=sort_func)

            # set the number of channels
            row['nchannels'] = nchannels

            # name of datasets: somewhat arbitrary
            data_name = 'd%i' % (i + 1)

            # make the compressed array
            compressor = tables.Filters(complevel=complevel, complib='zlib')
            h5file.createGroup(h5file.root.recordings.data.raw, data_name)

            k = 0
            for shanki in range(nshanks):
                # make a shank name
                shank_name = 'sh%i' % (shanki + 1)

                # get the data group
                data_group = getattr(h5file.root.recordings.data.raw, data_name)

                # create a group for the current shank
                h5file.createGroup(data_group, shank_name)

                # for each channel number
                for channeli in range(nchannels_per_shank):
                    # get the shank group
                    shank_group = getattr(data_group, shank_name)

                    # for this shank create an electrode array
                    d = np.load(npy_files[channel_map[k]])
                    a = h5file.createCArray(shank_group,
                                            'ch%i' % (channel_map[k] + 1),
                                            tables.atom.Float64Atom(),
                                            d.shape,
                                            'Raw Data',
                                            filters=compressor)

                    # assign the data to the compressed array
                    a[:] = d[:]

                    # make sure the data are written before decrefing the a and d
                    # arrays
                    h5file.flush()

                    # decrease the refcount of d and a
                    # del d, a

                    # increment the channel number counter
                    k += 1

            # for each typefunc, key, and metadatum in their respective arrays
            for typei, key, metadatum in zip(types, keys, metadata):

                # convert the datum using the appropriate function
                metadatum_hat = typei(metadatum)

                # print out the metadatum for visual inspection
                print(('{0}: {1}'.format(key, metadatum_hat)))

                # put it in the current table row
                row[key] = metadatum_hat

            # append the row to the data set
            row.append()

        # write the file
        h5file.flush()

        # message indicating completion
        print(('wrote %s to database' % fname))

    h5file.close()
