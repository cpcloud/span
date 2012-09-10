#!/usr/bin/env python

# 1. get the directory
# 2. get the map
# 3. get the

import os
import re
import glob
import hashlib
import collections

import numpy as np
import tables

# be careful here: these are globals from make_database.py
from make_database import get_map, dir_pattern, types, keys
from history_completer import setup_readline, input_loop


def hashable(obj):
    """Check to see if `obj` is hashable."""
    return isinstance(obj, collections.Hashable)


def githash(data):
    """Return a sha1 hash of `data`."""
    assert hashable(data), \
        'input type: {0} is not a hashable type'.format(type(data))
    sha = hashlib.sha1()
    sha.update("%u\0" % len(data))
    sha.update(data)
    return sha.hexdigest()


def isduplicate(data, ids):
    """Check to see if the hash of a string is equal to a given key."""
    return githash(data) in ids


def choose_directories(starting_directory=os.path.expanduser('~/analysis')):
    """Choose one or more directories from a dialog box."""
    import gtk
    buttons = (gtk.STOCK_CANCEL, gtk.RESPONSE_CANCEL,
               gtk.STOCK_OPEN, gtk.RESPONSE_OK)
    chooser = gtk.FileChooserDialog(
        title='Choose a directory containing NPY files',
                                    parent=None,
                                    action=gtk.FILE_CHOOSER_ACTION_OPEN,
                                    buttons=buttons)
    chooser.set_current_folder(starting_directory)
    chooser.set_default_response(gtk.RESPONSE_CANCEL)
    chooser.set_select_multiple(True)
    chooser.set_action(gtk.FILE_CHOOSER_ACTION_SELECT_FOLDER)
    response = chooser.run()

    direcs = []
    if response == gtk.RESPONSE_OK:
        direcs += chooser.get_filenames()

    chooser.destroy()
    return direcs


def check_directories(direcs=None):
    """Check to make sure directories exist in the current folder and
    have the subdirectory 'npy' containing npy files.
    """
    return all(os.path.exists(subdir)
               for subdir in (os.path.join(d, 'npy') for d in direcs))


def open_data_group(fn, mode='r+'):
    """Open an HDF5 file and return the raw data group and the file."""
    assert os.path.exists(fn), '%s does not exist' % fn
    assert mode in ['r', 'r+', 'a'], 'invalid mode for opening file'
    h5file = tables.openFile(fn, mode=mode)
    return h5file.root.recordings.data.raw, h5file


def channel_sort(string, pattern=re.compile(r'.+[cC]h(\d{1,3}).+')):
    """Sort channel files' names by the channel number."""
    if isinstance(pattern, str):
        pattern = re.compile(pattern)

    return int(pattern.search(string).group(1))


def check_map(direc, prompt='Map changed for experiment: %s? [N/y] '):
    """Check to see if the channel map has changed."""
    prompt %= direc
    map_changed = input(prompt).lower()
    if not map_changed:
        return None

    while map_changed not in 'yn':
        map_changed = input(prompt).lower()
        if not map_changed:
            return None

    chan_map = get_map() if map_changed == 'y' else None
    return chan_map


def get_nshanks(prompt='Number of shanks? [4] '):
    """Get the number of shanks from the user."""
    return input_loop(prompt=prompt, converter=int,
                      default=4)


def get_compressor(prompt='Compression level? [2] '):
    """Get the level of compression from the user."""
    return tables.Filters(complevel=input_loop(prompt=prompt,
                                               converter=int,
                                               default=2), complib='zlib')


def get_ids(f):
    """Get the hashes of all the datasets in the database."""
    assert hasattr(f, 'isopen'), 'f must have an "isopen" attribute'
    assert hasattr(f, 'mode'), 'f must have a "mode" attribute'
    assert f.isopen, 'f must be open'
    assert f.mode in ['r', 'r+'], 'f must be readable'
    return [row['id'] for row in f.root.recordings.meta]


def get_last_dataset_number(h5file, ds='d', pattern=None):
    """Get the largest dataset number in the database."""
    assert h5file.isopen, 'h5file must be open'
    assert h5file.mode in ['r', 'r+'], 'h5file must be in a reading mode'

    if pattern is None:
        pattern = r'^{ds}(\d+)'.format(ds=ds)
    elif isinstance(pattern, str):
        pattern = re.compile(pattern)

    dataset_names = [x for x in
                     dir(h5file.root.recordings.data.raw) if x.startswith(ds)]
    sort_func = lambda string: int(pattern.match(string).group(1))
    dataset_names.sort(key=sort_func)
    return sort_func(dataset_names[-1])


def main(filename='data.h5'):
    """ """
    import itertools
    FLOAT64_ATOM = tables.atom.Float64Atom()
    directories = choose_directories()

    if check_directories(directories):
        # open the dataset and file
        raw, f = open_data_group(filename)

        # ask for user input channel map
        channel_map = get_map()
        nchannels = len(channel_map)
        nshanks = get_nshanks()
        nchannels_per_shank = nchannels / nshanks

        # ask for compression level
        compressor = get_compressor()

        # for each directory
        for i, directory in enumerate(directories):
            # get all of the numpy files
            npy_files = glob.glob('{0}{1}{2}{1}*.{2}'.format(directory, os.sep,
                                                             'npy'))

            # check to make sure the number of channels == num npy_files
            if nchannels != len(npy_files):
                f.close()
                raise ValueError('channel map channels != number of txt files')

            # sort file names numerically by channel
            npy_files.sort(key=channel_sort)

            # if more than one directory ask if the map has changed
            if i > 0:
                channel_map = check_map(directory)

            # get the base directory
            directory = directory.rstrip(os.sep)

            # get the right most filename
            fname = os.path.basename(directory)

            # if the data are already in the database go to the
            # next dataset
            if isduplicate(fname, get_ids(f)):
                continue

            # get the experiment metadata from the folder name
            metadata = dir_pattern.match(fname).groups()

            # name of a dataset
            data_name = 'd%i' % (get_last_dataset_number(f) + 1)

            # make the dataset group
            f.createGroup(raw, data_name)

            # notify the user what directory we're writing to the database
            print(('adding %s to database...' % fname))
            k = 0
            for shanki in range(nshanks):
                # make a shank name
                shank_name = 'sh%i' % (shanki + 1)

                # get the data group
                data_group = getattr(raw, data_name)

                # create a group for the current shank
                f.createGroup(data_group, shank_name)

                # for each channel number
                for _ in range(nchannels_per_shank):
                    # channel number
                    chn = channel_map[k]
                    # get the shank group
                    shank_group = getattr(data_group, shank_name)

                    # for this shank create an electrode array
                    d = np.asarray(np.load(npy_files[chn]))
                    a = f.createCArray(shank_group, 'ch%i' % (chn + 1),
                                       FLOAT64_ATOM, d.shape, 'Raw Data',
                                       filters=compressor)

                    # assign the data to the compressed array
                    a[:] = d[:]
                    f.flush()

                    # increment the channel number counter
                    k += 1

            row = f.root.recordings.meta.row

            # set the number of channels
            row['nchannels'] = nchannels

            # set the id
            row['id'] = githash(fname)

            # for each typefunc, key, and metadatum in their respective arrays
            for typei, key, metadatum in zip(types, keys, metadata):

                # convert the datum using the appropriate function
                mdhat = typei(metadatum)

                # print out the metadatum for visual inspection
                print(('{key}: {metadatum}'.format(key=key, metadatum=mdhat)))

                # put it in the current table row
                row[key] = mdhat

            # append the row to the data set
            row.append()

        # write the file
        f.flush()

        # message indicating completion
        print(('wrote %s to database' % fname))

    f.close()


if __name__ == '__main__':
    setup_readline()
    main()
