#!/usr/bin/env python

from __future__ import division

import os
import struct
import itertools
import subprocess
import multiprocessing
import argparse
import re
import numbers
import cPickle as pickle

import tables
import numpy as np

import spike_sort

from thread_read import load_data


def compute_threshold(sp, fs, contact, sc=5.0, const=0.6745, indfact=10):
    """Compute a threshold using parameters from `scholarpedia.org`.

    Parameters
    ----------
    sp : dict_like
        Dictionary of spike data
    fs : int or long
        Sampling frequency
    contact : int or long
        The reference contact to use
    scale : float, optional
        How much to scale the threshold by
    constant : float, optional
        Dividing constant for threshold
    index_factor : int, optional
        How much of the data to take.

    Returns
    -------
    threshed_data : array
    """
    samps = sp['data'][contact, :indfact * fs]
    return sc * np.median(np.abs(samps)) / const

x = np.zeros((10, 10))

def detect_align_extract(sp, win, fs, contact=0, thresh=None, align=False,
                         verbose=False):
    """Detect, align, and extract spikes.

    Parameters
    ----------
    sp : dict_like
    win : array_like
    fs : int or long
    contact : int or long, optional
    thresh : None or float, optional
    align : bool, optional
    verbose : bool, optional

    Returns
    -------
    sp_waves, spt, unaligned : tuple of dict_like
    """
    if thresh is None:
        if verbose:
            print 'computing threshold...'
        thresh = compute_threshold(sp, fs, contact)

    if verbose:
        print 'detecting spikes...'
    unaligned = spike_sort.extract.detect_spikes(sp,
                                                 contact=contact,
                                                 thresh=thresh)
    spt = unaligned

    if align:
        if verbose:
            print 'aligning spikes...'
        aligned = spike_sort.extract.align_spikes(sp, unaligned, win)
        spt = aligned

    if verbose:
        print 'extracting spikes...'
    sp = spike_sort.extract.extract_spikes(sp, spt, win)
    return sp, spt


def compute_features(sp_waves, spt, sc=None, norm=False, dtype=np.int32):
    """Compute the features of a dataset.

    Parameters
    ----------
    sp_waves : dict_like
        A dictionary-like object with spike data
    spt : dict_like
        Dictionary-like object of spike time data
    sc : int or long, optional
        A scale factor for klusters
    norm : bool, optional
        Whether to normalize the data
    dtype : dtype, optional
        Convert the output data to this type, for Klusters

    Returns
    -------
    feature_data, spike_times : tuple of dict_like, array_like
        A dictionary of features and an array of spike times
    """
    p2p = spike_sort.features.fetP2P(sp_waves)
    pcs = spike_sort.features.fetPCs(sp_waves)
    features = spike_sort.features.combine((p2p, pcs), norm=norm)
    fd = features['data']
    if sc is None:
        sc = (2 ** 31 - 1) / np.abs(fd).max()
    d = spt['data'][:, np.newaxis]
    fs = sp_waves['FS']
    features = fd * sc
    times = d * fs / 1000
    try:
        a = features.astype(dtype, copy=False)
        b = times.astype(dtype, copy=False)
    except TypeError:
        a, b = features.astype(dtype), times.astype(dtype)
    feature_data = np.hstack((a, b))
    return feature_data, b


def write_features(features, prefix, ext, suffix='fet'):
    """Write the feature data set to a text file.

    Parameters
    ----------
    features : array_like
        The data to be written
    prefix : str
    ext : int
    suffix : str, optional
    """
    nfeatures = min(features.shape)
    filename = '{prefix}.{suffix}.{ext}'.format(prefix=prefix,
                                                suffix=suffix,
                                                ext=ext)
    with open(filename, 'w') as f:
        f.write('%i\n' % nfeatures)
        np.savetxt(f, features, fmt='%i')


def run_klustakwik(prefix, ext, suffix='fet', verbose=False, log=0):
    """Check that KlustaKwik exists and run it if it does, else raise
    an exception telling the user that it doesn't exist.

    Parameters
    ----------
    prefix : str
    ext : int
    suffix : str, optional
    verbose : int, optional
    log : int, optional

    Returns
    -------
    retcode : int
    """
    filename = '{prefix}.{suffix}.{ext}'.format(prefix=prefix,
                                                suffix=suffix,
                                                ext=ext)
    assert os.path.exists(filename), '{0} must exist'.format(filename)

    cmd_name = 'KlustaKwik'
    cmd = which(cmd_name)
    assert cmd, '%s does not exist' % cmd_name
    args = (cmd, prefix, str(ext), '-Screen', str(int(verbose)),
            '-Log', str(log))
    print 'Running %s ...' % cmd_name
    retcode = subprocess.check_call(args)
    print '... finished running %s' % cmd_name
    return retcode


def isexe(fpath):
    return os.path.isfile(fpath) and os.access(fpath, os.X_OK)


def which(prog):
    """Python version of the *NIX command-line program `which`.
    Shamelessly jacked from Stack Overflow.

    Parameters
    ----------
    prog : str
        The program whose existence will be checked.

    Returns
    -------
    ret : str_like or None
        The program name as a string or None.
    """
    fpath, _ = os.path.split(prog)
    if fpath and isexe(prog):
        return prog
    else:
        for path in os.environ['PATH'].split(os.pathsep):
            exefile = os.path.join(path, prog)
            if isexe(exefile):
                return exefile


def whichall(progs):
    for prog in progs:
        yield which(prog)


def klustakwik_results(prefix, ext, suffix='clu', dtype=np.int32):
    """Output the results of KlustaKwik.

    Parameters
    ----------
    prefix : str, optional
    suffix : str, optional
    dtype : dtype, optional

    Returns
    -------
    x : array_like
        The results of `KlustaKwik`
    """
    res = np.loadtxt('{prefix}.{suffix}.{ext}'.format(prefix=prefix,
                                                      suffix=suffix,
                                                      ext=ext))
    if res.dtype != dtype:
        try:
            res = res.astype(dtype, copy=False)
        except TypeError:
            res = res.astype(dtype)
    nclusters = res[0]
    res = res[1:]
    return nclusters, res


def write_res(spike_times, prefix, ext, verbose=True):
    """Write the data and the spike time array in an appropriate format for
    neuroscope.

    Parameters
    ----------
    spike_times : array_like
    prefix : str
    ext : int
    verbose : bool, optional

    Returns
    -------
    data : array_like
    """
    res_filename = '{prefix}.{suffix}.{ext}'.format(prefix=prefix,
                                                    suffix='res',
                                                    ext=ext)
    if verbose:
        print 'writing RES file...'

    np.savetxt(res_filename, spike_times, fmt='%i')

    if verbose:
        print '...wrote RES file'


def load_dataset(f, shank_number, where=None, raw_name='raw', fs=None):
    """Read in a dataset using filters from SpikeSort.

    Parameters
    ----------
    f : tables.file.File
    shank_number : int
    where : optional
    raw_name : str, optional

    Returns
    -------
    r : dict_like

    See Also
    --------
    `tables.Array`
    """
    assert f.isopen, 'file must be open'
    assert f.mode != 'w', 'file must be readable'
    shank_attr = 'sh%i' % shank_number
    assert hasattr(where, shank_attr), 'no shank: %s' % shank_attr
    shi = getattr(where, shank_attr)
    assert hasattr(shi, raw_name), ('no sh%i attribute named %s' %
                                    (shank_number, raw_name))
    raw = getattr(shi, raw_name)
    if fs is None:
        assert hasattr(raw.attrs, 'sampfreq'), 'no sampfreq attribute'
        fs = raw.attrs.sampfreq
    else:
        assert isinstance(fs, numbers.Number)
    return {'data': raw, 'FS': fs, 'n_contacts': min(raw.shape)}


def get_electrode_map(f, where=None):
    """"""
    assert f.isopen, 'HDF5 file is not open'
    assert f.mode != 'w', 'file must be readable'
    if where is None:
        where = f.root.data
    shanks = f.iterNodes(where)
    return np.vstack(shank.raw.attrs.map.flatten() for shank in shanks)


def write_spk(data, prefix, ext, sc, dtype=np.int16, suffix='spk',
              perm=(1, 0, 2)):
    """Write a data set `data` to a file with file name `filename`.

    This function writes the SPK file for use with Klusters.

    Parameters
    ----------
    data : array_like
    prefix : str
    ext : int or string
    sc : float
    dtype : numpy.dtype, optional
    suffix : str, optional
    perm : array_like, optional
    """
    fn = '{prefix}.{suffix}.{ext}'.format(prefix=prefix,
                                          suffix=suffix, ext=ext)
    print 'writing SPK file...'
    d = data * sc
    try:
        c = d.astype(dtype, copy=False)
    except TypeError:
        c = d.astype(dtype)
    c.transpose(perm).tofile(fn)
    print '...wrote SPK file'


def write_par_file_impl(prefix, fs, elec_map, nchans=16, numbits=16):
    """Write a generic PAR file for Klusters.

    Parameters
    ----------
    prefix : str
    fs : int
    elec_map : array_like
    nchans : int, optional
    nbits : int, optional
    """
    one = nchans, numbits
    two = int(np.floor(1e6 / fs)), 0
    three = len(elec_map),
    first3 = '\n'.join(' '.join(str(el) for el in r)
                       for r in (one, two, three))
    elec_map_str = '\n'.join(' '.join(str(el) for el in r) for r in elec_map)
    fn = '{prefix}.{ext}'.format(prefix=prefix, ext='par')
    with open(fn, 'w') as f:
        f.write('\n'.join((first3, elec_map_str)))


def write_parn_file(filename, nchannels, elec_ids, grp_num, fs,
                    samps_per_wave, npcs=2, fr=90):
    """Write a shank-specific par.n file where n is the shank number

    filename : str
    nchannels : int
    elec_ids : array_like
    grp_num : int
    fs : {int, long, float}
    samps_per_wave : int
    npcs : int
    """
    one = nchannels, len(elec_ids), int(np.floor(1.0 / fs * 1e6))
    two = tuple(elec_ids) if not isinstance(elec_ids, tuple) else elec_ids
    three = 0, 0
    four = fr,
    five = samps_per_wave, int(np.floor(samps_per_wave / 2))
    six = 0, 0
    seven = 0, 0
    eight = npcs, samps_per_wave
    nine = 0,
    rows = one, two, three, four, five, six, seven, eight, nine
    fn = '{filename}.{ext}.{grp_num}'.format(filename=filename,
                                             ext='par',
                                             grp_num=grp_num)
    with open(fn, 'w') as f:
        f.write('\n'.join(' '.join(str(el) for el in row) for row in rows))


def write_par_file(prefix, elec_map, fs, numbits):
    write_par_file_impl(prefix, fs, elec_map, nchans=elec_map.size,
                        numbits=numbits)


def datasize(dtype):
    """"""
    assert isinstance(dtype, basestring), 'dtype must be a string'
    return struct.calcsize('1%s' % dtype)


def nbits(dtype):
    """"""
    return 8 * datasize(dtype)


def truncate_and_scale(data, sc, dtype=np.int16):
    """"""
    print 'truncating...'
    d = data * sc
    try:
        c = d.astype(dtype, copy=False)
    except TypeError:
        c = d.astype(dtype)
    print '...truncated'
    return c.T.ravel()


def write_neuroscope(f, sc, data, filename):
    """"""
    if data is None:
        data = load_data(f)

    d = truncate_and_scale(data, sc)
    print 'writing dat file...'
    d.tofile(filename)
    print '...wrote dat file'


def get_sc(f, prefix, ext='dat'):
    """"""
    path = os.path.join(os.curdir,
                        '.{prefix}.absmax.pkl'.format(prefix=prefix))
    data = None
    if os.path.exists(path):
        with open(path, 'r') as fl:
            sc = pickle.load(fl)
    else:
        print 'computing constant...'
        data = load_data(f, where=f.root.data, raw_name='raw')
        sc = (2 ** 15 - 1) / np.abs(data).max()

        with open(path, 'w') as fl:
            pickle.dump(sc, fl)
        print '...wrote constant'

    dat_filename = '{prefix}.{ext}'.format(prefix=prefix, ext=ext)
    if not os.path.exists(os.path.join(os.curdir, dat_filename)):
        multiprocessing.Process(target=write_neuroscope,
                                args=(f, sc, data, dat_filename)).start()
    return sc


def main(filename, contact, window, thresh, shank, prefix, dtype='h'):
    if not os.path.exists(prefix):
        os.mkdir(prefix)
    os.chdir(prefix)

    try:
        f = tables.openFile(filename, mode='r')

        els = get_electrode_map(f)
        elsd = els.copy()

        if shank is not None:
            elsd = elsd[shank][np.newaxis]

        sc = get_sc(f, prefix)

        for i, elecs in enumerate(elsd, start=1):
            sp = load_dataset(f, i, where=f.root.data, )

            print 'shank: {0}, electrodes: {1}'.format(i, elecs)

            # detect and align the waveforms
            fs = sp['FS']
            sp_waves, spt  = detect_align_extract(sp, window, fs,
                                                  contact=contact,
                                                  thresh=thresh,
                                                  align=True,
                                                  verbose=True)

            # try to get the feature data, if none go to the next shank
            try:
                all_data, spike_times = compute_features(sp_waves, spt)
            except np.linalg.LinAlgError:
                continue

            print 'no. spikes in shank {i}: {s}'.format(i=i, s=spike_times.size)

            write_features(all_data, prefix, i)

            funcs = write_res, write_spk, run_klustakwik, write_parn_file
            arg_sets = ((spike_times, prefix, i),
                        (sp_waves['data'][:], prefix, i, sc),
                        (prefix, i),
                        (prefix, els.size, elecs, i, fs,
                         sp_waves['data'].shape[0], 2, 90))

            for func, arg_set in itertools.izip(funcs, arg_sets):
                proc = multiprocessing.Process(name=func.__name__,
                                               target=func, args=arg_set)
                proc.daemon = True
                proc.start()

        write_par_file(prefix, els, fs, nbits(dtype))
    except IOError:
        os.chdir(os.pardir)
    except BaseException:
        f.close()
        os.chdir(os.pardir)
    else:
        f.close()


def parse_window(window):
    """docstring for parse_window"""
    seps = r'[,-]'
    opens = r'[\(\[\{]'
    closes = r'[\)\[\}]'
    number = r'[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?'
    match = re.match("{opens}?\s*({number})\s*(?:{seps}|\s)\s*({number})"
                         "\s*{closes}?".format(opens=opens, number=number,
                                               seps=seps, closes=closes,
                                               ws=r'\s'), window)
    assert match is not None, 'no match found'
    w = tuple(float(v) for v in match.groups())
    assert np.diff(w) > 0, 'window must be a positive interval'
    return w


def parse_args():
    """Function that builds an argument parser"""

    names = 'filename', 'contact', 'window', 'thresh', 'shank', 'prefix'
    short_names = ('-%s' % name[0] for name in names)
    types = str, int, parse_window, float, tuple, str
    defaults = None, 0, (-0.2, 0.8), 0, None, (0,), 'data'
    help_strings = ('an HDF5 file name',
                    'the reference electrode for alignment',
                    'the window used to find spikes',
                    'Spike detection threshold',
                    'Shank or shanks to analyze',
                    'Folder name for output files')
    parser = argparse.ArgumentParser(
        description="Perform a full analysis on a dataset for use in "
                    "Klusters/Neuroscope"
    )

    for n, t, d, s, h in itertools.izip(names, types, defaults, short_names,
                                        help_strings):
        parser.add_argument(s, '--%s' % n, type=t, dest=n, metavar=n.upper(),
                            nargs='?', default=d, help=h)
    return parser.parse_args()


if __name__ == '__main__':
    p = parse_args()
    main(p.filename, p.contact, p.window, p.thresh, p.shank, p.prefix)
