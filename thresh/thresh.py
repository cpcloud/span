#!/usr/bin/env python

from __future__ import division

import os
import glob
import itertools

import numpy as np
import pylab as pl
from pandas import HDFStore

import theano
import theano.tensor as T

from ..clear_refrac import thresh_and_clear
from ..xcorr import xcorr


def bin_data(data, bins):
    return np.hstack(np.histogram(data[:, j], bins)[0][1:, np.newaxis]
                     for j in xrange(min(data.shape))).T


def median(a, axis=None):
    if axis is None:
        sorted_ = T.sort(T.flatten(a))
    else:
        sorted_ = T.sort(a, axis=axis)
    if axis is None:
        axis = 0

    ashape = T.shape(a)
    ashape_axis = ashape[axis]
    index, isodd_div = divmod(ashape_axis, 2)
    if isodd_div == 1:
        sorted_ind = sorted_[index:index + 1]
    else:
        sorted_ind = sorted_[index - 1:index + 1]

    return T.mean(sorted_ind, axis=axis)


def make_threshold(data, threshes=None, sc=5.0, const=0.6745):
    """
    """
    minshape = np.min(data.shape)
    if threshes is None:
        scale_factor, constant = T.dscalars('scale_factor', 'constant')
        dataset = T.dmatrix('dataset')
        median_func = median(T.abs_(dataset) / constant, axis=0)
        f = scale_factor * median_func
        thresh_func = theano.function([scale_factor, dataset, constant], f,
                                      mode='FAST_RUN')
        threshes = thresh_func(sc, data, const)
    elif np.isscalar(threshes):
        threshes = np.repeat(threshes, minshape)
        try:
            threshes = threshes.astype(data.dtype, copy=False)
        except TypeError:
            threshes = threshes.astype(data.dtype)
    elif pl.isvector(threshes):
        assert threshes.size == minshape
        if threshes.ndim > 1:
            threshes = threshes.squeeze()
    else:
        raise TypeError("invalid type of threshold, must be ndarray, "
                        "scalar, or None")
    return threshes


def ndtuples(*dims):
    """
    """
    dims = list(dims)
    n = dims.pop()
    cur = np.arange(n)[:, np.newaxis]
    while dims:
        d = dims.pop()
        cur = np.kron(np.ones((d, 1)), cur)
        front = np.arange(d).repeat(n)[:, np.newaxis]
        cur = np.hstack((front, cur))
        n *= d
    return cur


def ndlinspace(ranges, *nelems):
    """
    """
    x = ndtuples(*nelems) + 1.0
    lbounds = []
    ubounds = []
    b = np.asarray(nelems, dtype=np.float64)
    lbounds, ubounds = itertools.izip(*((r[0], r[1]) for r in ranges))
    # for r in ranges:
        # lbounds.append(r[0])
        # ubounds.append(r[1])

    lbounds = np.asarray(lbounds, dtype=np.float64)
    ubounds = np.asarray(ubounds, dtype=np.float64)
    return lbounds + (x - 1) / (b - 1) * (ubounds - lbounds)


def spike_times(cleared, fs, const=1e3):
    """
    """
    c = const / fs
    times = np.zeros(cleared.shape, dtype=np.uint32)
    for i in xrange(np.min(times.shape)):
        w, = np.where(cleared[:, i])
        times[w, i] = w * c
    return times


def spike_window(ms, fs, const=1e3):
    """Perform a transparent conversion from time to samples.

    Parameters
    ----------
    ms : int
    fs : number
    const : number

    Returns
    -------
    Conversion of milleseconds to samples
    """
    return int(np.floor(ms / const * fs))


def create_spikes(data, cleared):
    """
    """
    try:
        clr = cleared.astype(bool, copy=False)
    except TypeError:
        clr = cleared.astype(bool)
    return np.ma.masked_array(data, np.logical_not(clr))


def show_spikes(ch, data, spikes, n=10, m=0, win=2.0, fs=None, mp=None):
    """
    """
    assert n - m > 0, 'range must be positive'
    if fs is None:
        msg = "data object doesn't have sampling rate"
        assert hasattr(data, 'attrs'), msg
        assert hasattr(data.attrs, 'sampfreq'), msg
        fs = data.attrs.sampfreq

    if mp is None:
        msg = "data object doesn't have channel map"
        assert hasattr(data, 'attrs'), msg
        assert hasattr(data.attrs, 'map'), msg
        mp = data.attrs.map

    window = spike_window(win, fs)
    spike_indices = spikes[ch].dropna().keys()
    assert n < max(spike_indices.shape)
    spike_indices = spike_indices[m:n]
    chind, = np.where(ch == mp)
    data_chind = data[chind[0]]
    for i, spike_index in enumerate(spike_indices, start=1):
        r = np.arange(spike_index - window, spike_index + window + 1)
        t = r * 1e3 / fs
        spike = data_chind[r]
        tm = spike_index * 1e3 / fs
        val = data_chind[spike_index]
        pl.plot(t, spike)
        ind = tm
        pl.axvline(ind, linewidth=1, color='r')
        pl.axhline(val, linewidth=1, color='r')
        pl.text(tm, val, '%.3g' % val, fontsize=12)
        try:
            while raw_input('Spike %d at %d' % (i, t[0])):
                pass
        except KeyboardInterrupt:
            break
        except EOFError:
            print
            break
        pl.clf()
    pl.close('all')


def make_scale(x, y):
    m = x.shape[0]
    z0 = 10000 * x + y
    z1 = np.zeros(m)
    for i in xrange(m):
        z1[i] = (z0 == z0[i]).sum()
    return z1 * 2


def scatter_bins(times, binsize, elec_map, window_title):
    plot_func = lambda x, y: pl.scatter(x, y, s=make_scale(x, y),
                                        edgecolor='b', facecolor='b')
    plot_binned_data(times, binsize, elec_map, window_title, plot_func)


def xcorr_bins(times, binsize, elec_map, window_title, maxlags=None,
               scale_type=None):
    def plot_func(x, y):
        c, l = xcorr(x, y, maxlags=maxlags, scale_type=scale_type)
        pl.plot(l, c)
    plot_binned_data(times, binsize, elec_map, window_title, plot_func)


def plot_binned_data(times, binsize, elec_map, window_title, plot_func):
    nplots = min(times.shape)
    npm1 = nplots - 1
    maxtime = times.max()
    binned = bin_data(times, np.r_[times.min():maxtime:binsize])
    ch_fmt_str = 'Ch. %i'
    title_fmt_str = '{0} vs. {1}, {2} ms bins'
    for i in xrange(nplots):
        emi = elec_map[i]
        xlab = ch_fmt_str % emi
        bini = binned[:, i]
        for j in xrange(i):
            emj = elec_map[j]
            pl.subplot(npm1, npm1, npm1 * (i - 1) + j + 1)
            binj = binned[:, j]
            plot_func(bini, binj)
            pl.title(title_fmt_str.format(emj, emi, binsize))
            pl.xlabel(xlab)
            pl.ylabel(ch_fmt_str % emj)
            pl.axis('tight')
    pl.gcf().canvas.set_window_title(window_title)


def check_times_binsizes(times, binsizes):
    if not np.any(times):
        raise ValueError('times cannot be empty')

    if not np.any(binsizes):
        raise ValueError('binsizes cannot be empty')

    if np.isscalar(binsizes):
        binsizes = np.atleast_1d(binsizes)
    return times, binsizes


def make_timescale_scatters(times, binsizes, elec_map, window_title):
    times, binsizes = check_times_binsizes(times, binsizes)
    for binsize in binsizes:
        pl.figure()
        scatter_bins(times, binsize, elec_map, window_title)


def make_timescale_xcorrs(times, binsizes, elec_map, window_title):
    times, binsizes = check_times_binsizes(times, binsizes)
    for binsize in binsizes:
        pl.figure()
        xcorr_bins(times, binsize, elec_map, window_title)


def aggregate_plot(filenames, binsizes, ms=2.0, plot_type='scatter', **kwargs):
    if plot_type == 'scatter':
        assert not kwargs, \
            'no keyword arguments are valid for plot_type == "scatter"'
    if isinstance(filenames, basestring):
        filenames = filenames,

    binsizes = np.atleast_1d(binsizes)

    plot_types = {
        None: make_timescale_scatters,
        'scatter': make_timescale_scatters,
        'xcorr': lambda *args, **kwargs: make_timescale_xcorrs(*args, **kwargs)
    }

    for filename in filenames:
        data, fs, elec_map, threshes = load_data(filename)
        win = spike_window(ms, fs)
        threshes = make_threshold(data, threshes=threshes)
        clr = thresh_and_clear(data, threshes, win)
        times = spike_times(clr, fs)
        window_name = os.path.basename(os.path.dirname(filename))
        plot_types[plot_type](times, binsizes, elec_map, window_name, **kwargs)


def load_data(filename):
    shanks = HDFStore(filename, mode='r')
    sh1 = shanks['sh1']
    return sh1.values, shanks['fs'].values[0], sh1.keys(), 2e-5


def parse_args():
    import argparse
    del argparse


def main():
    home = os.path.expanduser('~')
    all_dirs = '*'
    path = os.path.join(home, 'analysis', 'data', 'interesting', all_dirs)
    datadirs = glob.glob(path)
    filenames = tuple(os.path.join(datadir, 'store.h5') for datadir in datadirs)
    aggregate_plot(filenames[0], 1000, plot_type='scatter')
