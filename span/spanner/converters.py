import os
import collections
import tarfile
import subprocess

import numpy as np

CHAR_BIT = 8

try:
    from bottleneck import nanmax
except ImportError:
    from numpy import nanmax

from lxml.builder import ElementMaker
from lxml import etree

from span.utils import randcolors
from span.spanner import SpanCommand
from span.spanner import error


class BaseConverter(object):
    store_index = False

    def __init__(self, base_type, precision, date):
        self.base_type, self.precision = base_type, precision
        self.dtype = np.dtype(self.base_type + str(self.precision))
        self.date = date

    def split_data(self, raw):
        shank = raw.columns.get_level_values('shank').values
        channels = raw.columns.get_level_values('channel').values
        index = raw.index.values
        values = raw.values
        fs = raw.fs
        date = self.date
        elapsed = (raw.index.freq.n *
                   np.zeros(raw.nsamples)).cumsum().astype('timedelta64[ns]')
        return locals()

    def convert(self, raw, outfile):
        if not self.store_index:
            raw.sortlevel('channel', axis=1, inplace=True)

        self._convert(raw, outfile)


class NeuroscopeConverter(BaseConverter):

    def _convert(self, raw, outfile):
        max_prec = float(2 ** (self.precision * CHAR_BIT - 1) - 1)
        const = max_prec / nanmax(np.abs(raw.values))
        xc = raw.values * const
        xc.astype(self.dtype).tofile(outfile)


class H5Converter(BaseConverter):
    store_index = True
    store_fs = True

    def _convert(self, raw, outfile):
        raw.to_hdf(outfile, 'raw')


class NumPyConverter(BaseConverter):
    store_index = True
    store_fs = True

    def _convert(self, raw, outfile):
        split = self.split_data(raw)
        values = split['values']

        if self.dtype != values.dtype:
            split['values'] = values.astype(self.dtype)

        np.savez(outfile, **split)


class MATLABConverter(BaseConverter):
    store_index = True
    store_fs = True

    def _convert(self, raw, outfile):
        try:
            from scipy.io import savemat
        except ImportError:
            return error('scipy not installed please install it with pip '
                         'install scipy')
        savemat(outfile, self.split_data(raw))


_converters = {'neuroscope': NeuroscopeConverter, 'matlab': MATLABConverter,
               'h5': H5Converter, 'numpy': NumPyConverter}


class Converter(SpanCommand):

    def _run(self, args):
        spikes = self._load_data()
        converter = _converters[args.format](args.base_dtype, args.precision)
        converter.convert(spikes, args.outfile)


def _build_anatomical_description_element(index, E):
    anatomicalDescription = E.anatomicalDescription
    channelGroups = E.channelGroups
    group = E.group
    channel = E.channel
    groups = collections.defaultdict(list)

    for shank, channel in index:
        groups[shank].append(E.channel(str(channel)))

    items = groups.items()
    items.sort(key=lambda x: x[0])
    grouplist = []
    for gn, grp in items:
        grouplist.append(group(*grp))
    return anatomicalDescription(channelGroups(*grouplist))


def _build_spike_detection_element(index, E):
    spikeDetection = E.spikeDetection
    channelGroups = E.channelGroups
    group = E.group
    channel = E.channel
    groups = collections.defaultdict(list)

    for shank, channel in index:
        groups[shank].append(E.channel(str(channel), skip='0'))

    items = groups.items()
    items.sort(key=lambda x: x[0])
    grouplist = []
    for gn, grp in items:
        grouplist.append(group(*grp))
    return spikeDetection(channelGroups(*grouplist))


def _build_channels_element(index, E, colors):
    def _build_single_channel_color(channel, color):
        return E.channelColors(
            E.channel(channel),
            E.color(color),
            E.anatomyColor(color),
            E.spikeColor(color)
        )

    def _build_single_channel_offset(channel):
        return E.channelOffset(
            E.channel(channel),
            E.defaultOffset('0')
        )

    elements = []

    for shank, channel in index:
        c = str(channel)
        elements.append(_build_single_channel_color(c, colors[shank]))
        elements.append(_build_single_channel_offset(c))
    return E.channels(*elements)


def _make_neuroscope_xml(spikes, base, precision, voltage_range, amp, tarfile):
    E = ElementMaker()
    parameters = E.parameters
    acquisitionSystem = E.acquisitionSystem
    nBits = E.nBits
    nChannels = E.nChannels
    samplingRate = E.samplingRate
    voltageRange = E.voltageRange
    amplification = E.amplification
    offset = E.offset
    columns = spikes.columns
    colors = randcolors(columns.get_level_values('shank').unique().size)

    doc = parameters(
        acquisitionSystem(
            nBits(str(precision)),
            nChannels(str(spikes.nchannels)),
            samplingRate(str(int(spikes.fs))),
            voltageRange(str(voltage_range)),
            amplification(str(amp)),
            offset('0')
        ),

        E.fieldPotentials(
            E.lfpSamplingRate('1250')
        ),

        _build_anatomical_description_element(columns, E),
        _build_spike_detection_element(columns, E),

        E.neuroscope(
            E.miscellaneous(
                E.screenGain('0.2'),
                E.traceBackgroundImage()
            ),

            E.video(
                E.rotate('0'),
                E.flip('0'),
                E.videoImage(),
                E.positionsBackground('0')
            ),

            E.spikes(
                E.nsamples('72'),
                E.peakSampleIndex('36')
            ),

            _build_channels_element(columns, E, colors),
            version='1.3.3'
        ),
        creator='spanner.py',
        version='0.1'
    )

    filename = base + os.extsep + 'xml'

    with open(filename, 'w') as f:
        f.write(etree.tostring(doc, pretty_print=True))

    tarfile.add(filename)
    os.remove(filename)


def _make_neuroscope_nrs(spikes, base, start_time, window_size, tarfile):
    def _build_channel_positions():
        return (
            E.channelPosition(
                E.channel(
                    str(channel)
                ),
                E.gain('10'),
                E.offset('0')
            ) for channel in channels
        )
    E = ElementMaker()
    channels = spikes.columns.get_level_values('channel')

    doc = E.neuroscope(
        E.files(),
        E.displays(
            E.display(
                E.tabLabel('Field Potentials Display'),
                E.showLabels('0'),
                E.startTime(str(start_time)),
                E.duration(str(window_size)),
                E.multipleColumns('0'),
                E.greyScale('0'),
                E.positionView('0'),
                E.showEvents('0'),
                E.spikePresentation('0'),
                E.rasterHeight('33'),
                E.channelPositions(
                    *_build_channel_positions()
                ),
                E.channelsSelected(),
                E.channelsShown(
                    *(E.channel(str(channel)) for channel in channels)
                )
            )
        )
    )

    filename = base + os.extsep + 'nrs'

    with open(filename, 'w') as f:
        f.write(etree.tostring(doc, pretty_print=True))

    tarfile.add(filename)
    os.remove(filename)


def _build_neuroscope_package(spikes, converter, base, outfile, zipped_name,
                              args):
    tarfile_name = base + os.extsep + 'tar{0}{1}'.format(os.extsep,
                                                         args.format)
    with tarfile.open(tarfile_name, 'w:{0}'.format(args.format)) as f:
        converter.convert(spikes, outfile)
        f.add(outfile)
        os.remove(outfile)
        _make_neuroscope_xml(spikes, base, args.precision, args.voltage_range,
                             args.amplification, f)
        _make_neuroscope_nrs(spikes, base, args.start_time, args.window_size,
                             f)


def _get_dat_from_tarfile(tarfile):
    members = tarfile.getmembers()
    names = [member.name for member in members]
    for name, member in zip(names, members):
        if name.endswith('.dat'):
            return member
    else:
        return error('no DAT file found in neuroscope package. files found '
                     'were {1}'.format(names))


