import numpy as np
import pandas as pd

import span.utils

TsqFields = ('size', 'type', 'name', 'channel', 'sort_code', 'timestamp',
             'file_pointer_location', 'format', 'fs')
TsqNumpyTypes = (np.int32, np.int32, np.uint32, np.uint16, np.uint16,
                 np.float64, np.int64, np.int32, np.float32)

ElectrodeMap = pd.Series(np.array([[1, 3, 2, 6],
                                   [7, 4, 5, 8],
                                   [13, 12, 10, 9],
                                   [14, 16, 11, 15]]).ravel() - 1, name='Electrode Map')

NShanks = 4
ElectrodesPerShank = 4
NSides = NShanks * 2
ShankMap = pd.Series(np.outer(xrange(NShanks),
                              np.ones(NShanks)).astype(int).ravel(),
                     name='Shank Map')
MedialLateral = pd.Series(np.asanyarray(('medial', 'lateral'))[np.hstack((np.zeros(NSides, int),
                                                                          np.ones(NSides, int)))],
                          name='Side Map')
Indexer = pd.DataFrame(dict(zip(('channel', 'shank', 'side'),
                                (ElectrodeMap, ShankMap, MedialLateral))))

EventTypes = pd.Series({
    0x0: np.nan,
    0x101: 'strobe_on',
    0x102: 'strobe_off',
    0x201: 'scaler',
    0x8101: 'stream',
    0x8201: 'snip',
    0x8801: 'mark'
}, name='Event Types')

DistanceMap = pd.Series(span.utils.distance_map(nshanks=NShanks,
                                                electrodes_per_shank=ElectrodesPerShank).values.ravel(),
                        name='Electrode Distance')
