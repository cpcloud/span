#!/usr/bin/env python

# spikeglobals.py ---

# Copyright (C) 2012 Copyright (C) 2012 Phillip Cloud <cpcloud@gmail.com>

# Author: Phillip Cloud <cpcloud@gmail.com>

# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.


import numpy as np
import pandas as pd


NeuroNexusMap = pd.DataFrame(np.array([[1,  3,  2,  6],
                                       [7,  4,  5,  8],
                                       [13, 12, 10,  9],
                                       [14, 16, 11, 15]]).T)


TdtEventTypes = pd.Series({
    0x0: 'unknown',
    0x101: 'strobe_on',
    0x102: 'strobe_off',
    0x201: 'scaler',
    0x8101: 'stream',
    0x8201: 'snip',
    0x8801: 'mark',
    0x8000: 'hasdata'
}, name='TDT Event Types')


TdtDataTypes = pd.Series({
    0: 'float32',
    1: 'int32',
    2: 'int16',
    3: 'int8',
    4: 'float64',
}, name='TDT Data Types')
