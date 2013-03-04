# __init__.py ---

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


from span.utils.utils import (name2num, ndtuples, iscomplex,
                              get_fft_funcs, isvector,
                              assert_nonzero_existing_file,
                              clear_refrac, ispower2, fromtimestamp,
                              create_repeating_multi_index, _diag_inds_n)
from span.utils.ordereddict import OrderedDict
from span.utils.math import (detrend_none, detrend_mean, detrend_linear,
                             cartesian, nextpow2, samples_per_ms, compose,
                             compose2, composemap)
from span.utils.decorate import thunkify, cached_property

__all__ = ('name2num', 'ndtuples', 'iscomplex', 'get_fft_funcs', 'isvector',
           'assert_nonzero_existing_file', 'clear_refrac', 'ispower2',
           'thunkify', 'ispower2', 'detrend_none', 'detrend_mean',
           'detrend_linear', 'cartesian', 'nextpow2', 'samples_per_ms',
           'compose', 'compose2', 'composemap', 'num2name',
           'create_repeating_multi_index', 'OrderedDict', '_diag_inds_n')
