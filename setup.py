#! /usr/bin/env python

# setup.py ---

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


from future_builtins import zip, map

import os
import glob

try:
    import numpy as np
except ImportError:
    raise ImportError('NumPy is not installed on your system')

from distutils.core import setup

from Cython.Distutils import build_ext
from Cython.Distutils.extension import Extension

extra_compile_args = ['-fopenmp']
extra_link_args = ['-fopenmp']

npy_includes = np.get_include()
include_dirs = [npy_includes]

# define_macros = [('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')]
define_macros = []

# module file names sans extension
names = 'clear_refrac', 'mult_mat_xcorr', 'read_tev'

# module packages
mod_pkgs = 'span.utils', 'span.xcorr', 'span.tdt', 'span.stats'

# prefix for *.so files
underscore = '_'

# name of python module as if one was going to import it
und_join = lambda x, y: os.extsep.join((x, underscore + y))
base_names = tuple(map(und_join, mod_pkgs, names))

# directory of the modules/files
dirs = map(lambda x: x.replace(os.extsep, os.sep), mod_pkgs)

ext_modules = []


for d, base_name in zip(dirs, base_names):
    pyx_file_base = base_name.split(os.extsep)[-1].lstrip(underscore)
    pyx_file = os.path.join(d, pyx_file_base + os.extsep + 'pyx')

    ext_modules.append(Extension(base_name, [pyx_file],
                                 define_macros=define_macros,
                                 extra_compile_args=extra_compile_args,
                                 extra_link_args=extra_link_args,
                                 include_dirs=include_dirs))


if __name__ == '__main__':
    readme_filename = glob.glob('README*')

    if readme_filename:
        readme_filename = readme_filename[0]
    else:
        readme_filename = ''

    if readme_filename:
        with open(readme_filename, 'r') as f:
            readme = f.read()
    else:
        readme = ''

    setup(name='span',
          version='0.1',
          author='Phillip Cloud',
          author_email='cpcloud@gmail.com',
          packages=['span', 'span.tdt', 'span.utils', 'span.xcorr',
                    'span.stats'],
          scripts=[os.path.join('bin', 'serv2mat.py')],
          ext_modules=ext_modules,
          license='LICENSE.txt',
          description='Spike train analysis',
          long_description=readme,
          cmdclass={'build_ext': build_ext})
