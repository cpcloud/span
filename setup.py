#! /usr/bin/env python

from future_builtins import zip

import os
import platform
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

if platform.system().lower() == 'linux':
    extra_compile_args.append('-march=native')

npy_includes = np.get_include()
include_dirs = [npy_includes]

# define_macros = [('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')]
define_macros = []

utils_dir = os.path.join('span', 'utils')
xcorr_dir = os.path.join('span', 'xcorr')
tdt_dir = os.path.join('span', 'tdt')


base_names = 'clear_refrac', 'bin_data', 'mult_mat_xcorr', 'read_tev'
dirs = utils_dir, utils_dir, xcorr_dir, tdt_dir

ext_modules = []

for d, base_name in zip(dirs, base_names):
    ext_modules.append(Extension('_%s' % base_name,
                                 [os.path.join(d, '{0}{1}pyx'.format(base_name,
                                                                     os.extsep))],
                                 define_macros=define_macros,
                                 extra_compile_args=extra_compile_args,
                                 extra_link_args=extra_link_args,
                                 include_dirs=include_dirs))


if __name__ == '__main__':
    readme_filename = glob.glob('README*')[0]
    with open(readme_filename, 'r') as f:
        readme = f.read()
    setup(name='span',
          version='0.1',
          author='Phillip Cloud',
          author_email='cpcloud@gmail.com',
          packages=['span', 'span.tdt', 'span.utils', 'span.xcorr'],
          scripts=[os.path.join('bin', 'serv2mat.py')],
          ext_modules=ext_modules,
          license='LICENSE.txt',
          description='Spike train analysis',
          long_description=readme,
          cmdclass={'build_ext': build_ext})
