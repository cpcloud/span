#! /usr/bin/env python

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
    extra_compile_args.append('-O3')

npy_includes = np.get_include()
include_dirs = [npy_includes]

utils_dir = os.path.join('span', 'utils')
ext_modules = [Extension('_clear_refrac',
                         [os.path.join(utils_dir,
                                       'clear_refrac{sep}pyx'.format(sep=os.extsep))],
                         extra_compile_args=extra_compile_args,
                         extra_link_args=extra_link_args,
                         include_dirs=include_dirs),
               Extension('_bin_data',
                         [os.path.join(utils_dir,
                                       'bin_data{sep}pyx'.format(sep=os.extsep))],
                         extra_compile_args=extra_compile_args,
                         extra_link_args=extra_link_args,
                         include_dirs=include_dirs)]

if __name__ == '__main__':
    readme_filename = glob.glob('README*')[0]
    with open(readme_filename, 'r') as f:
        readme = f.read()
    setup(name='span',
          version='0.1',
          author='Phillip Cloud',
          author_email='cpcloud@gmail.com',
          packages=['span', 'span.tdt', 'span.utils'],
          scripts=[os.path.join('bin', 'serv2mat.py')],
          ext_modules=ext_modules,
          license='LICENSE.txt',
          description='Spike train analysis',
          long_description=readme,
          cmdclass={'build_ext': build_ext})
