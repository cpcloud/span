#! /usr/bin/env python

import os
import platform
import glob
import shutil

try:
    import numpy as np
except ImportError:
    raise ImportError('NumPy is not installed on your system')

try:
    import Cython
except ImportError:
    raise ImportError('Cython is not installed on your system')

from distutils.core import setup

from Cython.Distutils import build_ext
from Cython.Distutils.extension import Extension

extra_compile_args = []
if platform.system().lower() == 'linux':
    extra_compile_args.append('-march=native')
    

utils_dir = os.path.join('span', 'utils')
ext_modules = [Extension('_clear_refrac',
                         [os.path.join(utils_dir,
                                       'clear_refrac%spyx' % os.extsep)],
                         extra_compile_args=extra_compile_args,
                         include_dirs=[np.get_include()])]

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
          install_requires=['Cython >=0.17', 'numpy >=1.6.0',
                            'scipy >=0.10.0', 'pandas >=0.8.0',
                            'matplotlib >=1.1.1'],
          cmdclass={'build_ext': build_ext})
