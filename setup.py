import os
import platform

import numpy as np
from distutils.core import setup

from Cython.Distutils import build_ext
from Cython.Distutils.extension import Extension

extra_compile_args = []
if platform.system().lower() == 'linux':
    extra_compile_args.append('-march=native')
    

ext_modules = [Extension('_clear_refrac',
                         [os.path.join('span', 'clear_refrac',
                                       'clear_refrac%spyx' % os.extsep)],
                         extra_compile_args=extra_compile_args,
                         include_dirs=[np.get_include()])]

if __name__ == '__main__':
    with open('README.txt') as f:
        readme = f.read()
    setup(name='span',
          version='0.1',
          author='Phillip Cloud',
          author_email='cpcloud@gmail.com',
          packages=['span'],
          scripts=[os.path.join('bin', 'serv2mat.py')],
          license='LICENSE.txt',
          description='Spike train analysis',
          long_description=readme,
          install_requires=[''],
          ext_modules=ext_modules,
          cmdclass={'build_ext': build_ext})
