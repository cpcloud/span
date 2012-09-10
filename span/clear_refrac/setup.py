#!/usr/bin/env python

import platform

import numpy as np
from distutils.core import setup
from Cython.Distutils import build_ext
from Cython.Distutils.extension import Extension

extra_compile_args = ['-m64']

if platform.system().lower() == 'linux':
    extra_compile_args += ['-march=native', '-O3']

ext_modules = [Extension('_clear_refrac',
                         ['clear_refrac.pyx'],
                         extra_compile_args=extra_compile_args,
                         include_dirs=[np.get_include()])]

setup(name='clear_refrac',
      cmdclass={'build_ext': build_ext},
      ext_modules=ext_modules)
