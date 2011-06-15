from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

# python setup_cython.py build_ext --inplace

import numpy

ext = Extension("calculate_z", ["calculate_z.pyx"],
    include_dirs = [numpy.get_include()])
                
setup(ext_modules=[ext],
      cmdclass = {'build_ext': build_ext})

