from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

setup(
  name='GPM Initialization',
  ext_modules=[Extension('gpm_initialize', ['gpm_initialize.pyx'],)],
  cmdclass={'build_ext': build_ext},
)
