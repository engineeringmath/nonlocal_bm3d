from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

setup(
  name='GPM Improvement',
  ext_modules=[Extension('gpm_improve', ['gpm_improve.pyx'],)],
  cmdclass={'build_ext': build_ext},
)
