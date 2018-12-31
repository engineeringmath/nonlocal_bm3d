from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

setup(
  name='Compute NND',
  ext_modules=[Extension('compute_nnd', ['compute_nnd.pyx'],)],
  cmdclass={'build_ext': build_ext},
)
