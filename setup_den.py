from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

setup(
  name='GPM Denoising',
  ext_modules=[Extension('gpm_denoise', ['gpm_denoise.pyx'],)],
  cmdclass={'build_ext': build_ext},
)
