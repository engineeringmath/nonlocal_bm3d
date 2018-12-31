from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

setup(
  name='GPM Improvement Parallel',
  ext_modules = [
    Extension(
        "gpm_improve_par",
        ["gpm_improve_par.pyx"],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
    )
],
  cmdclass={'build_ext': build_ext},
)
