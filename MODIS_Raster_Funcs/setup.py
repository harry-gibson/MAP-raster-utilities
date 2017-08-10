try:
    from setuptools import setup
    from setuptools import Extension
except ImportError:
    from distutils.core import setup
    from distutils.extension import  Extension
# see https://github.com/cython/cython/wiki/CythonExtensionsOnWindows  for more setup tips on windows
from Cython.Build import cythonize

cythonexts = [
    Extension(
        "SynopticData",
        ["SynopticData.pyx"],
        extra_compile_args=['/openmp'],
        extra_link_args=['/openmp']
    )

]


setup(
    name = "Cython Raster Functions",
    ext_modules = cythonize(cythonexts)
)