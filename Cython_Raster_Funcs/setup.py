try:
    from setuptools import setup
    from setuptools import Extension
except ImportError:
    from distutils.core import setup
    from distutils.extension import  Extension
  
from Cython.Build import cythonize

cythonexts = [
    Extension(
        "RasterAggregator_Categorical",
        ["RasterAggregator_Categorical.pyx"],
        extra_compile_args=['/openmp'],
        extra_link_args=['/openmp']
    ),
    Extension(
        "RasterAggregator_float",
        ["RasterAggregator_float.pyx"]
    ),
    Extension(
        "CoastlineMatching", ["CoastlineMatching.pyx"]
    )

]


setup(
    name = "Cython Raster Functions",
    ext_modules = cythonize(cythonexts)
)