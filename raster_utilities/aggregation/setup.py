try:
    from setuptools import setup
    from setuptools import Extension
except ImportError:
    from distutils.core import setup
    from distutils.extension import  Extension
# see https://github.com/cython/cython/wiki/CythonExtensionsOnWindows  for more setup tips on windows
from Cython.Build import cythonize
import os

# https://github.com/cython/cython/wiki/PackageHierarchy
def scandir(dir, files=[]):
    for file in os.listdir(dir):
        path = os.path.join(dir, file)
        if os.path.isfile(path) and path.endswith(".pyx"):
            # get as a dot-separated path excluding the extensio
            files.append((path.replace(os.path.sep, ".")[:-4]))
        elif os.path.isdir(path):
            # recursive scan
            scandir(path, files)
    return files

def makeExtension(extName):
    extPath = extName.replace(".", os.path.sep)+".pyx"
    return Extension(
        extName,
        [extPath],
        include_dirs=["."],
        extra_compile_args=['/openmp', '-O3'],
        extra_link_args=['/openmp']
    )
#
# cythonexts = [
#     Extension(
#         "continuous",
#         ["spatial/core/continuous.pyx"],
#         extra_compile_args=['/openmp'],
#         extra_link_args=['/openmp'],
#         include_dirs=['.']
#     ),
#     Extension(
#         "categorical",
#         ["spatial/core/categorical.pyx"],
#         extra_compile_args=['/openmp'],
#         extra_link_args=['/openmp'],
#         include_dirs=['.']
#     ),
#     Extension(
#         "temporal",
#         ["temporal/core/temporal.pyx"],
#         extra_compile_args = ['/openmp'],
#         extra_link_args = ['/openmp'],
#         include_dirs=['.']
#     )
# ]

spatialextNames = scandir("spatial")
temporalextNames = scandir("temporal")
allextNames = spatialextNames + temporalextNames
extensions = [makeExtension(name) for name in allextNames]

setup(
    name = "MAP Cython Aggregation Functions",
    packages=["spatial", "spatial.core", "temporal", "temporal.core"],
    #ext_modules = cythonize(cythonexts),
    ext_modules=extensions,
    py_modules=['aggregation_values',
                'spatial.spatial_aggregation_runner',
                'temporal.temporal_aggregation_runner']
)