import os, sys
from distutils.core import setup
from distutils.extension import  Extension

try:
    from Cython.Distutils import build_ext
except:
    print("Cython does not seem to be installed")
    sys.exit(1)


#from Cython.Build import cythonize


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
        # LINUX:
        #extra_compile_args=['-fopenmp', '-O3'],
        #extra_link_args=['-fopenmp']
        # WINDOWS:
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
    name = "Aggregation",
    description = "MAP Cython Raster Aggregation and Summary Functions",
    author = "Harry Gibson",
    author_email = "harry.s.gibson@gmail.com",
    long_description='''
    Contains functions written in Cython for the aggregation of large numerical arrays, 
    probably representing the contents of singleband raster images. Functions are provided 
    for aggregating a raster spatially (reducing resolution) or temporally (the value at a point 
    being derived from multiple other rasters of the same alignment). Aggregations are done in memory 
    so depending on the output size a lot of RAM may be required, but input data can exceed the 
    available RAM.
    ''',
    packages=["spatial", "spatial.core", "temporal", "temporal.core"],
    #ext_modules = cythonize(cythonexts),
    ext_modules=extensions,
    cmdclass={'build_ext':build_ext},
    py_modules=['aggregation_values',
                'spatial.SpatialAggregator',
                'temporal.TemporalAggregator']
)