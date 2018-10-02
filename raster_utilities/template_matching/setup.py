try:
    from setuptools import setup
    from setuptools import Extension
except ImportError:
    from distutils.core import setup
    from distutils.extension import  Extension

from Cython.Build import cythonize
import os

# https://github.com/cython/cython/wiki/PackageHierarchy
def scandir(dir, files=[]):
    for file in os.listdir(dir):
        path = os.path.join(dir, file)
        if os.path.isfile(path) and path.endswith(".pyx"):
            # get as a dot-separated path excluding the extension
            if dir == ".":
                files.append(file[:-4])
            else:
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

extNames = scandir(".")
extensions = [makeExtension(name) for name in extNames]

setup(
    name = "Coastline Matching",
    description = "MAP Cython Coastline Matching Functions",
    author = "Harry Gibson",
    author_email = "harry.s.gibson@gmail.com",
    long_description='''
    Contains functions written in Cython for the matching of data in one raster to a template provided in 
    another raster. This is intended for ensuring that a given defintion of "land area" will always have data 
    pixels. There are two types of matching: infilling, and redistribution. Infilling is akin to the "Nibble" 
    operation in ArcGIS: where an incoming dataset has missing values in some "land" pixels according to the 
    template dataset, the missing values are allocated based on the nearest existing data; meanwhile data that 
    are found "in the sea" according to the template can just be removed (set to missing). Redistribution is for 
    data such as population counts where the totals must be maintained. We cannot just clip data that occur "in the sea". 
    Instead the values of such cells are re-allocated to the nearest "land" pixel (meanwhile any land pixels with no
    data can be set to zero).
    ''',
    packages=["template_matching"],
    #ext_modules = cythonize(cythonexts),
    ext_modules=extensions,

)