import os
import numpy as np
from .TiffFile import RasterProps, SingleBandTiffFile
from osgeo import gdal_array

def SaveLZWTiff(data, _NDV, geotransform, projection, outDir, outName,
                cOpts=None,
                outShape=None,
                outOffset=None):
    '''
    Save a numpy array to a single-band LZW-compressed tiff file in the specified folder

    Deprecated, provided as a convenience wrapper to the SingleBandTiffFile object for
    existing code.

    The file should not already exist.

    The data-type of the tiff file depends on the input array.
    The file will be saved with LZW compression, predictor 2 , bigtiff yes.
    To over-ride this, provide cOpts as an array of creation option strings.

    GeoTransform should be a 6-tuple conforming to the GDAL geotransform spec. Projection
    should be something retrieved from another file with dataset.GetProjection(), unless
    you like writing out really long complicated projection specifications by hand.
    '''

    if outShape is None:
        outShape = data.shape
    if outOffset is None:
        outOffset = (0,0)
    else:
        raise NotImplementedError("call back another time")

    if outShape[0] < data.shape[0] or outShape[1] < data.shape[1]:
        raise ValueError("cannot write to a smaller image than the supplied data")
    if ((data.shape[0] + outOffset[0]) > outShape[0]) or ((data.shape[1] + outOffset[1]) > outShape[1]):
        raise ValueError("data + offset are larger than the specified image size")

    rprop = RasterProps(gt=geotransform, proj=projection, ndv=_NDV,
                        width=outShape[1], height=outShape[0],
                        res="5km", datatype=gdal_array.NumericTypeCodeToGDALTypeCode(data.dtype))
    outPath = os.path.join(outDir, outName)
    writer = SingleBandTiffFile(filePath=outPath)
    writer.SetProperties(rprop)
    writer.Save(data, cOpts)

def SaveLZWTiffPart(data, shape, xOff, yOff, geotransform, projection, outDir, outName, cOpts=None):
    raise  NotImplementedError()
    #if not os.path.exists(outDir):
    #    os.makedirs(outDir)

def ReadAOI_PixelLims_Inplace(gdalDatasetName, xLims, yLims, dataBuffer, useRoundedResolution = False):
    raise NotImplementedError()

def ReadAOI_PixelLims(gdalDatasetName, xLims, yLims, useRoundedResolution=False, maskNoData=False):
    ''' Read a subset of band 1 of a GDAL dataset, specified by bounding x and y coordinates.

    Returns a 4-tuple where item 0 is the data as a 2D array, item 1 is the geotransform,
    item 2 is the projection, and item 3 is the nodata value - items 1, 2, 3 can be used
    to save the data or another array representing same area/resolution to a tiff file

    Deprecated, provided as a convenience wrapper for SingleBandTiffFile for compatibility with existing code'''

    f = SingleBandTiffFile(gdalDatasetName)
    if not f._Exists:
        raise RuntimeError("File does not exist")
    if useRoundedResolution:
        raise NotImplementedError("Sorry, the incorrectly-rounded coordinates are no longer supported")
    return f.ReadForPixelLims(xLims=xLims, yLims=yLims, readAsMasked=maskNoData)

def GetRasterProperties(gdalDatasetName):
    '''Returns the properties of an existing geotiff file

    Return object is a RasterProps named tuple type, consisting of the following fields
    "gt", "proj", "ndv", "width", "height", "res", "datatype"
    '''
    f = SingleBandTiffFile(gdalDatasetName)
    return f._Properties

