import os
from osgeo import gdal_array, gdal
from ..utils.geotransform_calcs   import  CalculateClippedGeoTransform, CalculateClippedGeoTransform_RoundedRes
from collections import namedtuple
def SaveLZWTiff(data, _NDV, geotransform, projection, outDir, outName, cOpts=None):
    '''
    Save a numpy array to a single-band LZW-compressed tiff file in the specified folder

    The data-type of the tiff file depends on the input array.
    The file will be saved with LZW compression, predictor 2 , bigtiff yes.
    To over-ride this, provide cOpts as an array of creation option strings.

    GeoTransform should be a 6-tuple conforming to the GDAL geotransform spec. Projection
    should be something retrieved from another file with dataset.GetProjection(), unless
    you like writing out really long complicated projection specifications by hand.
    '''
    if not os.path.exists(outDir):
        os.makedirs(outDir)
    gdType = gdal_array.NumericTypeCodeToGDALTypeCode(data.dtype)

    outDrv = gdal.GetDriverByName('GTiff')

    outRasterName = os.path.join(outDir, outName)
    if cOpts is None:
        cOpts = ["TILED=YES", "SPARSE_OK=FALSE", "BIGTIFF=YES", "COMPRESS=LZW", "PREDICTOR=2"]
    outRaster = outDrv.Create(outRasterName, data.shape[1], data.shape[0], 1, gdType,
                              cOpts)
    outRaster.SetGeoTransform(geotransform)
    outRaster.SetProjection(projection)
    outBand = outRaster.GetRasterBand(1)
    if _NDV:
        outBand.SetNoDataValue(_NDV)
    outBand.WriteArray(data)
    outBand.FlushCache()
    del outBand
    outRaster = None

def ReadAOI_PixelLims_Inplace(gdalDatasetName, xLims, yLims, dataBuffer, useRoundedResolution = False):
    gdalDatasetIn = gdal.Open(gdalDatasetName)
    assert isinstance(gdalDatasetIn, gdal.Dataset)
    if xLims is None:
        xLims = (0, gdalDatasetIn.RasterXSize)
    if yLims is None:
        yLims = (0, gdalDatasetIn.RasterYSize)
    assert max(xLims) <= gdalDatasetIn.RasterXSize
    assert max(yLims) <= gdalDatasetIn.RasterYSize
    inputBnd = gdalDatasetIn.GetRasterBand(1)
    inputBnd.ReadAsArray(xLims[0], yLims[0], xLims[1] - xLims[0], yLims[1] - yLims[0], buf_obj=dataBuffer)


def ReadAOI_PixelLims(gdalDatasetName, xLims, yLims, useRoundedResolution = False):
    ''' Read a subset of band 1 of a GDAL dataset, specified by bounding x and y coordinates.

    Returns a 4-tuple where item 0 is the data as a 2D array, item 1 is the geotransform,
    item 2 is the projection, and item 3 is the nodata value - items 1, 2, 3 can be used
    to save the data or another array representing same area/resolution to a tiff file'''
    gdalDatasetIn = gdal.Open(gdalDatasetName)
    assert isinstance(gdalDatasetIn, gdal.Dataset)
    if xLims is None:
        xLims = (0, gdalDatasetIn.RasterXSize)
    if yLims is None:
        yLims = (0, gdalDatasetIn.RasterYSize)
    assert max(xLims) <= gdalDatasetIn.RasterXSize
    assert max(yLims) <= gdalDatasetIn.RasterYSize
    inputBnd = gdalDatasetIn.GetRasterBand(1)

    inputArr = inputBnd.ReadAsArray(xLims[0], yLims[0], xLims[1] - xLims[0], yLims[1] - yLims[0])

    if useRoundedResolution:
        clippedGT = CalculateClippedGeoTransform_RoundedRes(gdalDatasetIn.GetGeoTransform(), xLims, yLims)
    else:
        clippedGT = CalculateClippedGeoTransform(gdalDatasetIn.GetGeoTransform(), xLims, yLims)

    dsProj = gdalDatasetIn.GetProjection()
    ndv = inputBnd.GetNoDataValue()
    return (inputArr, clippedGT, dsProj, ndv)


RasterProps = namedtuple("RasterProps", ["gt", "proj", "ndv", "width", "height", "res", "datatype"])

def GetRasterProperties(gdalDatasetName):
    gdalDatasetIn = gdal.Open(gdalDatasetName, gdal.GA_ReadOnly)
    assert isinstance(gdalDatasetIn, gdal.Dataset)
    inGT = gdalDatasetIn.GetGeoTransform()
    inProj = gdalDatasetIn.GetProjection()
    inBand = gdalDatasetIn.GetRasterBand(1)
    inNDV = inBand.GetNoDataValue()
    inWidth = gdalDatasetIn.RasterXSize
    inHeight = gdalDatasetIn.RasterYSize
    b = gdalDatasetIn.GetRasterBand(1)
    gdalDType = b.DataType
    res = inGT[1]
    if abs(res - 0.008333333333333) < 1e-9:
        res = "1km"
    elif abs(res - 0.0416666666666667) < 1e-9:
        res = "5km"
    elif abs(res - 0.08333333333333) < 1e-9:
        res = "10km"

    outObj = RasterProps (inGT, inProj, inNDV, inWidth, inHeight, res, gdalDType)
    gdalDatasetIn = None
    return outObj
