import os
from osgeo import gdal_array, gdal
from GeotransformCalcs import  CalculateClippedGeoTransform

def SaveLZWTiff(data, _NDV, geotransform, projection, outDir, outName, cOpts=None):
    '''
    Save a numpy array to a single-band LZW-compressed tiff file in the specified folder

    The data-type of the tiff file depends on the input array.
    The file will be saved with LZW compression, predictor 2, sparse ok, bigtiff yes.
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
        cOpts = ["TILED=YES", "SPARSE_OK=TRUE", "BIGTIFF=YES", "COMPRESS=LZW", "PREDICTOR=2"]
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

def ReadAOI_PixelLims(gdalDatasetName, xLims, yLims):
    ''' Read a subset of band 1 of a GDAL dataset, specified by bounding x and y coordinates.

    Returns a 2-tuple where item 1 is the data as a 2D array and item 2 is the geotransform
    it should be saved to'''
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

    clippedGT = CalculateClippedGeoTransform(gdalDatasetIn.GetGeoTransform(), xLims, yLims)
    dsProj = gdalDatasetIn.GetProjection()
    ndv = inputBnd.GetNoDataValue()
    return (inputArr, clippedGT, dsProj, ndv)



