import os
from osgeo import gdal_array, gdal
from ..utils.geotransform_calcs import CalculateClippedGeoTransform, CalculatePixelLims
from collections import namedtuple
import numpy as np

RasterProps = namedtuple("RasterProps", ["gt", "proj", "ndv", "width", "height", "res", "datatype"])


class SingleBandTiffFile:

    def __init__(self, filePath, overwriteExisting=False, shouldCache=False):
        self._filePath = filePath
        self._Exists = False
        propsOrNot = self.__tryGetExistingProperties(filePath)
        if propsOrNot:
            self._Properties = propsOrNot
            self._Exists = True
        else:
            self._Exists = False
            self._Properties = None
        self._cacheReads = shouldCache
        self._cachedData = None
        # check it is a tiff, make folders, etc etc blah blah

    def log(self, msg):
        print(msg)

    def Save(self, data):
        if self._Exists:
            raise ValueError("Overwriting existing file not yet supported")
        if self._Properties is None:
            raise ValueError("File properties must be set first using SetProperties(RasterProps)")
        rProps = self._Properties
        outDir = os.path.dirname(self._filePath)
        if not os.path.exists(outDir):
            os.makedirs(outDir)
        gdType = gdal_array.NumericTypeCodeToGDALTypeCode(data.dtype)

        outDrv = gdal.GetDriverByName('GTiff')
        cOpts = ["TILED=YES", "SPARSE_OK=FALSE", "BIGTIFF=YES", "COMPRESS=LZW", "PREDICTOR=2",
                 "NUM_THREADS=ALL_CPUS"]

        dataShape = data.shape
        outShape = (rProps.height, rProps.width)
        if dataShape != outShape:
            raise ValueError("Provided array shape does not match expected file shape - use SavePart to write subsets")

        outRaster = outDrv.create(self._filePath, outShape[1], outShape[0], 1, gdType, cOpts)
        outRaster.SetGeoTransform(rProps.gt)
        outRaster.SetProjection(rProps.proj)
        outBand = outRaster.GetRasterBand(1)

        if rProps.ndv is not None:
            outBand.SetNoDataValue(rProps.ndv)
        outBand.writeArray(data)
        outBand.FlushCache()
        del outBand
        outRaster = None

    def SavePart(self, shape, xOff, yOff):
        raise NotImplementedError("call back another time")

    def __tryGetExistingProperties(self, filePath):
        if not (self._Exists or os.path.exists(filePath)):
            return False
        ds = gdal.Open(filePath, gdal.GA_ReadOnly)
        if not isinstance(ds, gdal.Dataset):
            return False
        inGT = ds.GetGeoTransform()
        inProj = ds.GetProjection()
        inBand = ds.GetRasterBand(1)
        inNDV = inBand.GetNoDataValue()
        gdalDType = inBand.DataType
        inWidth = ds.RasterXSize
        inHeight = ds.RasterYSize
        res = inGT[1]
        if abs(res - 0.008333333333333) < 1e-9:
            res = "1km"
        elif abs(res - 0.0416666666666667) < 1e-9:
            res = "5km"
        elif abs(res - 0.08333333333333) < 1e-9:
            res = "10km"
        outObj = RasterProps(inGT, inProj, inNDV, inWidth, inHeight, res, gdalDType)
        ds = None
        return outObj

    def ReadAll(self, readAsMasked=False):
        # we allow caching so that in situations where we want to read the same multiple times we can just
        # maintain the object associated with it and client code doesn't have to store the arrays etc explicitly
        return self.ReadForPixelLims(xLims=None, yLims=None, readAsMasked=readAsMasked)

    def ReadForPixelLims(self, xLims=None, yLims=None, readAsMasked=False):
        ''' Read a subset of this 1-band dataset, specified by bounding x and y coordinates.

        Returns a 4-tuple where item 0 is the data as a 2D array, item 1 is the geotransform,
        item 2 is the projection, and item 3 is the nodata value - items 1, 2, 3 can be used
        to save the data or another array representing same area/resolution to a tiff file

        Coordinates may be omitted to read the whole file, in which case the data will be cached
        if this was configured at instantiation, allowing quicker repeat reads'''

        if not self._Exists:
            raise ValueError("can't read a file that hasn't been created yet!")

        hasLims = True
        if xLims is None:
            xLims = (0, self._Properties.width)
            hasLims = False
        if yLims is None:
            yLims = (0, self._Properties.height)
            hasLims = False
        assert max(xLims) <= self._Properties.width
        assert max(yLims) <= self._Properties.height
        assert min(xLims) >= 0
        assert min(yLims) >= 0
        assert xLims[1] >= xLims[0]
        assert yLims[1] >= yLims[0]

        clippedGT = CalculateClippedGeoTransform(self.GetGeoTransform(), xLims, yLims)
        dsProj = self.GetProjection()
        ndv = self.GetNdv()

        if self._cachedData is None:
            gdalDatasetIn = gdal.Open(self._filePath, gdal.GA_ReadOnly)
            assert isinstance(gdalDatasetIn, gdal.Dataset)
            inputBnd = gdalDatasetIn.GetRasterBand(1)
            inputArr = inputBnd.ReadAsArray(xLims[0], yLims[0], xLims[1] - xLims[0], yLims[1] - yLims[0])
            if self._cacheReads and not hasLims:
                self.log("Caching data for file " + self._filePath)
                self._cachedData = (inputArr, clippedGT, dsProj, ndv)
            if readAsMasked and ndv is not None:
                return (np.ma.masked_equal(inputArr, ndv), clippedGT, dsProj, ndv)
            else:
                return(inputArr, clippedGT, dsProj, ndv)
        else:
            # the cache can only be populated by a full read so we can just return a slice of it
            if readAsMasked and ndv is not None:
                return (np.ma.masked_equal(self._cachedData[0][yLims[0]:yLims[1],xLims[0]:xLims[1]], ndv),
                        clippedGT, dsProj, ndv)
            else:
                if hasLims:
                    return(self._cachedData[0][yLims[0]:yLims[1],xLims[0]:xLims[1]], clippedGT, dsProj, ndv)
                else:
                    return self._cachedData # avoid making another copy with an explicit check to only do so if needed

    def PopulateCache(self):
        ''' convenience method to load the cache without returning the data

        Calls to ReadForPixelLims for subsets should then run faster.'''

        if not self._cacheReads:
            self.log("This TiffFile has not been configured with caching, not populating cache")
        elif self._cachedData is not None:
            self.log("Cache already populated")
        else:
            self.ReadForPixelLims()

    def SetProperties(self, props):
        if self._Exists:
            raise RuntimeError("(Re)setting properties of an existing tiff file is not supported")
        if not isinstance(props, RasterProps):
            raise ValueError("Props argument must be an instance of the RasterProps named tuple type")
        self._Properties = props

    def GetProjection(self):
        if self._Properties is not None:
            return self._Properties.proj
        raise RuntimeError("Properties have not been set")

    def GetNdv(self):
        if self._Properties is not None:
            return self._Properties.ndv
        raise RuntimeError("Properties have not been set")

    def GetGeoTransform(self):
        if self._Properties is not None:
            return self._Properties.gt
        raise RuntimeError("Properties have not been set")

    def GetShape(self):
        if self._Properties is not None:
            return (self._Properties.height, self._Properties.width)
        raise RuntimeError("Properties have not been set")

    def ReadForLatLonLims(self, lonLims, latLims, readAsMasked=False):
        if lonLims is None or latLims is None:
            return self.ReadAll(readAsMasked)
        else:
            pixelLimsLon, pixelLimsLat = CalculatePixelLims(self._Properties.gt, lonLims, latLims)
            return self.ReadForPixelLims(xLims=pixelLimsLon, yLims=pixelLimsLat, readAsMasked=readAsMasked)





