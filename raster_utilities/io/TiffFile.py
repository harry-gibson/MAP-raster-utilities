import os
from osgeo import gdal_array, gdal
from ..utils.geotransform_calcs import CalculateClippedGeoTransform, CalculatePixelLims
from collections import namedtuple
import numpy as np

RasterProps = namedtuple("RasterProps", ["gt", "proj", "ndv", "width", "height", "res", "datatype"])

class TemplateProps:
    _EPSG4326 = 'GEOGCS["WGS 84",DATUM["WGS_1984",' \
                'SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],' \
                'AUTHORITY["EPSG","6326"]],' \
                'PRIMEM["Greenwich",0],' \
                'UNIT["degree",0.0174532925199433],' \
                'AUTHORITY["EPSG","4326"]]'
    GLOBAL_1K = RasterProps(gt=(-180.0, 0.008333333333333, 0.0, 90.0, 0.0, -0.008333333333333),
                            proj=_EPSG4326,
                            ndv=None, width=43200, height=21600, res='1km', datatype=6)
    GLOBAL_5K = RasterProps(gt=(-180.0, 0.041666666666667, 0.0, 90.0, 0.0, -0.041666666666667),
                            proj=_EPSG4326,
                            ndv=None, width=8640, height=4320, res='5km', datatype=6)
    GLOBAL_10K = RasterProps(gt=(-180.0, 0.08333333333333, 0.0, 90.0, 0.0, -0.08333333333333),
                            proj=_EPSG4326,
                            ndv=None, width=4320, height=2160, res='10km', datatype=6)
    NOPOLAR_1K = RasterProps(gt=(-180.0, 0.008333333333333, 0.0, 85.0, 0.0, -0.008333333333333),
                            proj=_EPSG4326,
                            ndv=None, width=43200, height=17400, res='1km', datatype=6)
    NOPOLAR_5K = RasterProps(gt=(-180.0, 0.041666666666667, 0.0, 85.0, 0.0, -0.041666666666667),
                            proj=_EPSG4326,
                            ndv=None, width=8640, height=3480, res='5km', datatype=6)
    NOPOLAR_10K = RasterProps(gt=(-180.0, 0.08333333333333, 0.0, 90.0, 0.0, -0.08333333333333),
                             proj=_EPSG4326,
                             ndv=None, width=4320, height=1740, res='10km', datatype=6)


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

    def log(self, msg):
        print(msg)

    def Save(self, data, cOpts=None):
        if self._Exists:
            raise ValueError("Overwriting entire existing file not yet supported: use SavePart instead")
        if self._Properties is None:
            raise ValueError("File properties must be set first using SetProperties(RasterProps)")
        rProps = self._Properties
        outDir = os.path.dirname(self._filePath)
        if not os.path.exists(outDir):
            os.makedirs(outDir)
        incomingType = gdal_array.NumericTypeCodeToGDALTypeCode(data.dtype)
        if incomingType > rProps.datatype:
            Warning("Array has higher type than configured file output, values may be truncated")
        outDrv = gdal.GetDriverByName('GTiff')
        if cOpts is None:
            cOpts = ["TILED=YES", "SPARSE_OK=FALSE", "BIGTIFF=YES", "COMPRESS=LZW", "PREDICTOR=2",
                "NUM_THREADS=ALL_CPUS"]

        outShape = (rProps.height, rProps.width)
        if data is not None:
            dataShape = data.shape
            if dataShape != outShape:
                raise ValueError("Provided array shape does not match expected file shape - use SavePart to write subsets")

        outRaster = outDrv.Create(self._filePath, outShape[1], outShape[0], 1, rProps.datatype, cOpts)

        if rProps.gt is not None:
            outRaster.SetGeoTransform(rProps.gt)
        if rProps.proj is not None:
            outRaster.SetProjection(rProps.proj)

        outBand = outRaster.GetRasterBand(1)
        if rProps.ndv is not None:
            outBand.SetNoDataValue(rProps.ndv)

        if data is not None:
            outBand.WriteArray(data)
        else:
            self.log("Created empty file")
        outBand.FlushCache()
        del outBand
        outRaster = None
        self._Exists = True

    def SavePart(self, data, outOffsetYX, cOpts=None):

        if self._Properties is None:
            raise ValueError("File properties must be set first using SetProperties(RasterProps)")
        rProps = self._Properties

        if outOffsetYX is None:
            outOffset = (0, 0)

        dataShape = data.shape
        currentShape = (rProps.height, rProps.width)
        if ((dataShape[0] + outOffsetYX[0]) > currentShape[0]) or ((dataShape[1] + outOffsetYX[1]) > currentShape[1]):
            raise ValueError("data + offset are larger than the specified image size")
        if dataShape[0] == rProps.height and dataShape[1] == rProps.width:
            isFull = True
        else:
            isFull = False
        if self._Exists:
            ds = gdal.Open(self._filePath, gdal.GA_Update)

        else:
            outDir = os.path.dirname(self._filePath)
            if not os.path.exists(outDir):
                os.makedirs(outDir)
            incomingType = gdal_array.NumericTypeCodeToGDALTypeCode(data.dtype)
            if incomingType > rProps.datatype:
                Warning("Array has higher type than configured file output, values may be truncated")

            outDrv = gdal.GetDriverByName('GTiff')
            if cOpts is None:
                cOpts = ["TILED=YES", "SPARSE_OK=FALSE", "BIGTIFF=YES", "COMPRESS=LZW", "PREDICTOR=2"]
                if isFull:
                    # seem to get corruption when writing subsets of files that seems to occur more when using multithreaded
                    # writers, so use a single thread unless we are writing whole file
                    cOpts.append("NUM_THREADS=ALL_CPUS")
            ds = outDrv.Create(self._filePath, rProps.width, rProps.height, 1, rProps.datatype, cOpts)
            if rProps.gt is not None:
                ds.SetGeoTransform(rProps.gt)
            if rProps.proj is not None:
                ds.SetProjection(rProps.proj)
            if rProps.ndv is not None:
                outBand = ds.GetRasterBand(1)
                outBand.SetNoDataValue(rProps.ndv)
            self._Exists = True

        outBand = ds.GetRasterBand(1)
        outBand.WriteArray(data, outOffsetYX[1], outOffsetYX[0])
        outBand.FlushCache()
        ds = None


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

    def Exists(self):
        return self._Exists

    def ReadAll(self, readAsMasked=False):
        # we allow caching so that in situations where we want to read the same multiple times we can just
        # maintain the object associated with it and client code doesn't have to store the arrays etc explicitly
        return self.ReadForPixelLims(xLims=None, yLims=None, readAsMasked=readAsMasked)

    def ReadForPixelLims(self, xLims=None, yLims=None, readAsMasked=False, existingBuffer=None):
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
        x0 = xLims[0]
        x1 = xLims[1]
        y0 = yLims[0]
        y1 = yLims[1]

        if not isinstance(x0, int):
            x0 = int(x0)
            assert x0 == xLims[0]
        if not isinstance(x1, int):
            x1 = int(x1)
            assert x1 == xLims[1]
        if not isinstance(y0, int):
            y0 = int(y0)
            assert y0 == yLims[0]
        if not isinstance(y1, int):
            y1 = int(y1)
            assert y1 == yLims[1]

        clippedGT = CalculateClippedGeoTransform(self.GetGeoTransform(), (x0,x1), (y0,y1))
        dsProj = self.GetProjection()
        ndv = self.GetNdv()

        if self._cachedData is None:
            gdalDatasetIn = gdal.Open(self._filePath, gdal.GA_ReadOnly)
            assert isinstance(gdalDatasetIn, gdal.Dataset)
            inputBnd = gdalDatasetIn.GetRasterBand(1)
            if existingBuffer is not None:
                inputBnd.ReadAsArray(xLims[0], yLims[0], xLims[1] - xLims[0], yLims[1] - yLims[0], buf_obj=existingBuffer)
                return (None, clippedGT, dsProj, ndv)
            else:
                inputArr = inputBnd.ReadAsArray(x0, y0, x1 - x0, y1 - y0)
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
                return (np.ma.masked_equal(self._cachedData[0][y0:y1, x0:x1], ndv),
                        clippedGT, dsProj, ndv)
            else:
                if hasLims:
                    return(self._cachedData[0][y0:y1, x0:x1], clippedGT, dsProj, ndv)
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

    def GetProperties(self):
        if self._Properties is not None:
            return self._Properties
        else:
            raise RuntimeError("Properties have not been set")

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

    def GetExtent(self):
        pass
        #todo return extent as a lonlim latlims pair of tuples

    def ReadForLatLonLims(self, lonLims, latLims, readIntoLonLims = None, readIntoLatLims = None, readAsMasked=False):
        if lonLims is None or latLims is None:
            return self.ReadAll(readAsMasked)
        else:
            pixelLimsLonInput, pixelLimsLatInput = CalculatePixelLims(self._Properties.gt, longitudeLims=lonLims, latitudeLims=latLims)
            # todo calculate output shape and window we are in
            return self.ReadForPixelLims(xLims=pixelLimsLonInput, yLims=pixelLimsLatInput, readAsMasked=readAsMasked)





