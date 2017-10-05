from continuous_aggregation import Continuous_Aggregator_Flt
from categorical_aggregation import Categorical_Aggregator
from ...utils.logger import logMessage
from ...utils.raster_tiling import getTiles
from ...io.tiff_management import GetRasterProperties, ReadAOI_PixelLims, SaveLZWTiff, RasterProps
from ...utils.geotransform_calcs import calcAggregatedProperties
from ...utils.raster_tiling import getTiles
import numpy as np
import os
from osgeo import gdal_array
from ..aggregation_values import CategoricalAggregationStats as catstats, \
    ContinuousAggregationStats as contstats,

from collections import namedtuple

from math import floor



class SpatialAggregator:
    '''Runs spatial aggregation across a supplied list of files

    Aggregation can be either continuous (mean, min, max etc) or categorical (majority,
    fractions, etc) - this will be determined from the requested stats.
    '''
    def __init__(self, filesList, outFolder, ndvOut, stats, aggregationArgs):
        assert isinstance(filesList, list) and len(filesList)>0
        assert isinstance(outFolder, str)
        assert isinstance(aggregationArgs, dict)

        self.filesList = filesList
        self.outFolder = outFolder
        self.ndvOut = ndvOut

        assert (all([s in _continuousAggregationStats.All for s in stats]) or
                all([s in _categoricalAggregationStats.All for s in stats]))
        if stats[0] in _continuousAggregationStats.All:
            self._mode = _aggregationModes.Continuous
        else:
            self._mode = _aggregationModes.Categorical
            assert aggregationArgs.has_key("categories")
            categories = aggregationArgs["categories"]
            if isinstance(categories, int):
                assert categories <= 256
                assert categories < 256
                assert categories >= 0
                nCategories = categories
            else:
                assert isinstance(categories, list)
                assert len(categories) <= 256
                assert max(categories) < 256
                assert min(categories) >= 0
                nCategories = len(categories)
            self.categories = categories

        self.stats = stats

        self._aggResolution = None
        self._aggFactor = None
        self._aggShape = None
        if aggregationArgs.has_key("resolution"):
            self.aggregationType = _aggregationTypes.Resolution
            assert (isinstance(aggregationArgs["resolution"], float)
                    or aggregationArgs["resolution"] in ["1km", "5km", "10km"])
            self._aggResolution = aggregationArgs["resolution"]

        elif aggregationArgs.has_key("x_size"):
            assert aggregationArgs.has_key("y_size")
            self.aggregationType = _aggregationTypes.Size
            self._aggShape = (aggregationArgs["y_size"], aggregationArgs["x_size"])
        else:
            assert aggregationArgs.has_key("factor")
            assert (isinstance(aggregationArgs["factor"], int))
            self.aggregationType = _aggregationTypes.Factor
            self._aggFactor = aggregationArgs["factor"]
        if aggregationArgs.has_key["resolution_name"]:
            self._resName = aggregationArgs["resolution_name"]
        else:
            self._resName = "Aggregated"

    def _fnGetterContinuous(self, filename, stat):
        assert self._mode == _aggregationModes.Continuous
        statname = _continuousAggregationStats.Names[stat]
        return (os.path.basename(filename).replace(".tif", "") + "."
                + self._resName + "." + statname + ".tif")

    def _fnGetterCategorical(self, filename, cat, stat):
        assert self._mode == _aggregationModes.Categorical
        outNameTemplate = r'{0!s}.{1!s}.{2!s}.tif'
        statname = _categoricalAggregationStats.Names[stat]
        fOut = outNameTemplate.format(
            os.path.basename(filename).replace(".tif", ""),
            "Class-"+str(cat),
            statname)
        return fOut


    def _estimateAgggregationMemory(self, totalHeight, totalWidth, tileHeight, tileWidth):
        # it's a fairly wrong estimation of peak memory as it doesn't account for anything other
        # than the main arrays themselves, including for instance temporary objects that are formed
        # by casting. never mind, just make all other estimates conservative and fingers crossed
        if self._mode == _aggregationModes.Continuous:
            nPix = totalHeight * totalWidth #tileHeight * tileWidth
            bpp = {_continuousAggregationStats.Count: 2,
                   _continuousAggregationStats.Mean: 8,
                   _continuousAggregationStats.SD: 8, "min": 4, "max": 4, "sum": 4, "range": 4}
            try:
                bppTot = sum([bpp[s] for s in self.stats])
            except KeyError:
                raise KeyError("Invalid statistic specified! Valid items are " + str(bpp.keys()))
            # calculating sd requires calculating mean anyway
            if (("sd" in self.stats) and ("mean" not in self.stats)):
                bppTot += bpp["mean"]
            if (("mean" in self.stats) and ("sum" not in self.stats)):
                # outputting mean requires calculating sum too
                bppTot += bpp["sum"]
            bTot = bppTot * nPix
            # plus the input tile
            bTot += tileWidth * tileHeight * 4
            return bTot
        else:
            # similar for categorical
            pass

    def RunAggregation(self):
        for f in self.filesList:
            self._aggregateFile(f)

    def _aggregateFile(self, filename):
        logMessage("Processing file " + filename)
        # the input files could all be different, if needed
        inputProperties = GetRasterProperties(filename)
        outGT, outShape = calcAggregatedProperties(self.aggregationType,
                                                    inputProperties,
                                                    self._aggFactor,
                                                    self._aggShape,
                                                   self._aggResolution)
        # establish a tile size that can be done in 30GB memory +- wild estimation factor
        tileH = outShape[0]
        tileW = outShape[1]
        bytesTile = self._estimateAgggregationMemory(outShape[0], outShape[1], tileH, tileW)
        while bytesTile > 2e30:
            tileH = tileH // 2
            tileW = tileW // 2
            bytesTile = self._estimateAgggregationMemory(outShape[0], outShape[1], tileH, tileW)
        # and actually just ignore all the non squareness of the world and stuff and use the larger dim
        tileMax = max(tileH, tileW)

        tiles = getTiles(inputProperties.width, inputProperties.height, tileMax)
        logMessage("Processing in {0!s} tiles of {1!s} pixels".format(len(tiles), tileMax))
        if inputProperties.ndv is not None:
            ndvIn = inputProperties.ndv
        else:
            logMessage("No NDV defined")
            ndvIn = self.ndvOut
        if self._mode == _aggregationModes.Continuous:
            aggregator = Continuous_Aggregator_Flt(inputProperties.width, inputProperties.height,
                                               self.outShape[1], self.outShape[0],
                                               self.ndvOut,
                                               self.stats)
        else:
            if isinstance(self.categories, int):
                nCategories = self.categories
            else:
                nCategories = len(self.categories)

            doLikeAdjacency = "like"
            aggregator = Categorical_Aggregator(inputProperties.width, inputProperties.height,
                                                self.outShape[1], self.outShape[0],
                                                self.nCategories,
                                            )
        for tile in tiles:
            logMessage(".")
            xOff = tile[0][0]
            yOff = tile[1][0]
            xSize = tile[0][1] - xOff
            ySize = tile[1][1] - yOff
            inArr = ReadAOI_PixelLims(filename, (xOff, xOff+xSize), (yOff, yOff+ySize))
            aggregator.addTile(inArr, xOff, yOff).astype(np.float32)
        r = aggregator.GetResults()
        for stat in self.stats:
            fnOut = self._fnGetter(os.path.basename(filename), stat)
            logMessage("Saving to "+fnOut)
            if stat in ['min', 'max', 'range']:
                nptype = gdal_array.GDALTypeCodeToNumericTypeCode(inputProperties["datatype"])
                SaveLZWTiff(r[stat].astype(nptype), self.ndvOut, outGT, inputProperties["gt"],
                            self.outFolder, fnOut)
            elif stat in ['mean', 'sd', 'sum']:
                SaveLZWTiff(r[stat], self.ndvOut, outGT, inputProperties["gt"],
                self.outFolder, fnOut)
            elif stat in ['count']:
                SaveLZWTiff(r[stat].astype(np.int32), self.ndvOut, outGT, inputProperties["gt"],
                            self.outFolder, fnOut)
            else:
                assert False

    def _aggregateCategoricalFile(self, filename):
        inputProperties = GetRasterProperties(filename)
        outGT, outShape = calcAggregatedProperties(self.aggregationType,
                                                   inputProperties,
                                                   self._aggFactor,
                                                   self._aggShape,
                                                   self._aggResolution)

        # establish a tile size that can be done in 30GB memory +- wild estimation factor
        tileH = outShape[0]
        tileW = outShape[1]
        bytesTile = self._estimateCategoricalAggregationMemory(outShape[0], outShape[1], tileH, tileW)
        while bytesTile > 2e30:
            tileH = tileH // 2
            tileW = tileW // 2
            bytesTile = self._estimateCategoricalAgggregationMemory(outShape[0], outShape[1], tileH, tileW)
        # and actually just ignore all the non squareness of the world and stuff and use the larger dim
        tileMax = max(tileH, tileW)
        doLikeAdj = "like-adjacency" in self.stats
        if inputProperties.ndv is not None:
            ndvIn = inputProperties.ndv
        else:
            logMessage("No NDV defined")
            ndvIn = self.ndvOut
        tiles = getTiles(inputProperties.width, inputProperties.height, tileMax)
        #nBytesRequired = ds.RasterXSize * ds.RasterYSize * 1 * nCategories
        aggregator = Categorical_Aggregator(inputProperties.width, inputProperties.height,
                                           self.outShape[1], self.outShape[0],
                                           nCategories,
                                           doLikeAdj,
                                           self.ndvOut,
                                           ndvIn)
        for tile in tiles:
            logMessage(".")
            xOff = tile[0][0]
            yOff = tile[1][0]
            xSize = tile[0][1] - xOff
            ySize = tile[1][1] - yOff
            inArr = (ReadAOI_PixelLims(filename, (xOff, xOff + xSize), (yOff, yOff + ySize))
                .astype(np.uint8))
            aggregator.addTile(inArr, xOff, yOff)
        r = aggregator.GetResults()
        for stat in self.stats:
            fnOut = self._fnGetter()






