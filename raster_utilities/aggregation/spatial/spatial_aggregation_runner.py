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
    ContinuousAggregationStats as contstats, AggregationModes, AggregationTypes

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

        assert (all([s in contstats.All for s in stats]) or
                all([s in catstats.All for s in stats]))
        if stats[0] in contstats.All:
            self._mode = AggregationModes.Continuous
        else:
            self._mode = AggregationModes.Categorical
            assert aggregationArgs.has_key("categories")
            categories = aggregationArgs["categories"]
            if isinstance(categories, int):
                assert categories <= 256
                assert categories < 256
                assert categories >= 0
                # todo set categories to range here
                self.nCategories = categories
            else:
                assert isinstance(categories, list)
                assert len(categories) <= 256
                assert max(categories) < 256
                assert min(categories) >= 0
                self.nCategories = len(categories)

        self.stats = stats

        self._aggResolution = None
        self._aggFactor = None
        self._aggShape = None
        if aggregationArgs.has_key("resolution"):
            self.aggregationType = AggregationTypes.Resolution
            assert (isinstance(aggregationArgs["resolution"], float)
                    or aggregationArgs["resolution"] in ["1km", "5km", "10km"])
            self._aggResolution = aggregationArgs["resolution"]

        elif aggregationArgs.has_key("x_size"):
            assert aggregationArgs.has_key("y_size")
            self.aggregationType = AggregationTypes.Size
            self._aggShape = (aggregationArgs["y_size"], aggregationArgs["x_size"])
        else:
            assert aggregationArgs.has_key("factor")
            assert (isinstance(aggregationArgs["factor"], int))
            self.aggregationType = AggregationTypes.Factor
            self._aggFactor = aggregationArgs["factor"]
        if aggregationArgs.has_key["resolution_name"]:
            self._resName = aggregationArgs["resolution_name"]
        else:
            self._resName = "Aggregated"

    def _fnGetterContinuous(self, filename, stat):
        assert self._mode == AggregationModes.Continuous
        statname = contstats.Names[stat]
        return (os.path.basename(filename).replace(".tif", "") + "."
                + self._resName + "." + statname + ".tif")

    def _fnGetterCategorical(self, filename, cat, stat):
        assert self._mode == AggregationModes.Categorical
        outNameTemplate = r'{0!s}.{1!s}.{2!s}.tif'
        statname = catstats.Names[stat]
        fOut = outNameTemplate.format(
            os.path.basename(filename).replace(".tif", ""),
            "Class-"+str(cat),
            statname)
        return fOut


    def _estimateAgggregationMemory(self, totalHeight, totalWidth, tileHeight, tileWidth):
        # it's a fairly wrong estimation of peak memory as it doesn't account for anything other
        # than the main arrays themselves, including for instance temporary objects that are formed
        # by casting. never mind, just make all other estimates conservative and fingers crossed
        if self._mode == AggregationModes.Continuous:
            nPix = totalHeight * totalWidth #tileHeight * tileWidth
            bpp = {contstats.Count: 2,
                   contstats.Mean: 8,contstats.SD: 8,
                   contstats.Min: 4, contstats.Max: 4, contstats.Sum: 4, contstats.Range: 4}
            try:
                bppTot = sum([bpp[s] for s in self.stats])
            except KeyError:
                raise KeyError("Invalid statistic specified! Valid items are " + str(bpp.keys()))
            # calculating sd requires calculating mean anyway
            if ((contstats.SD in self.stats) and (contstats.Mean not in self.stats)):
                bppTot += bpp[contstats.Mean]
            if ((contstats.Mean in self.stats) and (contstats.Sum not in self.stats)):
                # outputting mean requires calculating sum too
                bppTot += bpp[contstats.Sum]
            bTot = bppTot * nPix
            # plus the input tile
            bTot += tileWidth * tileHeight * 4
            return bTot
        else:
            nPix = totalHeight * totalWidth
            bpp = { catstats.Majority: 5,
                    catstats.Fractions: 4 * self.nCategories,
                    catstats.LikeAdjacencies: 4 * self.nCategories
            }
            try:
                bppTot = sum([bpp[s] for s in self.stats])
            except KeyError:
                raise KeyError("Invalid statistic specified! Valid items are "  + str(bpp.keys()))
            bTot = bppTot * nPix
            # plus the input tile which is always 8 bit
            bTot += tileWidth * tileHeight * 1
            return bTot

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
        tileH = inputProperties.height
        tileW = inputProperties.width
        bytesTile = self._estimateAgggregationMemory(outShape[0], outShape[1], tileH, tileW)
        splits = 1
        while bytesTile > 2e30:
            tileH = tileH // 2
            tileW = tileW // 2
            bytesTile = self._estimateAgggregationMemory(outShape[0], outShape[1], tileH, tileW)
            splits += 1
            ntiles = 2^splits
            if ntiles > 4096:
                # 2^12
                raise MemoryError("This is going to take too many tiles: try processing a smaller output area")
        # and actually just ignore all the non squareness of the world and stuff and use the larger dim
        tileMax = max(tileH, tileW)

        tiles = getTiles(inputProperties.width, inputProperties.height, tileMax)
        logMessage("Processing in {0!s} tiles of {1!s} pixels".format(len(tiles), tileMax))
        if inputProperties.ndv is not None:
            catNdv = inputProperties.ndv
        else:
            logMessage("No NDV defined")
            catNdv = None
        if self._mode == AggregationModes.Continuous:
            aggregator = Continuous_Aggregator_Flt(inputProperties.width, inputProperties.height,
                                               self.outShape[1], self.outShape[0],
                                               self.ndvOut,
                                               self.stats)
        else:
            doLikeAdjacency = catstats.LikeAdjacencies in self.stats
            aggregator = Categorical_Aggregator(inputProperties.width, inputProperties.height,
                                                self.outShape[1], self.outShape[0],
                                                self.nCategories,
                                                doLikeAdjacency,
                                                self.ndvOut, catNdv
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
            fnOut = self._fnGetterContinuous(os.path.basename(filename), stat)
            logMessage("Saving to "+fnOut)
            if self._mode == AggregationModes.Continuous:
                if stat in [contstats.Min, contstats.Max, contstats.Range]:
                    nptype = gdal_array.GDALTypeCodeToNumericTypeCode(inputProperties["datatype"])
                    SaveLZWTiff(r[stat].astype(nptype), self.ndvOut, outGT, inputProperties["proj"],
                                self.outFolder, fnOut)
                elif stat in [contstats.Mean, contstats.SD, contstats.Sum]:
                    SaveLZWTiff(r[stat], self.ndvOut, outGT, inputProperties["proj"],
                    self.outFolder, fnOut)
                elif stat in [contstats.Count]:
                    SaveLZWTiff(r[stat].astype(np.int32), self.ndvOut, outGT, inputProperties["proj"],
                                self.outFolder, fnOut)
                else:
                    assert False
            else:
                if stat == catstats.Majority:
                    SaveLZWTiff(r[stat], catNdv, outGT, inputProperties["proj"],
                                self.outFolder, fnOut)
                else:
                    for i in range(0, self.nCategories):
                        outValue = r['valuemap'][i]
                        if stat == catstats.Fractions:
                            fnOut = self._fnGetterCategorical(os.path.basename(filename),
                                                            outValue,
                                                            "Fraction")
                            SaveLZWTiff((r[stat][i], self.ndvOut, outGT, inputProperties["proj"],
                                         self.outFolder, fnOut))
                        elif stat == catstats.LikeAdjacencies:
                            fnOut = self._fnGetterCategorical(os.path.basename(filename),
                                                              outValue,
                                                              "LikeAdjacency")
                            SaveLZWTiff((r[stat][i], self.ndvOut, outGT, inputProperties["proj"],
                                         self.outFolder, fnOut))

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
        doLikeAdj = catstats.LikeAdjacencies in self.stats
        if inputProperties.ndv is not None:
            ndvIn = inputProperties.ndv
        else:
            logMessage("No NDV defined")
            ndvIn = self.ndvOut
        tiles = getTiles(inputProperties.width, inputProperties.height, tileMax)
        #nBytesRequired = ds.RasterXSize * ds.RasterYSize * 1 * nCategories
        aggregator = Categorical_Aggregator(inputProperties.width, inputProperties.height,
                                           self.outShape[1], self.outShape[0],
                                           self.nCategories,
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






