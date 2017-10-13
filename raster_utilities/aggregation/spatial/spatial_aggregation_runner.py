
import os

import numpy as np
from osgeo import gdal_array
from raster_utilities.aggregation.spatial.core.categorical import Categorical_Aggregator
from raster_utilities.aggregation.spatial.core.continuous import Continuous_Aggregator_Flt

from ..aggregation_values import CategoricalAggregationStats as catstats, \
    ContinuousAggregationStats as contstats, AggregationModes, AggregationTypes
from ...io.tiff_management import GetRasterProperties, ReadAOI_PixelLims, SaveLZWTiff, ReadAOI_PixelLims_Inplace
from ...utils.geotransform_calcs import calcAggregatedProperties
from ...utils.logger import logMessage
from ...utils.raster_tiling import getTiles


class SpatialAggregator:
    '''Runs spatial aggregation across a supplied list of files

    Aggregation can be either continuous (mean, min, max etc) or categorical (majority,
    fractions, etc) - this will be determined from the requested stats. Stats should be
    provided as a list of members from aggregation_values.CategoricalAggregationStats OR
    from aggregation_values.ContinuousAggregationStats

    filesList should be a list of paths to geotiffs. These must all have identical size,
    projection, and geotransform.

    ndvOut can be used to set a specific nodata value in the output files which may be different
    to that defined in the inputs.

    aggregationArgs must have certain keys to define how the aggregation should run.
    ONE of the three constants from AggregationTypes must be provided, either:
    RESOLUTION with a numeric value OR "1km", "5km", or "10km"
    OR
    FACTOR with an integer value
    OR
    SIZE as a tuple of two ints (y_size, x_size)

    Additionally:
    - If the aggregation is categorical then there must be a key "categories" with value of a
    list of integer category values
    - A key "resolution_name" MAY be provided which will be used to generate output
    filenames

    '''
    def __init__(self, filesList, outFolder, ndvOut, stats, aggregationArgs):
        assert isinstance(filesList, list) and len(filesList)>0
        assert isinstance(outFolder, str)
        assert isinstance(aggregationArgs, dict)

        self.filesList = filesList
        self.outFolder = outFolder
        self.ndvOut = ndvOut

        assert (all([s in contstats.ALL for s in stats]) or
                all([s in catstats.ALL for s in stats]))
        if stats[0] in contstats.ALL:
            self._mode = AggregationModes.CONTINUOUS
        else:
            self._mode = AggregationModes.CATEGORICAL
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
                self.categories = categories

        self.stats = stats

        self._aggResolution = None
        self._aggFactor = None
        self._aggShape = None
        atR = AggregationTypes.RESOLUTION
        atF = AggregationTypes.FACTOR
        atS = AggregationTypes.SIZE
        if aggregationArgs.has_key(atR):
            self.aggregationType = atR
            assert (isinstance(aggregationArgs[atR], float)
                    or aggregationArgs[atR] in ["1km", "5km", "10km"])
            self._aggResolution = aggregationArgs[atR]

        elif aggregationArgs.has_key(atS):
            assert (isinstance(aggregationArgs[atS], tuple)
                    and len(aggregationArgs[atS]) == 2)
            self.aggregationType = AggregationTypes.SIZE
            self._aggShape = (aggregationArgs[atS][0], aggregationArgs[atS][1])
        else:
            assert aggregationArgs.has_key(atF)
            assert (isinstance(aggregationArgs[atF], int))
            self.aggregationType = AggregationTypes.FACTOR
            self._aggFactor = aggregationArgs[atF]
        if aggregationArgs.has_key("resolution_name"):
            self._resName = aggregationArgs["resolution_name"]
        else:
            self._resName = "Aggregated"

        if aggregationArgs.has_key("mem_limit_gb"):
            self._gbLimit = aggregationArgs["mem_limit_gb"]
        else:
            self._gbLimit = 30

    def _fnGetter(self, filename, stat, cat=None):
        if self._mode == AggregationModes.CONTINUOUS:
            statname = stat
            return (os.path.basename(filename).replace(".tif", "") + "."
                    + self._resName + "." + statname + ".tif")

        else:
            assert self._mode == AggregationModes.CATEGORICAL
            assert ((cat is not None) or
                    (stat == catstats.MAJORITY))
            if stat == catstats.MAJORITY:
                outNameTemplate = r'{0!s}.{1!s}.tif'
                statname = stat
                fOut = outNameTemplate.format(
                    os.path.basename(filename).replace(".tif", ""),
                    statname)
            else:
                outNameTemplate = r'{0!s}.{1!s}.{2!s}.tif'
                statname = stat
                fOut = outNameTemplate.format(
                    os.path.basename(filename).replace(".tif", ""),
                    "Class-" + str(cat),
                    statname)
            return fOut


    def _estimateAgggregationMemory(self, totalHeight, totalWidth, tileHeight, tileWidth):
        # it's a fairly wrong estimation of peak memory as it doesn't account for anything other
        # than the main arrays themselves, including for instance temporary objects that are formed
        # by casting and temporary copies that i can't seem to track down.
        # never mind, just make all other estimates conservative and fingers crossed
        if self._mode == AggregationModes.CONTINUOUS:
            nPix = totalHeight * totalWidth #tileHeight * tileWidth
            bpp = {contstats.COUNT: 2,
                   contstats.MEAN: 8,contstats.SD: 8,
                   contstats.MIN: 4, contstats.MAX: 4, contstats.SUM: 4, contstats.RANGE: 4}
            try:
                bppTot = sum([bpp[s] for s in self.stats])
            except KeyError:
                raise KeyError("Invalid statistic specified! Valid items are " + str(bpp.keys()))
            # calculating sd requires calculating mean anyway
            if ((contstats.SD in self.stats) and (contstats.MEAN not in self.stats)):
                bppTot += bpp[contstats.MEAN]
            if ((contstats.MEAN in self.stats) and (contstats.SUM not in self.stats)):
                # outputting mean requires calculating sum too
                bppTot += bpp[contstats.SUM]
            bTot = bppTot * nPix * 2 # double for all the casting we do.... this is rough right
            # plus the input tile
            bTot += tileWidth * tileHeight * 4
            return bTot
        else:
            nPix = totalHeight * totalWidth
            bpp = { catstats.MAJORITY: 5,
                    catstats.FRACTIONS: 4 * self.nCategories,
                    catstats.LIKEADJACENCIES: 4 * self.nCategories
            }
            try:
                bppTot = sum([bpp[s] for s in self.stats])
            except KeyError:
                raise KeyError("Invalid statistic specified! Valid items are "  + str(bpp.keys()))
            bTot = bppTot * nPix * 2 # double for all the casting we do.... this is rough right
            # plus the input tile which is actually going to be twice the size called for as the tiler just
            # works off the max dimension, hence *2, but actually somewhere
            # some copies must be getting made and i can't figure out where... so *4
            bTot += tileWidth * tileHeight * 4
            return bTot

    def RunAggregation(self):
        for f in self.filesList:
            self._aggregateFile(f)

    def _aggregateFile(self, filename):
        logMessage("Processing file " + filename)
        # the input files could all be different, if needed
        inputProperties = GetRasterProperties(filename)
        npDataType = gdal_array.GDALTypeCodeToNumericTypeCode(inputProperties.datatype)
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
        byteLim = self._gbLimit * (pow(2,30))
        while bytesTile > byteLim:
            tileH = tileH // 2
            tileW = tileW // 2
            bytesTile = self._estimateAgggregationMemory(outShape[0], outShape[1], tileH, tileW)
            splits += 1
            ntiles = 2^splits
            if ntiles > 4096:
                # 2^12
                raise MemoryError("This is going to take too many tiles: try processing a smaller output area")
        estMem = bytesTile / float(pow(2,30))
        print ("estimated GB: "+str(estMem))

        # and actually just ignore all the non squareness of the world and stuff and use the larger dim
        tileMax = max(tileH, tileW)

        tiles = getTiles(inputProperties.width, inputProperties.height, tileMax)
        logMessage("Processing in {0!s} tiles of {1!s} pixels".format(len(tiles), tileMax))
        if inputProperties.ndv is not None:
            catNdv = inputProperties.ndv
            logMessage("Incoming nodata value is "+str(catNdv))
        else:
            logMessage("No NDV defined")
            catNdv = None
        if self._mode == AggregationModes.CONTINUOUS:
            aggregator = Continuous_Aggregator_Flt(inputProperties.width, inputProperties.height,
                                               outShape[1], outShape[0],
                                               self.ndvOut,
                                               self.stats)
        else:
            doLikeAdjacency = catstats.LIKEADJACENCIES in self.stats
            aggregator = Categorical_Aggregator(inputProperties.width, inputProperties.height,
                                                outShape[1], outShape[0],
                                                self.nCategories,
                                                doLikeAdjacency,
                                                self.ndvOut, catNdv
                                            )
        # for each tile read the data and add to the aggregator
        # todo (maybe) don't bother adding complete nodata tiles? (just add fake one)
        for tile in tiles:
            logMessage(".", newline=False)
            xOff = tile[0][0]
            yOff = tile[1][0]
            xSize = tile[0][1] - xOff
            ySize = tile[1][1] - yOff

            inArr = np.empty(shape=(ySize, xSize), dtype=npDataType)
            ReadAOI_PixelLims_Inplace(filename,
                                      (xOff, xOff+xSize),
                                      (yOff, yOff+ySize),
                                      inArr)
            #inArr, x1, x2, x3 = ReadAOI_PixelLims(filename,
            #                                        (xOff, xOff+xSize),
            #                                        (yOff, yOff+ySize))
            # todo check the actual datatypes of the files here esp for categorical
            if self._mode == AggregationModes.CONTINUOUS:
                aggregator.addTile(inArr.astype(np.float32), xOff, yOff)
            else:
                aggregator.addTile(inArr#.astype(np.uint8)
                                   , xOff, yOff)
            inArr = None

        # all tiles added, get results
        r = aggregator.GetResults()
        for stat in self.stats:
            logMessage("Saving outputs: " + stat)
            if self._mode == AggregationModes.CONTINUOUS:
                fnOut = self._fnGetter(os.path.basename(filename), stat)
                if stat in [contstats.MIN, contstats.MAX, contstats.RANGE]:
                    nptype = gdal_array.GDALTypeCodeToNumericTypeCode(inputProperties.datatype)
                    SaveLZWTiff(r[stat].astype(nptype), self.ndvOut, outGT, inputProperties.proj,
                                self.outFolder, fnOut)
                elif stat in [contstats.MEAN, contstats.SD, contstats.SUM]:
                    SaveLZWTiff(r[stat], self.ndvOut, outGT, inputProperties.proj,
                    self.outFolder, fnOut)
                elif stat in [contstats.COUNT]:
                    SaveLZWTiff(r[stat].astype(np.int32), self.ndvOut, outGT, inputProperties.proj,
                                self.outFolder, fnOut)
                else:
                    assert False
            else:
                if stat == catstats.MAJORITY:
                    fnOut = self._fnGetter(os.path.basename(filename),
                                           stat)
                    SaveLZWTiff(r[stat], catNdv, outGT, inputProperties.proj,
                                self.outFolder, fnOut)
                else:
                    for i in range(0, self.nCategories):
                        outValue = r['valuemap'][i]
                        fnOut = self._fnGetter(os.path.basename(filename),
                                               stat,
                                               outValue)
                        SaveLZWTiff(r[stat][i], self.ndvOut, outGT, inputProperties.proj,
                                         self.outFolder, fnOut)







