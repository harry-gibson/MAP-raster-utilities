
import os, math

import numpy as np
from osgeo import gdal_array
from raster_utilities.aggregation.spatial.core.categorical import Categorical_Aggregator
from raster_utilities.aggregation.spatial.core.continuous import Continuous_Aggregator_Flt

from ..aggregation_values import CategoricalAggregationStats as catstats, \
    ContinuousAggregationStats as contstats, AggregationModes, AggregationTypes, SnapTypes
from ...io.tiff_management import GetRasterProperties, ReadAOI_PixelLims, SaveLZWTiff, RasterProps # ReadAOI_PixelLims_Inplace
from ...utils.geotransform_calcs import SanitiseResolution, SnapAndAlignGeoTransform
#from ...utils.geotransform_calcs import calcAggregatedProperties
from ...utils.logger import MessageLogger, LogLevels
from ...utils.raster_tiling import getTiles


class SpatialAggregator:
    '''Runs spatial aggregation across a supplied list of files

    Aggregation can be either continuous (mean, min, max etc) or categorical (majority,
    fractions, etc) - this will be determined from the requested stats. 

    filesList should be a list of paths to geotiffs. These must all have identical size,
    projection, and geotransform.

    outFolder gives the location that files will be written to with filenames determined automatically
    from the input and the aggregation type.
    
    ndvOut can be used to set a specific nodata value in the output files which may be different
    to that defined in the inputs.

    stats must be a list of output statistics, from EITHER ContinuousAggregationStats OR 
    CategoricalAggregationStats; it may not contain values from both enums. This will determine the 
    aggregation method and the available output statistics.
    
    aggregationArgs must have certain keys to define how the aggregation should run.
    * ONE key "aggregation_type" must be provided with a value that is one of the three constants from 
    AggregationTypes, either "RESOLUTION", "FACTOR", or "SIZE"
    * ONE key "aggregation_specifier" must be provided with a value dependent on the aggregation_type:
      - If aggregation_type==RESOLUTION, then a numeric value OR "1km", "5km", or "10km"
      - If aggregation_type==FACTOR, then an integer value
      - If aggreagation_type==SIZE, then a tuple of two ints (y_size, x_size)
    Additionally:
    - If the aggregation is categorical then there must be a key "categories" with value of a
    list of integer category values
    - A key "resolution_name" MAY be provided which will be used to generate output
    filenames

    '''
    def __init__(self, filesList, outFolder, ndvOut, stats, aggregationArgs, loggingLevel = LogLevels.INFO):
        assert isinstance(filesList, list) and len(filesList)>0
        assert isinstance(outFolder, str)
        assert isinstance(aggregationArgs, dict)

        self._Logger = MessageLogger(loggingLevel)
        logMessage = self._Logger.logMessage
        self.filesList = filesList
        self.outFolder = outFolder
        self.ndvOut = ndvOut
        mystats = []
        self._mode = None
        for s in stats:
            if isinstance(s, contstats):
                assert s.value in contstats.ALL.value
                if self._mode is None:
                    self._mode = AggregationModes.CONTINUOUS
                else:
                    assert self._mode == AggregationModes.CONTINUOUS
                mystats.append(s)
            elif isinstance(s, catstats):
                assert s.value in catstats.ALL.value
                if self._mode is None:
                    self._mode = AggregationModes.CATEGORICAL
                else:
                    assert self._mode == AggregationModes.CATEGORICAL
                mystats.append(s)
            else:
                try:
                    sCast = contstats(s)
                    if self._mode is None:
                        self._mode = AggregationModes.CONTINUOUS
                    else:
                        assert self._mode == AggregationModes.CONTINUOUS
                    mystats.append(sCast)
                except ValueError:
                    sCast = catstats(s) # and except if it fails again
                    if self._mode is None:
                        self._mode = AggregationModes.CATEGORICAL
                    else:
                        assert self._mode == AggregationModes.CATEGORICAL
                    mystats.append(sCast)
        self.stats = mystats

        if self._mode == AggregationModes.CATEGORICAL:
            if not aggregationArgs.has_key("categories"):
                raise ValueError("If a categorical-type aggregation stat is requested, then a 'categories' parameter must "
                             "be given which is a list of integer category values")
            categories = aggregationArgs["categories"]
            assert isinstance(categories, list)
            assert len(categories) <= 256
            assert max(categories) < 256
            assert min(categories) >= 0
            self.nCategories = len(categories)
            self.categories = categories

        if not aggregationArgs.has_key("aggregation_type"):
            raise ValueError("Aggregation arguments object must have a key 'aggregation_type' with value being a "
                             "member of the AggregationTypes enumeration")
        if not aggregationArgs.has_key("aggregation_specifier"):
            raise ValueError("Aggregation arguments object must have a key 'aggregation_specifier' with value type "
                             "being dependent on the type of aggregation requested")

        aggType = aggregationArgs['aggregation_type']
        aggSpec = aggregationArgs['aggregation_specifier']
        self._aggregationType = aggType

        if aggregationArgs.has_key("assume_correct_input"):
            self._AssumeCorrectInput = aggregationArgs["assume_correct_input"]
        else:
            self._AssumeCorrectInput = False
            logMessage("By default, input resolution / alignment will be pre-snapped", LogLevels.INFO)

        if aggregationArgs.has_key("sanitise_resolution"):
            self._SanitiseResolution = aggregationArgs["sanitise_resolution"]
        else:
            self._SanitiseResolution = True
            logMessage("By default, output resolution will be sanitised", LogLevels.INFO)

        if aggregationArgs.has_key("snap_alignment"):
            self._SnapType = aggregationArgs["snap_alignment"]
        else:
            self._SnapType = SnapTypes.NEAREST
            logMessage("By default, output geotransform will be snapped to nearest globally-aligned pixel origin", LogLevels.INFO)

        if aggType == AggregationTypes.RESOLUTION:
            assert (isinstance(aggSpec, float)
                    or aggSpec in ["1km", "5km", "10km"])
            self._aggregationSpec = aggSpec

        elif aggType == AggregationTypes.SIZE:
            ok = (isinstance(aggSpec, tuple)
                    and len(aggSpec) == 2)
            if not ok:
                raise ValueError("For aggregation type 'SIZE' the specification must be a 2-tuple giving output height*width")
            self._aggregationSpec= aggSpec

        elif aggType == AggregationTypes.FACTOR:
            ok = isinstance(aggSpec, int) and (aggSpec > 1)
            if not ok:
                raise ValueError("For aggregation type 'FACTOR' the specification must be an integer >1 giving cell size factor")
            self._aggregationSpec = aggSpec

        else:
            raise ValueError("Aggregation type must be a member of the AggregationTypes enumeration")

        if aggregationArgs.has_key("resolution_name"):
            self._resName = aggregationArgs["resolution_name"]
        else:
            self._resName = "Aggregated"

        if aggregationArgs.has_key("mem_limit_gb"):
            self._gbLimit = aggregationArgs["mem_limit_gb"]
        else:
            self._gbLimit = 30


    def _fnGetter(self, filename, stat, cat=None):
        outNameTemplate = "{0!s}.{1!s}.{2!s}.{3!s}.{4!s}.{5!s}.tif"
        outNameTemplate_NonTemporal = "{0!s}.{1!s}.{2!s}.tif"

        statname = stat.value
        existingParts = os.path.basename(filename).split(".")
        if len(existingParts)==7:
            # assume it's a 6-token mastergrid format name (and .tif on the end)
            varTag = existingParts[0]
            yrTag = existingParts[1]
            mthTag = existingParts[2]
            temporalTag = existingParts[3]
        else:
            varTag = "-".join(existingParts[:-1])
            yrTag = None
            mthTag = None
            temporalTag = None
        if stat == catstats.LIKEADJACENCIES or stat == catstats.FRACTIONS:
            # assert cat is not None
            if cat is None:
                cat = ""
            varTag = varTag + "-Class-" + str(cat)
        if yrTag is None or mthTag is None or temporalTag is None:
            outname = outNameTemplate_NonTemporal.format(varTag, self._resName, statname)
        else:
            outname = outNameTemplate.format(
            varTag, yrTag, mthTag, temporalTag,
            self._resName, statname
            )
        return outname


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

    def GetAggregatedProperties(self, inRasterProps):
        ''' Given an input raster, get the post-aggregation geotransform and dimensions

        method should be one of AggregationTypes.Factor, AggregationTypes.Size, or
        AggregationTypes.Resolutions (or a string "factor", "size", or "resolution");
        the appropriate one of the other parameters should then also be set to determine
        either the cell size multiplication factor, a desired output shape, or a desired
        output resolution.

        For the resolution method, a numeric value can be provided, or a string. A string
        must be one of "1k", "5k", or "10k" to set the resolution to 30 arcseconds, 2.5
        arcminutes, or 5 arcminutes respectively (assuming that the geotransform units are in
        degrees).

        Resolution and size methods may result in a different aspect ratio from the input, meaning
        that the cell shapes may be changed. Resolution method will always result in square cells.

        Returns a 4-tuple where :
        (
            inputGT, # because it may have been changed (sanitised) from what was passed in
            outputGT,
            (outYSize, outXSize),
            actualFactors
        )
        - first item is the geotransform of the input data (itself a 6-tuple,
        according to the GDAL standard) - because this can optionally be pre-snapped in case the input
        data were not properly mastergrid-aligned
        - second item is the geotransform of the output data
        - third item is the shape (a 2-tuple (ysize,zsixe), according to numpy standard)
        - fourth item is the aggregation factor that should be used in calculating the output cell to which
        a given input cell will be mapped.

        Note that the factor cannot be directly calculated by client code from the input and output sizes because
        if the input shape is 25 and the requested output shape (in size mode) is 5, BUT the output geotransform is
        snapped by (say) 0.5 cells towards the origin, then an additional output cell will be needed to cover the
        full input extent, i.e. the output will be reset to 6, but the aggregation should be run as if the factor was 5.

        Client code must be aware of the different origins of input and output geotransform that are possible due to
        snapping and calculating the output cell to which an input cell maps needs to take this, plus the true factor,
        into account.
        '''

        if not isinstance(inRasterProps, RasterProps):
            raise ValueError("GetAggregatedProperties must be called with a RasterProps object")

        # this can either be a string or an int or a float value, depending on the other settings
        aggSpec = self._aggregationSpec
        inputGT = inRasterProps.gt

        logMessage = self._Logger.logMessage
        # we most likely want to get the input gt clean first, it will make it a lot easier to figure out what is the
        # correct thing to do with the cleaning and snapping of the output gt
        if not self._AssumeCorrectInput:
            logMessage("Pre-snapping geotransform of input file {}".format(inputGT))
            inputGT = SnapAndAlignGeoTransform(inputGT, True, SnapTypes.NEAREST)

        # FIRST CALCULATE THE OUTPUT SIZE AND GEOTRANSFORM ASSUMING WITHOUT CHANGES TO ALIGNMENT, SUCH THAT
        # ORIGIN OF THE OUTPUT IS THE SAME AS THE ORIGIN OF THE INPUT
        if self._aggregationType == AggregationTypes.FACTOR:
            # Aggregate by an integer factor, like the ArcGIS Aggregate tool
            # Output pixel dimensions are just n times less than before and resolution is n time larger than before
            # If it's not a clean match then we will expand by one pixel to get everything
            # Implies the same pixel shape as before just larger
            # aggSpec is an int - checked in constructor
            outXSize = int(math.ceil(1.0 * inRasterProps.width / aggSpec))
            outYSize = int(math.ceil(1.0 * inRasterProps.height / aggSpec))
            if inRasterProps.width % aggSpec != 0 or inRasterProps.height % aggSpec != 0:
                # this will tend to trigger with irrational cell sizes such as 1/120 so we should probably
                # put a tolerance on it
                logMessage("warning, input size of {},{} (x,y) was not clean multiple of factor {}; ".format(
                    inRasterProps.width, inRasterProps.height, aggSpec) +
                           "output will have 1 cell greater extent", LogLevels.WARNING)
            outputGT_raw = (inputGT[0], inputGT[1] * aggSpec, 0.0,
                        inputGT[3], 0.0, inputGT[5] * aggSpec)
            actualFactors = (aggSpec, aggSpec)

        elif self._aggregationType == AggregationTypes.SIZE:
            # we will specify the required pixel dimensions of the output: aspect ratio may change
            # this is generally the best method to use if we're working with non-integer cell resolutions
            # aggSpec is a 2-tuple - checked in constructor
            outYSize, outXSize = aggSpec
            # now we need to calculate the resolution this implies (for a maintained extent)
            inputHeightPx = inRasterProps.height
            inputWidthPx = inRasterProps.width
            if 1.0 * inputHeightPx / inputWidthPx != 1.0 * outYSize / outXSize:
                # the coverage in ground extent will be the same, so the pixels themselves must change shape
                logMessage("warning, output size is different proportion to input, " +
                           "cells will change shape", LogLevels.WARNING)
            inputXMin = inputGT[0]
            inputYMax = inputGT[3]
            inputXMax = inputXMin + inputGT[1] * inputWidthPx
            # y resolution is negative as origin is top left
            inputYMin = inputYMax + inputGT[5] * inputHeightPx
            inputHeightProj = inputYMax - inputYMin
            inputWidthProj = inputXMax - inputXMin
            outputResX = inputWidthProj / outXSize
            outputResY = inputHeightProj / outYSize
            outputGT_raw = (inputGT[0], outputResX, 0.0, inputGT[3], 0.0, -outputResY)
            actualFactors = (inputHeightPx / outYSize, inputWidthPx / outXSize)

        elif self._aggregationType == AggregationTypes.RESOLUTION:
            inResX = inputGT[1]
            inResY = inputGT[5]
            inputHeightPx = inRasterProps.height
            inputWidthPx = inRasterProps.width
            xOrigin = inputGT[0]
            yOrigin = inputGT[3]
            if not inResY == -inResX:
                logMessage("warning, input had non-square cells, " +
                           "resolution mode creates square cells so they will change shape", LogLevels.WARNING)
            if isinstance(aggSpec, str):
                # use the hardcoded resolutions for the 3 main MAP resolutions to avoid
                # cocking about with irrational numbers not multiplying / dividing cleanly
                #inResX = SanitiseResolution(inResX)
                #inResY = -(SanitiseResolution(-inResY))
                xExtent = inputWidthPx * inResX
                yExtent = inputHeightPx * -inResY
                if aggSpec.lower().startswith("1k"):
                    outXSize = int(120 * xExtent)
                    outYSize = int(120 * yExtent)
                    outResolution = 1.0 / 120
                elif aggSpec.lower().startswith("5k"):
                    outXSize = int(24 * xExtent)
                    outYSize = int(24 * yExtent)
                    outResolution = 1.0 / 24
                elif aggSpec.lower().startswith("10k"):
                    outXSize = int(12 * xExtent)
                    outYSize = int(12 * yExtent)
                    outResolution = 1.0 / 12
                else:
                    logMessage("Unknown string resolution description!", LogLevels.ERROR)
                    assert False
            else:
                # specify a desired output resolution, implies the same in both directions (square pixels)
                outResolution = SanitiseResolution(aggSpec)
                if outResolution != aggSpec:
                    logMessage("Requested resolution of {} sanitised to {}".format(aggSpec, outResolution), LogLevels.DEBUG)
                xFactor = round(1.0 * outResolution / inputGT[1], 8)
                yFactor = round(-1.0 * outResolution / inputGT[5], 8)
                logMessage("Effective resolution-based factor is (x,y) {},{}".format(xFactor, yFactor), LogLevels.DEBUG)
                outX_exact = 1.0 * inputWidthPx / xFactor
                outY_exact = 1.0 * inputHeightPx / yFactor
                outXSize = math.ceil(outX_exact)
                outYSize = math.ceil(outY_exact)
                if outX_exact != outXSize or outY_exact != outYSize:
                    logMessage("warning, specified resolution was not clean multiple of input, " +
                               "output will have 1 cell greater extent" +
                               "(x,y {},{}) to (x,y {}{})".format(outXSize, outX_exact, outYSize, outY_exact), LogLevels.WARNING)
            actualFactors = (inputHeightPx / outYSize, inputWidthPx / outXSize)
            outputGT_raw = (inputGT[0], outResolution, 0.0, inputGT[3], 0.0, -outResolution)
        else:
            logMessage("Unknown aggregation type requested, valid values are 'size', 'resolution', 'factor'",
                       LogLevels.ERROR)
            assert False


        if self._SanitiseResolution or self._SnapType != SnapTypes.NONE:
            logMessage("Calculated raw output geotransform of {} which will now be snapped to the new resolution's global reference"
                       .format(outputGT_raw), LogLevels.DEBUG)
            outputGT = SnapAndAlignGeoTransform(outputGT_raw, fixResolution=self._SanitiseResolution, snapType=self._SnapType)
        else:
            outputGT = outputGT_raw


        inputXMin = inputGT[0]
        inputYMax = inputGT[3]
        inputXMax = inputXMin + inputGT[1] * inRasterProps.width
        # y resolution is negative as origin is top left
        inputYMin = inputYMax + inputGT[5] * inRasterProps.height
        outXMax = outputGT[0] + outputGT[1] * outXSize
        outYMin = outputGT[3] + outputGT[5] * outYSize
        if inputXMax > outXMax:
            logMessage("Aligned output being expanded one pixel in X dimension to allow for snapping shift", LogLevels.DEBUG)
            outXSize += 1
        if inputYMin < outYMin:
            logMessage("Aligned output being expanded one pixel in Y dimension to allow for snapping shift", LogLevels.DEBUG)
            outYSize += 1

        # because the output grid might have a different origin to the input we need to track the difference, for proper
        # calculation of which output cell each input should go into.
        # We also need to return the actual aggregation factor to use for passing to the core C code, because it isn't always
        # correct for the c code to just divide input size / output size, if the output has been expanded one pixel
        outputOriginOffsetYX = (inputGT[3] - outputGT[3], inputGT[0] - outputGT[0])
        return (
            inputGT, # because it may have been changed (sanitised) from what was passed in
            outputGT,
            (outYSize, outXSize),
            actualFactors
        )


    def RunAggregation(self):
        for f in self.filesList:
            self._aggregateFile(f)


    def _aggregateFile(self, filename):
        logMessage = self._Logger.logMessage
        logMessage("Processing file " + filename, LogLevels.INFO)
        # the input files could all be different, if needed
        inputProperties = GetRasterProperties(filename)
        npDataType = gdal_array.GDALTypeCodeToNumericTypeCode(inputProperties.datatype)
        inGT, outGT, outShape, aggFactors = self.GetAggregatedProperties(inputProperties)

        inOrigin_X, inRes_X, _, inOrigin_Y, _, inRes_Y = inGT
        outOrigin_X, outRes_X, _, outOrigin_Y, _, outRes_Y = outGT
        outputOriginCRSOffsetYX = (inOrigin_Y - outOrigin_Y, inOrigin_X - outOrigin_X)
        outputOriginPixOffset_Y = int(round(outputOriginCRSOffsetYX[0] / inRes_Y))
        outputOriginPixOffset_X = int(round(outputOriginCRSOffsetYX[1] / inRes_X))
        logMessage("Output shape will be {}(y,x) (input was {}, giving a factor of {}, {})".format(
            outShape, (inputProperties.height, inputProperties.width),
            1.0*inputProperties.height / outShape[0], 1.0*inputProperties.width / outShape[1]
        ), LogLevels.INFO)
        logMessage("Subpixel offset (input pixels from origin of output) is {}(y),{}(x)".format(
            outputOriginPixOffset_Y, outputOriginPixOffset_X),
            LogLevels.DEBUG)
        allDone = True
        for stat in self.stats:
            fnWillBe = os.path.join(self.outFolder, self._fnGetter(os.path.basename(filename), stat))
            if not os.path.exists(fnWillBe):
                allDone = False
        if allDone:
            logMessage("All outputs for {0!s} already exist, skipping!".format(os.path.basename(filename)), LogLevels.INFO)
            return

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
        # print ("estimated GB: "+str(estMem))

        # and actually just ignore all the non squareness of the world and stuff and use the larger dim
        tileMax = max(tileH, tileW)

        tiles = getTiles(inputProperties.width, inputProperties.height, tileMax)
        logMessage("Processing in {0!s} tiles of {1!s} pixels".format(len(tiles), tileMax), LogLevels.INFO)
        if inputProperties.ndv is  None:
            logMessage("No NDV defined in input, assuming -np.inf", LogLevels.DEBUG)
            ndv_In = -np.inf
        elif np.isnan(inputProperties.ndv):
            logMessage("Incoming nodata value is " + str(inputProperties.ndv) + " - resetting to -np.inf for compatibility",
                       LogLevels.DEBUG)
            ndv_In = -np.inf
        else:
            ndv_In = inputProperties.ndv
            logMessage("Incoming nodata value is " + str(ndv_In), LogLevels.DEBUG)
        if self._mode == AggregationModes.CONTINUOUS:
            aggregator = Continuous_Aggregator_Flt(inputProperties.height, inputProperties.width,
                                               outShape[0], outShape[1],
                                                outputOriginPixOffset_Y, outputOriginPixOffset_X,
                                                aggFactors[0], aggFactors[1],
                                               self.ndvOut,
                                               self.stats)
        else:
            doLikeAdjacency = catstats.LIKEADJACENCIES in self.stats
            if self.categories is not None:
                catArg = self.categories
            else:
                catArg = self.nCategories
            aggregator = Categorical_Aggregator(inputProperties.height, inputProperties.width,
                                                outShape[0], outShape[1],
                                                outputOriginPixOffset_Y, outputOriginPixOffset_X,
                                                aggFactors[0], aggFactors[1],
                                                catArg,
                                                doLikeAdjacency,
                                                self.ndvOut, ndv_In
                                            )
        # for each tile read the data and add to the aggregator
        # todo (maybe) don't bother adding complete nodata tiles? (just add fake one)
        for tile in tiles:
            logMessage("Processing tile {}".format(tile), level=LogLevels.DEBUG)
            xOff = tile[0][0]
            yOff = tile[1][0]
            xSize = tile[0][1] - xOff
            ySize = tile[1][1] - yOff

            inArr, thisGT, thisProj, thisNdv = ReadAOI_PixelLims(filename,
                                                                 (xOff, xOff+xSize),
                                                                 (yOff, yOff+ySize))
            #inArr, x1, x2, x3 = ReadAOI_PixelLims(filename,
            #                                        (xOff, xOff+xSize),
            #                                        (yOff, yOff+ySize))
            # todo check the actual datatypes of the files here esp for categorical
            if self._mode == AggregationModes.CONTINUOUS:
                if (np.isnan(thisNdv)):
                    # C does not have a nan so recode to something else
                    thisNdv = -np.inf
                    inArr[np.isnan(inArr)] = thisNdv
                aggregator.addTile(inArr.astype(np.float32), xOff, yOff, ndv_In)
            else:
                aggregator.addTile(inArr#.astype(np.uint8)
                                   , xOff, yOff)
            inArr = None

        # all tiles added, get results
        r = aggregator.GetResults()
        for stat in self.stats:
            logMessage("Saving outputs: " + stat.value, LogLevels.INFO)
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
                    SaveLZWTiff(r[stat], ndv_In, outGT, inputProperties.proj,
                                self.outFolder, fnOut)
                else:
                    for i in range(0, self.nCategories):
                        outValue = r['valuemap'][i]
                        fnOut = self._fnGetter(os.path.basename(filename),
                                               stat,
                                               outValue)
                        SaveLZWTiff(r[stat][i], self.ndvOut, outGT, inputProperties.proj,
                                         self.outFolder, fnOut)







