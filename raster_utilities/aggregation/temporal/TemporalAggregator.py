import os
import subprocess

from raster_utilities.aggregation.temporal.core.temporal import TemporalAggregator_Dynamic
from ..aggregation_values import TemporalAggregationStats as tempstats
from ...io.tiff_management import GetRasterProperties, ReadAOI_PixelLims, SaveLZWTiff
from ...utils.logger import MessageLogger, LogLevels
from ...utils.raster_tiling import getTiles
import numpy as np
from collections import defaultdict

class TemporalAggregator:
    def __init__(self, filesDict, outFolder, outputNDV, stats, doSynoptic, bytesLimit=40e9, loggingLevel = LogLevels.INFO):
        '''Runs temporal aggregation across a supplied dictionary of filenames

        The aggregation doesn't strictly need to be temporal, rather this is about summarising
        a number of rasters with identical shape into a single file of the same shape.

        The input dictionary should have keys that define the output filename, with values that are lists of filenames
        contributing to that output file. Specifically the keys should be the output filename with a '*' character that will
        be replaced by the name of the aggregation statistic in each file, e.g.:
            filesDict = {'MyVariableName.2000.Annual.*.1km.Data': [list of all files in year 2000]}
        Client code can therefore generate the list of files by calculating what the key will be then using it as a
        glob pattern to get a list of all matching files.

        All of the input files (all the members of the all the values for each key) should have the same
        dimensions and geotransform. This requirement is so that synoptic outputs can be calculated on the fly if required
        e.g. the mean etc of all files present in any of the keys.

        outputNDV doesn't have to match the incoming NDV

        stats is a list containing some or all of the values of the TemporalAggregationStats enumeration, or strings
        ["min", "max", "mean", "sd", "sum", "count"]
        The more statistics are specified, the more memory is required.

        doSynoptic specifies whether "overall" statistics should be calculated in addition to one
        per timestep - this doubles memory use, but means you don't have to calculate synoptic results separately by
        passing in a synoptic key with all files listed, thus speeding things up.
        This has no effect if filesDict only has one item. The synoptic calculation will only count each input file
        once, regardless of how many times it appears (e.g. it may be used in both annual and monthly summaries but
        will only count once towards the synoptic output)
        '''

        assert isinstance(filesDict, dict)
        self.filesDict = filesDict
        assert isinstance(outFolder, str)
        self.outFolder = outFolder
        self._tileFolder = os.path.join(outFolder, "aggregation_tiles")
        self._outputFilesTiles = defaultdict(list)
        self.outputNDV = outputNDV
        self._bytesLimit = bytesLimit
        self._logger = MessageLogger(loggingLevel)
        assert isinstance(stats, list)
        mystats = []
        for s in stats:
            if isinstance(s, tempstats):
                # the ALL property defines all the ones that we can produce through aggregation code
                # (the enum includes others like "DATA" that we don't handle here)
                assert s.value in tempstats.ALL.value
                mystats.append(s)
            else:
                try:
                    # see if it's a string representation of the stat, which it will be if the client has
                    # actually passed in tempstats.ALL
                    sCast = tempstats(s)
                    mystats.append(sCast)
                except ValueError:
                    raise

        assert (all([s in tempstats.ALL.value or isinstance(s, tempstats)
                     for s in stats]))
        self.stats = [tempstats(s) for s in stats]

        self.doSynoptic = doSynoptic == True
        self.synopticFileSet = set()
        aFilename = list(self.filesDict.items())[0][1][0]
        props = GetRasterProperties(aFilename)
        self.InputProperties = props
        self._logger.logMessage("Initialised aggregator with properties (geotranform and height) from {}".format(aFilename),
                                LogLevels.INFO)
        self._logger.logMessage("Properties are {}".format(props), LogLevels.DEBUG)

    def _timePoints(self):
        return sorted(self.filesDict.keys())

    def _fnGetter(self, outnameTemplate, temporalStat, sliceIndex=-1):
        temporalStatname = temporalStat.value
        if outnameTemplate.find("*") == -1:
            raise ValueError("{} does not match the expected format of a string with a single '*' character")
        outName = outnameTemplate.replace('*', temporalStatname)
        if sliceIndex != -1:
            outName = outName + ".{}".format(sliceIndex)
        outName = outName + ".tif"
        return outName

    def _fnGetter_MG(self, filename, newDatePart, temporalStat, where):
        outNameTemplate = "{0!s}.{1!s}.{2!s}.{3!s}.{4!s}.tif"
        temporalStatname = temporalStat.value
        existingParts = os.path.basename(filename).split(".")
        if len(existingParts)==7:
            varTag = existingParts[0]
            resTag = existingParts[4]
            spatialStat = existingParts[5]
            if where != -1:
                spatialStat = spatialStat + "." + str(where)
            outname = outNameTemplate.format(varTag, newDatePart, temporalStatname, resTag, spatialStat)
            return outname
        else:
            raise ValueError("does not appear to be a valid 6-token filename")

    def _fnGetter_Safe(self, whatandwhen, temporalStat, where):
        statname = temporalStat.value
        if where == -1:
            return ".".join([str(whatandwhen), statname,
                             str(self.InputProperties.res), "tif"])
        else:
            return ".".join([str(whatandwhen), statname,
                             str(self.InputProperties.res), str(where), "tif"])

    def _estimateTemporalAggregationMemory(self, height):
        nPix = height * self.InputProperties.width
        bpp = {tempstats.COUNT: 2, tempstats.MEAN: 16, tempstats.SD: 16,
               tempstats.MIN: 4, tempstats.MAX: 4, tempstats.SUM: 4}
        try:
            bppTot = sum([bpp[s] for s in self.stats])
        except KeyError:
            raise KeyError("Invalid statistic specified! Valid items are " + str(bpp.keys()))

        # calculating sd requires calculating mean anyway
        if ((tempstats.SD in self.stats) and (tempstats.MEAN not in self.stats)):
            bppTot += bpp[tempstats.MEAN]
        bTot = bppTot * nPix
        if self.doSynoptic:
            bTot *= 2
        bTot += 8 * nPix # the input data tile
        return bTot

    def RunAggregation(self, deleteIntermediateTiles=False):
        '''For each key in filesDict, aggregates the files on the associated value to the specified stats.

        The input files and the outputs they contribute to have been set in the constructor, see comments there.

        If the aggregation is too large to be done in memory (according to a very inaccurate estimation), it will
        be done in multiple tiles and then an attempt will be made to mosaic these tiles afterwards.

        All of the files (across all output time points) must have the same geotransform and extent.
        '''

        self._outputFilesTiles = defaultdict(list)
        w = self.InputProperties.width
        h = self.InputProperties.height
        runHeight = h
        bytesFull = self._estimateTemporalAggregationMemory(runHeight)
        while bytesFull > self._bytesLimit:
            runHeight = runHeight // 2 # force integer division on python 2.x
            bytesFull = self._estimateTemporalAggregationMemory(runHeight)
        slices = sorted(list(set([s[1] for s in getTiles(w, h, runHeight)])))
        isFullFile = len(slices) == 1
        if isFullFile:
            saveFolder = self.outFolder
            self._logger.logMessage("Running entire extent in one pass", LogLevels.INFO)
        else:
            saveFolder = self._tileFolder
            if len(slices)>1:
                self._logger.logMessage("Running by splitting across {0!s} tiles".format(len(slices)), LogLevels.INFO)

        for t, b in slices:
            self._temporalAggregationSliceRunner(t, b, saveFolder)

        if len(slices) > 1:
            self._stitchTiles(deleteIntermediateTiles)

        self._logger.logMessage("All done!", LogLevels.INFO)
        if len(slices) > 1 and not deleteIntermediateTiles:
            self._logger.logMessage("You can delete the tile files from the aggregation_tiles subfolder", LogLevels.INFO)

    def _stitchTiles(self, deleteTiles=False):
        ''' Attempts to mosaic intermediate tiles of aggregated output into complete files.
        Relies on gdal binaries (gdalbuildvrt, gdal_translate, gdaladdo, and gdalinfo) being present in the path
        :param deleteTiles: bool, should the intermediate files be deleted after mosaicing?
        :return:
        '''
        vrtBuilder = "gdalbuildvrt {0} {1}"
        transBuilder = "gdal_translate -of GTiff -co COMPRESS=LZW " + \
                       "-co PREDICTOR=2 -co TILED=YES -co BIGTIFF=YES " + \
                       "-co NUM_THREADS=ALL_CPUS --config GDAL_CACHEMAX 8000 {0} {1}"
        ovBuilder = "gdaladdo -ro --config COMPRESS_OVERVIEW LZW --config USE_RRD NO " + \
                    "--config TILED YES --config GDAL_CACHEMAX 8000 {0} 2 4 8 16 32 64 128 256 "
        statBuilder = "gdalinfo -stats {0}"
        vrts = []
        tifs = []

        for outname, tiles in self._outputFilesTiles.items():
            vrtFile = outname.replace(".tif", ".vrt")
            vrtPath = os.path.join(self.outFolder, vrtFile)
            vrtCommand = vrtBuilder.format(vrtPath, " ".join(tiles))
            self._logger.logMessage("Building vrt " + vrtFile, LogLevels.INFO)
            self._logger.logMessage(vrtCommand, LogLevels.DEBUG)
            vrts.append(vrtPath)
            subprocess.call(vrtCommand)

        for vrt in vrts:
            tif = vrt.replace("vrt", "tif")
            translateCommand = transBuilder.format(vrt, tif)
            self._logger.logMessage("Translating to output files {0!s}".format(tif), LogLevels.INFO)
            self._logger.logMessage(translateCommand, LogLevels.DEBUG)
            tifs.append(tif)
            subprocess.call(translateCommand)

        if deleteTiles:
            for _, tiles in self._outputFilesTiles.items():
                for tile in tiles:
                    os.remove(tile)
            for vrt in vrts:
                os.remove(vrt)

        # Build overviews and statistics on all of the output tiffs
        for tif in tifs:
            ovCommand = ovBuilder.format(tif)
            statCommand = statBuilder.format(tif)
            self._logger.logMessage("Building overviews and stats on {0!s}".format(tif), LogLevels.INFO)
            self._logger.logMessage(ovCommand, LogLevels.DEBUG)
            self._logger.logMessage(statCommand, LogLevels.DEBUG)
            subprocess.call(ovCommand)
            subprocess.check_output(statCommand)

    def _temporalAggregationSliceRunner(self,  top, bottom, outFolder):
        '''Runs temporal aggregation across a set of files, for a specified horizontal slice

        Not intended to be called by client code - use RunAggregation()

        top and bottom allow specification of a subset (horizontal slice) of the files to run the
        aggregation for, in case there isn't enough memory to do the whole extent in one go. In this
        case output files will have a suffix indicating the top pixel coordinate relative to the incoming
        files.
        '''

        logMessage = self._logger.logMessage("Beginning slice at top: {} , bottom: {}".format(top, bottom),
                                             LogLevels.DEBUG)
        if not ((bottom > top ) and top >= 0):
            raise ValueError("bottom must be greater than top, and top must be >= zero " +
                             "(because raster origin of 0,0 is the top-left corner)")
        sliceHeight = bottom - top

        runSynoptic = (len(self.filesDict.keys()) > 1) and self.doSynoptic

        bytesTot = self._estimateTemporalAggregationMemory(sliceHeight)
        gb = bytesTot / 2e30
        if gb > 30:
            logMessage("Requires more than 30GB, are you sure this is wise....", LogLevels.WARNING)

        statsCalculator = TemporalAggregator_Dynamic(sliceHeight, self.InputProperties.width,
                                                     self.outputNDV, self.stats, runSynoptic)
        sliceGT = None
        sliceProj = None
        isFullFile = sliceHeight == self.InputProperties.height
        if isFullFile:
            where = -1 # flag to exclude slice id from filenames
        else:
            where = top

        for timeKey, timeFiles in self.filesDict.items():

            self._logger.logMessage(timeKey, LogLevels.DEBUG)

            allDone = True

            for stat in self.stats:
                fnWillBe = self._fnGetter(timeKey, stat)
                if not os.path.exists(fnWillBe):
                    allDone = False
            if allDone:
                self._logger.logMessage("All outputs for {0!s} already exist, skipping!".format(timeKey),
                                        LogLevels.WARNING)
                continue

            for timeFile in timeFiles:
                self._logger.logMessage("Reading {}".format(timeFile), LogLevels.DEBUG)
                data, thisGT, thisProj, thisNdv = ReadAOI_PixelLims(timeFile, None, (top, bottom))
                if data.dtype != 'float32':
                    self._logger.logMessage("Data will be cast to float32, precision or data may be lost",
                                            LogLevels.WARNING)
                    if thisNdv < np.finfo(np.float32).min or thisNdv > np.finfo(np.float32).max:
                        self._logger.logMessage("Nodata value outside of range of float32, will be reset to -9999",
                                                LogLevels.WARNING)
                        data[data==thisNdv] = -9999
                        thisNdv = -9999
                    data = data.astype('float32')
                if sliceGT is None:
                    # first file
                    sliceGT = thisGT
                    sliceProj = thisProj
                else:
                    if sliceGT != thisGT or sliceProj != thisProj:
                        raise ValueError("File " + timeFile +
                                         " has a different geotransform or projection - cannot continue!")
                # pass the filename as a tag so the aggregator can avoid using the same file
                # multiple times in calculating overall values. E.g. we might be calculating annual
                # and, separately, monthly means, involving reading each file twice, but only want
                # to count each one once towards overall synoptic totals if they are also being tracked.
                statsCalculator.addFile(data, thisNdv, timeFile)
            periodResults = statsCalculator.emitStep()
            for stat in self.stats:
                tileFNWillBe = self._fnGetter(timeKey, stat, where)
                fullFnWillBe = self._fnGetter(timeKey, stat, -1)
                if not os.path.exists(tileFNWillBe):
                    SaveLZWTiff(periodResults[stat], self.outputNDV, sliceGT, sliceProj, outFolder,
                            tileFNWillBe)
                    self._outputFilesTiles[fullFnWillBe].append(os.path.join(outFolder, tileFNWillBe))
        if runSynoptic:
            overallResults = statsCalculator.emitTotal()
            if isFullFile:
                where = -1
            else:
                where = top
            timeKey = "Synoptic.Overall"
            for stat in self.stats:
                tileFNWillBe = self._fnGetter(timeKey, stat, where)
                fullFnWillBe = self._fnGetter(timeKey, stat, -1)

                SaveLZWTiff(overallResults[stat], self.outputNDV, sliceGT, sliceProj, outFolder,
                            os.path.basename(tileFNWillBe))
                self._outputFilesTiles[fullFnWillBe].append(tileFNWillBe)

