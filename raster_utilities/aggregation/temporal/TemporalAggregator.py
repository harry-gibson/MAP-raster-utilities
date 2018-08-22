import os
import subprocess

from raster_utilities.aggregation.temporal.core.temporal import TemporalAggregator_Dynamic
from ..aggregation_values import TemporalAggregationStats as tempstats
from ...io.tiff_management import GetRasterProperties, ReadAOI_PixelLims, SaveLZWTiff
from ...utils.logger import logMessage
from ...utils.raster_tiling import getTiles


class TemporalAggregator:
    def __init__(self, filesDict, outFolder, outputNDV, stats, doSynoptic, bytesLimit=40e9):
        assert isinstance(filesDict, dict)
        self.filesDict = filesDict
        assert isinstance(outFolder, str)
        self.outFolder = outFolder
        self._tileFolder = os.path.join(outFolder, "aggregation_tiles")
        self.outputNDV = outputNDV
        self._bytesLimit = bytesLimit

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
        aFilename = self.filesDict.iteritems().next()[1][0]
        props = GetRasterProperties(aFilename)
        self.InputProperties = props

    def _timePoints(self):
        return sorted(self.filesDict.keys())

    def _fnGetter(self, whatandwhen, stat, where):
        statname = stat.value
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

    def RunAggregation(self):
        '''For each key in filesDict, aggregates the files on the associated value to the specified stats.

        Returns true if the aggregation needed to be done in multiple tiles for memory reasons in
        which case you will need to mosaic the out  put tiles afterwards.

        Files should be provided in filesDict; each item should be a key representing an
        output point (timespan) and a value representing the file paths for that point in time.
        The keys could be months (e.g. "2001-02"), years (e.g. "2002"), synoptic months
        (e.g. "February"), or anything else. If there is only one item then only synoptic outputs
        can be generated. The string version of the key will be used to generate the output filenames
        in the form "StringKey.Stat.Resolution(.TileId).tif"

        All of the files (across all output time points) must have the same geotransform and extent.

        top and bottom allow specification of a subset (horizontal slice) of the files to run the
        aggregation for, in case there isn't enough memory to do the whole extent in one go. In this
        case output files will have a suffix indicating the top pixel coordinate relative to the incoming
        files.

        width must match the overall width of the incoming files (vertical slicing isn't supported)

        outputNDV doesn't have to match the incoming NDV

        stats is a list containing some or all of the values from aggregation_values.TemporalAggregationStats

        The more statistics are specified, the more memory is required.

        doSynoptic specifies whether "overall" statistics should be calculated in addition to one
        per timestep - this doubles memory use. This has no effect if filesDict only has one item.
        '''

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
            logMessage("Running entire extent in one pass")
        else:
            saveFolder = self._tileFolder
            if len(slices)>1:
                logMessage("Running by splitting across {0!s} tiles".format(len(slices)))

        for t, b in slices:
            self._temporalAggregationSliceRunner(t, b, saveFolder)

        if len(slices) > 1:
            self._stitchTiles()
        logMessage("All done!")
        if len(slices) > 1:
            logMessage("You can delete the tile files from the aggregation_tiles subfolder")

    def _stitchTiles(self):
        vrtBuilder = "gdalbuildvrt {0} {1}"
        transBuilder = "gdal_translate -of GTiff -co COMPRESS=LZW " + \
                       "-co PREDICTOR=2 -co TILED=YES -co SPARSE_OK=TRUE -co BIGTIFF=YES " + \
                       "-co NUM_THREADS=ALL_CPUS --config GDAL_CACHEMAX 8000 {0} {1}"
        ovBuilder = "gdaladdo -ro --config COMPRESS_OVERVIEW LZW --config USE_RRD NO " + \
                    "--config TILED YES {0} 2 4 8 16 32 64 128 256 --config GDAL_CACHEMAX 8000"
        statBuilder = "gdalinfo -stats {0}>nul"
        vrts = []
        tifs = []
        for stat in self.stats:
            statname = stat.value
            timeKeys = self._timePoints()
            ranSynoptic = (len(self.filesDict.keys()) > 1) and self.doSynoptic
            if ranSynoptic and ("Overall" not in timeKeys):
                timeKeys.append("Overall")
            for timeKey in timeKeys:
                tiffWildCard = self._fnGetter(str(timeKey), stat,  "*")
                sliceTiffs = os.path.join(self._tileFolder, tiffWildCard)
                vrtName = timeKey + "." + statname + ".vrt"
                vrtFile = os.path.join(self.outFolder, vrtName)
                vrtCommand = vrtBuilder.format(vrtFile, sliceTiffs)
                logMessage("Building vrt " + vrtFile)
                vrts.append(vrtFile)
                subprocess.call(vrtCommand)


        for vrt in vrts:
            tif = vrt.replace("vrt", "tif")
            translateCommand = transBuilder.format(vrt, tif)
            logMessage("Translating to output files {0!s}".format(tif))
            tifs.append(tif)
            subprocess.call(translateCommand)
        # Build overviews and statistics on all of the output tiffs
        for tif in tifs:
            ovCommand = ovBuilder.format(tif)
            statCommand = statBuilder.format(tif)
            subprocess.call(ovCommand)
            subprocess.check_output(statCommand)

    def _temporalAggregationSliceRunner(self,  top, bottom, outFolder):
        '''Runs temporal aggregation across a set of files.

        Files should be provided in filesDict; each item should be a key representing an
        output point (timespan) and a value representing the file paths for that point in time.
        The keys could be months (e.g. "2001-02"), years (e.g. "2002"), synoptic months
        (e.g. "February"), or anything else. If there is only one item then only synoptic outputs
        can be generated.

        All of the files (across all output time points) must have the same geotransform and extent.

        top and bottom allow specification of a subset (horizontal slice) of the files to run the
        aggregation for, in case there isn't enough memory to do the whole extent in one go. In this
        case output files will have a suffix indicating the top pixel coordinate relative to the incoming
        files.

        width must match the overall width of the incoming files (vertical slicing isn't supported)

        outputNDV doesn't have to match the incoming NDV

        stats is a list containing some or all of
        ["min", "max", "mean", "sd", "sum", "count"]
        The more statistics are specified, the more memory is required.

        doSynoptic specifies whether "overall" statistics should be calculated in addition to one
        per timestep - this doubles memory use. This has no effect if filesDict only has one item.
        '''

        #if not (isinstance(bottom, int) and isinstance(top, int)):
         #   raise TypeError("top and bottom must be integer values")
        if not ((bottom > top ) and top >= 0):
            raise ValueError("bottom must be greater than top and top must be GTE zero " +
                             "(raster origin is the top-left corner)")
        sliceHeight = bottom - top
        runSynoptic = (len(self.filesDict.keys()) > 1) and self.doSynoptic

        bytesTot = self._estimateTemporalAggregationMemory(sliceHeight)
        gb = bytesTot / 2e30
        if gb > 30:
            logMessage("Requires more than 30GB, are you sure this is wise....")

        statsCalculator = TemporalAggregator_Dynamic(sliceHeight, self.InputProperties.width,
                                                     self.outputNDV, self.stats, runSynoptic)
        sliceGT = None
        sliceProj = None
        isFullFile = sliceHeight == self.InputProperties.height
        if isFullFile:
            where = -1 # flag to exclude slice id from filenames
        else:
            where = top

        for timeKey, timeFiles in self.filesDict.iteritems():
            #for stat in self.stats:
            #    filenameWillBe = self._fnGetter(str(timeKey), stat, where)

            logMessage(timeKey)

            allDone = True
            for stat in self.stats:
                fnWillBe = os.path.join(outFolder, self._fnGetter(str(timeKey), stat, where))
                if not os.path.exists(fnWillBe):
                    allDone = False
            if allDone:
                logMessage("All outputs for {0!s} already exist, skipping!".format(timeKey))
                continue

            for timeFile in timeFiles:
                data, thisGT, thisProj, thisNdv = ReadAOI_PixelLims(timeFile, None, (top, bottom))
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
                # to count each one once towards overall totals.
                statsCalculator.addFile(data, thisNdv, timeFile)
            periodResults = statsCalculator.emitStep()
            for stat in self.stats:
                fnWillBe = os.path.join(outFolder, self._fnGetter(str(timeKey), stat, where))
                if not os.path.exists(fnWillBe):
                    SaveLZWTiff(periodResults[stat], self.outputNDV, sliceGT, sliceProj, outFolder,
                            self._fnGetter(str(timeKey), stat, where))
        if runSynoptic:
            overallResults = statsCalculator.emitTotal()
            if isFullFile:
                where = -1
            else:
                where = top
            for stat in self.stats:
                SaveLZWTiff(overallResults[stat], self.outputNDV, sliceGT, sliceProj, outFolder,
                            self._fnGetter("Overall", stat,  where))




