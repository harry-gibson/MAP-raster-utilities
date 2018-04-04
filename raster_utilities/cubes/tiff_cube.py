import os, glob
from datetime import date
from collections import  defaultdict
from cube_constants import  CubeResolutions, CubeLevels
from ..aggregation.aggregation_values import TemporalAggregationStats, ContinuousAggregationStats
from ..io.TiffFile import SingleBandTiffFile

class TiffCube:
    ''' Represents a view of a MAP mastergrids data cube folder for a single variable

    Based on a folder structure documented elsewhere. In summary the given Masterfolder should contain
    subfolders for resolution: any / all of "1k", "5k", "10k". Each of these should contain subfolders named
    any / all of "Monthly", "Annual", "Synoptic". Files within each of these folders should have base filenames
    with six (or more) dot-delimited tokens, with the following meanings:

    {VariableName}.{Year}.{Month}.{TemporalSummary}.{Res}.{SpatialSummary}.tif
    - The second token can be the string "Synoptic" instead of a year.
    - The third token can be a month number, or the string "Overall" iif "Synoptic" is in pos 2.
    - The fourth token can be a value from TemporalAggregationStats, although only "mean" is generally available
    - The sixth token can be a value from ContinuousAggregationStats, e.g. "mean", "max", "min"...

    {
        resolution (1km / 5km):
            {
                temporal

            }
    }
    '''

    def __init__(self, MasterFolder,
                 resolution = CubeResolutions.ONE_K,
                 temporalsummary = TemporalAggregationStats.MEAN,
                 spatialsummary = ContinuousAggregationStats.MEAN,
                 allowGlobalCache = True):
        self.MasterFolder = MasterFolder
        self.VariableName = ""
        self.MonthlyDictionary = {}
        self.AnnualDictionary = {}
        self.SynopticDictionary = {}
        self.__HasMonthly = False
        self.__HasAnnual = False
        self.__HasSynoptic = False
        self.__HasStatic = False

        self.__DataCache = {"Filename": None, "FileObject": None}
        self.__CanCacheData = allowGlobalCache

        self.VariableName = os.path.basename(MasterFolder)
        self.__InitialiseFiles(resolution, temporalsummary.value, spatialsummary.value)

    def log(self, msg):
        print(msg)

    def __InitialiseFiles(self, resolution, temporalsummary, spatialsummary):
        ''' Parse all the monthly, annual, and synoptic files for this cube

        Selects the files from a mastergrid cube folder structure corresponding
        to the specified resolution, temporal summary (how derived from 8-daily
        source) and spatial summary (how aggregated to 5k or 10k, if at all), and
        parses them into dictionaries keyed by time

        :param resolution:
        :param temporalsummary:
        :param spatialsummary:
        :return:
        '''
        # the constants are defined as numeric values, maybe could have done them as these strings
        if resolution == CubeResolutions.ONE_K:
            res = "1km"
        elif resolution == CubeResolutions.FIVE_K:
            res = "5km"
        elif resolution == CubeResolutions.TEN_K:
            res = "10km"
        else:
            raise ValueError("resoluion of " + resolution + " not recognised!")

        # this cube object represents one of the (maybe) available resolutions
        resFolderPath = os.path.join(self.MasterFolder, res)
        filenameTemplate = "{0!s}.{1!s}.{2!s}.{3!s}.{4!s}.{5!s}*.tif"

        # this cube object can reflect various temporal summaries of the data, depending on what's avail.
        # Parse all that's available from the various cube levels

        # if a monthly folder is present store all matching files against a dictionary keyed by a
        # date object (first of the month in question)
        monthlyFolderPath = os.path.join(resFolderPath, CubeLevels.MONTHLY.value)
        if os.path.isdir(monthlyFolderPath):
            monthlyWildcard = filenameTemplate.format(self.VariableName,
                                                      "*", "*",
                                                      temporalsummary, res, spatialsummary)
            monthlyWildcardPath = os.path.join(monthlyFolderPath, monthlyWildcard)
            monthlyFiles = glob.glob(monthlyWildcardPath)
            if len(monthlyFiles) == 0:
                self.log('No monthly files found with wildcard ' + monthlyWildcardPath)
            for f in monthlyFiles:
                (variablename, yearOrSynoptic, monthOrOverall, temporalType,
                 resolution, spatialType) = self._TryParseCubeFilename(f)
                if self.__isInt(yearOrSynoptic) and self.__isInt(monthOrOverall):
                    filedate = date(int(yearOrSynoptic), int(monthOrOverall), 1)
                    self.MonthlyDictionary[filedate] = f
                    self.__HasMonthly = True
                else:
                    self.log("File {0!s} is in monthly folder but filename does not match format"
                             .format(f))
        else:
            self.log("Monthly folder path of {0!s} not found".format(monthlyFolderPath))

        # if an annual folder is present store all matching files against a dictionary keyed by an
        # int - the year in question
        annualFolderPath = os.path.join(resFolderPath, CubeLevels.ANNUAL.value)
        if os.path.isdir(annualFolderPath):
            annualWildcard = filenameTemplate.format(self.VariableName,
                                                     "*", "Annual",
                                                     temporalsummary, res, spatialsummary)
            annualWildcardPath = os.path.join(annualFolderPath, annualWildcard)
            annualFiles = glob.glob(annualWildcardPath)
            if len(annualFiles) == 0:
                self.log('No annual files found with wildcard ' + annualWildcardPath)
            for f in annualFiles:
                (variablename, yearOrSynoptic, monthOrOverall, temporalType,
                 resolution, spatialType) = self._TryParseCubeFilename(f)
                if self.__isInt(yearOrSynoptic) and monthOrOverall == "Annual":
                    self.AnnualDictionary[int(yearOrSynoptic)] = f
                    self.__HasAnnual = True
                else:
                    self.log("File {0!s} is in annual folder but filename does not match format"
                             .format(f))
        else:
            self.log("Annual folder path of {0!s} not found".format(annualFolderPath))

        # if a synoptic folder is present store all matching files against a dictionary keyed by a
        # string - the 2-digits month number or the constant "Overall"
        synopticFolderPath = os.path.join(resFolderPath, CubeLevels.SYNOPTIC.value)
        if os.path.isdir(synopticFolderPath):
            synopticWildcard = filenameTemplate.format(self.VariableName,
                                                       "Synoptic", "*",
                                                       temporalsummary, res, spatialsummary)
            synopticWildcardPath = os.path.join(synopticFolderPath, synopticWildcard)
            synopticFiles = glob.glob(synopticWildcardPath)
            if len(synopticFiles) == 0:
                self.log('No synoptic files found with wildcard ' + synopticWildcardPath)

            for f in synopticFiles:
                (variablename, yearOrSynoptic, monthOrOverall, temporalType,
                 resolution, spatialType) = self._TryParseCubeFilename(f)
                if yearOrSynoptic == "Synoptic":
                    self.SynopticDictionary[monthOrOverall.zfill(2)] = f
                    self.__HasSynoptic = True
                else:
                    self.log("File {0!s} is in synoptic folder but filename does not match format"
                             .format(f))
        else:
            self.log("Synoptic folder path of {0!s} not found".format(synopticFolderPath))

        staticFolderPath = os.path.join(resFolderPath, CubeLevels.STATIC.value)
        if os.path.isdir(staticFolderPath):
            # static filenames should be "X.Synoptic.Overall.Data.res.spatial"
            staticWildcard = filenameTemplate.format(self.VariableName,
                                                     "Synoptic", "Overall",
                                                     "Data", res, spatialsummary)
            staticWildcardPath = os.path.join(staticFolderPath, staticWildcard)
            staticFiles = glob.glob(staticWildcardPath)
            if len(staticFiles) == 0:
                self.log('No static files found with wildcard ' + staticWildcardPath)
                if not (self.__HasAnnual or self.__HasMonthly or self.__HasSynoptic):
                    self.log('No matching files of any type have been located: aborting')
                    assert False
            elif len(staticFiles > 0):
                self.log('More than one static file found with wildcard ' + staticWildcardPath)
                assert False
            elif (self.__HasAnnual or self.__HasMonthly or self.__HasSynoptic):
                self.log('Found a static file as well as temporal ones: not supported! Skipping static file')
            else:
                self.__StaticFilename = staticFiles[0]
                self.__HasStatic = True


    def _TryParseCubeFilename(self, filePath):
        filename = os.path.basename(filePath)
        filetokens = filename.split('.')
        if len(filetokens) < 6:
            raise Exception ("does not appear to be a valid mastergrid tokenised filename")
        variablename, yearOrSynoptic, monthOrOverall, temporalType, resolution, spatialType = filetokens[0:6]
        if self.VariableName == "":
            self.VariableName = variablename
        elif self.VariableName != variablename:
            raise ValueError("Filename {0!s} has variable inconsistent with {1!s}".format(filename, self.VariableName))
        return variablename, yearOrSynoptic, monthOrOverall, temporalType, resolution, spatialType

    def CubeLevelIsAvailable(self, CubeLevel):
        assert isinstance (CubeLevel, CubeLevels)
        if CubeLevel == CubeLevels.MONTHLY:
            return self.__HasMonthly
        elif CubeLevel == CubeLevels.ANNUAL:
            return self.__HasAnnual
        elif CubeLevel == CubeLevels.SYNOPTIC:
            return self.__HasSynoptic
        elif CubeLevel == CubeLevels.STATIC:
            return self.__HasStatic
        else:
            return False

    def ReadMonthlyDataForDate(self, RequiredDate, latLims=None, lonLims=None):
        return self.ReadDataForDate(CubeLevels.MONTHLY, RequiredDate, latLims, lonLims)

    def ReadAnnualDataForDate(self, RequiredDate, latLims=None, lonLims=None):
        return self.ReadDataForDate(CubeLevels.ANNUAL, RequiredDate, latLims, lonLims)

    def ReadSynopticDataForDate(self, RequiredDate, latLims=None, lonLims=None):
        '''RequiredDate may be None for overall synoptic, or a date for the matching synoptic month'''
        return self.ReadDataForDate(CubeLevels.SYNOPTIC, RequiredDate, latLims, lonLims)

    def ReadDataForDate(self, CubeLevel, RequiredDate, latLims=None, lonLims=None, maskNoData=False, cacheThisRead=False):
        '''Reads the data representing the given "cube level" i.e. type of summary (monthly, annual, synoptic, static)


        '''
        rasterFilename = None
        if CubeLevel is None:
            CubeLevel = CubeLevels.STATIC
        isAvail = self.CubeLevelIsAvailable(CubeLevel)
        if not isAvail:
            raise ValueError("No files are present of the requested CubeLevel type")
        if CubeLevel == CubeLevels.MONTHLY:
            firstOfMonth = date(RequiredDate.year, RequiredDate.month, 1)
            if self.MonthlyDictionary.has_key(firstOfMonth):
                rasterFilename = self.MonthlyDictionary[firstOfMonth]
        elif CubeLevel == CubeLevels.ANNUAL:
            year = RequiredDate.year
            if self.AnnualDictionary.has_key(year):
                rasterFilename = self.AnnualDictionary[year]
        elif CubeLevel == CubeLevels.SYNOPTIC:
            if RequiredDate is None:
                mth = "Overall"
            else:
                mth = str(RequiredDate.month).zfill(2)
            if self.SynopticDictionary.has_key(mth):
                rasterFilename = self.SynopticDictionary[mth]
        elif CubeLevel == CubeLevels.STATIC:
            rasterFilename = self.__StaticFilename
        else:
            self.log("Unknown value for CubeLevel parameter")
        if rasterFilename is not None:
            if self.__CanCacheData and cacheThisRead:
                # the cube object can optionally maintain one TiffFile object with caching enabled (any more would
                # be memory prohibitive). An example use case would be if we know we're going to read the synoptic data
                # more than once and also read multiple dynamic files all from the same cube; we could cache the
                # synoptic one
                if self.__DataCache["Filename"] == rasterFilename:
                    thisTiff = self.__DataCache["FileObject"]
                else:
                    thisTiff = SingleBandTiffFile(rasterFilename, shouldCache=True)
                    thisTiff.PopulateCache()
                    self.__DataCache["Filename"] = rasterFilename
                    self.__DataCache["FileObject"] = thisTiff
            else:
                thisTiff = SingleBandTiffFile(rasterFilename, shouldCache=False)

            if latLims is None:
                # currently neither or both of latLims and lonLims must be set
                assert lonLims is None
                self.log("Reading data from complete file " + rasterFilename)
            else:
                self.log("Reading data from part of file " + rasterFilename)
            return thisTiff.ReadForLatLonLims(lonLims, latLims, readAsMasked=maskNoData)
            #dataArr, subsetGT, _, _ = ReadAOI_PixelLims(rasterFilename, pixelLimsLon, pixelLimsLat, maskNoData=maskNoData)

        else:
            self.log("No matching filename found")
            return None

    def __isInt(self, s):
        try:
            _ = int(s, 10)
            return True
        except:
            return False