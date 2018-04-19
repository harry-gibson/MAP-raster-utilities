import os, glob
from datetime import date
from collections import  defaultdict
from cube_constants import  CubeResolutions, CubeLevels
from ..aggregation.aggregation_values import TemporalAggregationStats, ContinuousAggregationStats
from ..io.TiffFile import SingleBandTiffFile

class TiffCube:
    ''' Represents a view of a subset of a MAP mastergrids data cube folder for a single variable and
     a single resolution and spatial / temporal aggregation statistic

     Based on a folder structure documented elsewhere, which in the present implementation must already exist
     and be populated (the TiffCube is currently readonly).
     In summary the given Masterfolder should contain subfolders for resolution: any / all of "1k", "5k", "10k".
     Each of these should contain subfolders named
     Files within each of these folders should have base filenames with six dot-delimited tokens,
     with the following meanings:

    {VariableName}.{Year}.{Month}.{TemporalSummary}.{Res}.{SpatialSummary}.tif
    - The first token can contain arbitrary description information with other delimiters, such as hyphens
    - The second token can be the string "Synoptic" instead of a year.
    - The third token can be a month number, or the string "Overall" iif "Synoptic" is in pos 2.
    - The fourth token can be a value from the TemporalAggregationStats enum, although only "mean" and "Data" are
      generally available
    - The sixth token can be a value from the ContinuousAggregationStats or CategoricalAggregationStats enums,
      e.g. "mean", "max", "min"...

    The TiffCube object is created with a MasterFolder path and a required temporal and spatial summary type.
    Assuming the folder matches the above structure the tree belowit will be searched for all the matching files
    of the requested spatial and temporal summary type (e.g. temporal mean and spatial SD). The TiffCube will then
    make available these data at whatever "CubeLevel" is requested (i.e. Monthly, Annual, Synoptic).

    The key functionality of this class currently is to read data for a requested date.

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

        self.VariableName = ""#os.path.basename(MasterFolder)
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
        filenameTemplate = "{0!s}.{1!s}.{2!s}.{3!s}.{4!s}.{5!s}.tif"

        # this cube object can reflect various temporal summaries of the data, depending on what's avail.
        # Parse all that's available from the various cube levels

        # if a monthly folder is present store all matching files against a dictionary keyed by a
        # date object (first of the month in question)
        monthlyFolderPath = os.path.join(resFolderPath, CubeLevels.MONTHLY.value)
        if os.path.isdir(monthlyFolderPath):
            monthlyWildcard = filenameTemplate.format("*", # variablename
                                                      "*", "*",
                                                      temporalsummary, res, spatialsummary)
            monthlyWildcardPath = os.path.join(monthlyFolderPath, monthlyWildcard)
            monthlyFiles = glob.glob(monthlyWildcardPath)
            if len(monthlyFiles) == 0:
                pass
                # self.log('No monthly files found with wildcard ' + monthlyWildcardPath)
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
            pass
            # self.log("Monthly folder path of {0!s} not found".format(monthlyFolderPath))

        # if an annual folder is present store all matching files against a dictionary keyed by an
        # int - the year in question
        annualFolderPath = os.path.join(resFolderPath, CubeLevels.ANNUAL.value)
        if os.path.isdir(annualFolderPath):
            annualWildcard = filenameTemplate.format("*",
                                                     "*", "Annual",
                                                     temporalsummary, res, spatialsummary)
            annualWildcardPath = os.path.join(annualFolderPath, annualWildcard)
            annualFiles = glob.glob(annualWildcardPath)
            if len(annualFiles) == 0:
                pass
                #self.log('No annual files found with wildcard ' + annualWildcardPath)
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
            pass
            #self.log("Annual folder path of {0!s} not found".format(annualFolderPath))

        # if a synoptic folder is present store all matching files against a dictionary keyed by a
        # string - the 2-digits month number or the constant "Overall"
        synopticFolderPath = os.path.join(resFolderPath, CubeLevels.SYNOPTIC_MONTHLY.value)
        if os.path.isdir(synopticFolderPath):
            synopticWildcard = filenameTemplate.format("*",
                                                       "Synoptic", "*",
                                                       temporalsummary, res, spatialsummary)
            synopticWildcardPath = os.path.join(synopticFolderPath, synopticWildcard)
            synopticFiles = glob.glob(synopticWildcardPath)
            if len(synopticFiles) == 0:
                pass
                #self.log('No synoptic files found with wildcard ' + synopticWildcardPath)

            for f in synopticFiles:
                (variablename, yearOrSynoptic, monthOrOverall, temporalType,
                 resolution, spatialType) = self._TryParseCubeFilename(f)
                if yearOrSynoptic == "Synoptic":
                    self.SynopticDictionary[monthOrOverall.zfill(2)] = f
                    self.__HasSynoptic = True
                else:
                    self.log("File {0!s} is in synoptic folder but filename does not match format"
                             .format(f))

            # if we have a synoptic.overall file but no others (synoptic or dynamic) then this counts as a
            # static variable, not a synoptic one
            if (self.SynopticDictionary.has_key("Overall") and len(self.SynopticDictionary.keys())==1
                and (not self.__HasAnnual) and (not self.__HasMonthly)):
                self.__HasSynoptic = False
                self.__HasStatic = True
                self.__StaticFilename = self.SynopticDictionary["Overall"]

        else:
            pass
            #self.log("Synoptic/static folder path of {0!s} not found".format(synopticFolderPath))

        if not (self.__HasAnnual or self.__HasMonthly or self.__HasSynoptic or self.__HasStatic):
            self.log('No matching files of any type have been located: aborting')
            assert False
        #staticFolderPath = os.path.join(resFolderPath, CubeLevels.STATIC.value)
        #if os.path.isdir(staticFolderPath):
            # static filenames should be "X.Synoptic.Overall.Data.res.spatial"
        #    staticWildcard = filenameTemplate.format(self.VariableName,
        #                                             "Synoptic", "Overall",
        #                                             "Data", res, spatialsummary)
        #    staticWildcardPath = os.path.join(staticFolderPath, staticWildcard)
        #    staticFiles = glob.glob(staticWildcardPath)
        #    if len(staticFiles) == 0:
        #        self.log('No static files found with wildcard ' + staticWildcardPath)

        #    elif len(staticFiles > 0):
        #        self.log('More than one static file found with wildcard ' + staticWildcardPath)
        #        assert False
        #    elif (self.__HasAnnual or self.__HasMonthly or self.__HasSynoptic):
        #        self.log('Found a static file as well as temporal ones: not supported! Skipping static file')
        #    else:
        #        self.__StaticFilename = staticFiles[0]
        #        self.__HasStatic = True


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
        elif CubeLevel == CubeLevels.SYNOPTIC_MONTHLY:
            return (self.__HasSynoptic and len(self.SynopticDictionary.keys())>1)
        elif CubeLevel == CubeLevels.SYNOPTIC_OVERALL:
            return (self.__HasSynoptic and self.SynopticDictionary.has_key("Overall"))
        elif CubeLevel == CubeLevels.STATIC:
            return self.__HasStatic
        else:
            return False

    def GetExtent(self):
        # todo calculate  and return lat lon lims
        pass

    def GetAvailableMonthlyDateRange(self):
        if self.CubeLevelIsAvailable(CubeLevels.MONTHLY):
            months = self.MonthlyDictionary.keys()
            return (min(months), max(months))
        else:
            return None

    def GetAvailableAnnualDateRange(self):
        if self.CubeLevelIsAvailable(CubeLevels.ANNUAL):
            years = self.AnnualDictionary.keys()
            return (min(years), max(years))
        else:
            return None

    def ReadMonthlyDataForDate(self, RequiredDate, lonLims=None, latLims=None):
        return self.ReadDataForDate(CubeLevels.MONTHLY, RequiredDate, lonLims, latLims)

    def ReadAnnualDataForDate(self, RequiredDate, lonLims=None, latLims=None):
        return self.ReadDataForDate(CubeLevels.ANNUAL, RequiredDate, lonLims, latLims)

    def ReadSynopticDataForDate(self, RequiredDate, lonLims=None, latLims=None):
        '''RequiredDate may be None for overall synoptic, or a date for the matching synoptic month'''
        return self.ReadDataForDate(CubeLevels.SYNOPTIC_MONTHLY, RequiredDate, lonLims, latLims)

    def ReadDataForDate(self, CubeLevel, RequiredDate,
                        lonLims=None, latLims=None,
                        maskNoData=False, cacheThisRead=False,
                        useClosestAvailableYear = False):
        '''Reads the data representing the given "cube level" i.e. type of summary (monthly, annual, synoptic, static)

        Return object is a tuple of (data, geotransform-tuple, projection-string, no-data-value).
        If maskNoData is set then the data will be a numpy masked array with the locations matching the nodata value
        being masked. Otherwise it will be a standard numpy array.
        If cacheThisRead is set then this method will read the whole file corresponding to the requested date,
        regardless of limits, and store the result - overwriting any previously cached file - before returning the
        subset requested.
        '''
        rasterFilename = None
        if CubeLevel is None:
            CubeLevel = CubeLevels.STATIC
        isAvail = self.CubeLevelIsAvailable(CubeLevel)

        if not isAvail:
            raise ValueError("No files are present of the requested CubeLevel type")

        if CubeLevel == CubeLevels.MONTHLY:
            reqYear = RequiredDate.year
            reqMonth = RequiredDate.month
            firstOfMonth = date(reqYear, reqMonth, 1)
            if self.MonthlyDictionary.has_key(firstOfMonth):
                rasterFilename = self.MonthlyDictionary[firstOfMonth]
            elif useClosestAvailableYear:
                availableOccurrencesOfThisMonth = [k for k in self.MonthlyDictionary.keys() if k.month == reqMonth]
                if RequiredDate > max(availableOccurrencesOfThisMonth):
                    firstOfMonth = max(availableOccurrencesOfThisMonth)
                elif RequiredDate < min(availableOccurrencesOfThisMonth):
                    firstOfMonth = min(availableOccurrencesOfThisMonth)
                rasterFilename = self.MonthlyDictionary[firstOfMonth]
                self.log("Requested month of {0!s} unavailable, using alternate year of {1!s}"
                         .format(RequiredDate, firstOfMonth))

        elif CubeLevel == CubeLevels.ANNUAL:
            year = RequiredDate.year
            if self.AnnualDictionary.has_key(year):
                rasterFilename = self.AnnualDictionary[year]
            elif useClosestAvailableYear:
                years = self.GetAvailableAnnualDateRange()
                if year > years[1]:
                    year = years[1]
                elif year < years[0]:
                    year = years[0]
                rasterFilename = self.AnnualDictionary[year]
                self.log("Requested year of {0!s} unavailable, using alternate year of {1!s}"
                         .format(RequiredDate, year))

        elif CubeLevel == CubeLevels.SYNOPTIC_MONTHLY:
            if RequiredDate is None:
                mth = "Overall"
            else:
                mth = str(RequiredDate.month).zfill(2)
            if self.SynopticDictionary.has_key(mth):
                rasterFilename = self.SynopticDictionary[mth]

        elif CubeLevel == CubeLevels.SYNOPTIC_OVERALL:
            mth = "Overall"
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
                pass
                self.log("Reading data from part of file " + rasterFilename)
            return thisTiff.ReadForLatLonLims(lonLims=lonLims, latLims=latLims, readAsMasked=maskNoData)
            #dataArr, subsetGT, _, _ = ReadAOI_PixelLims(rasterFilename, pixelLimsLon, pixelLimsLat, maskNoData=maskNoData)

        else:
            self.log("No matching filename found for {0!s} / {1!s} / {2!s}".format(CubeLevel, RequiredDate, self.MasterFolder))
            return None

    def __isInt(self, s):
        try:
            _ = int(s, 10)
            return True
        except:
            return False