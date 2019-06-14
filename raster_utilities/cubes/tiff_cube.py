import os, glob
from datetime import date
from dateutil.relativedelta import relativedelta
from collections import defaultdict
from raster_utilities.cubes.cube_constants import CubeResolutions, CubeLevels
from ..aggregation.aggregation_values import TemporalAggregationStats, ContinuousAggregationStats
from ..io.TiffFile import SingleBandTiffFile

class TiffCube:
    ''' Represents a view of a subset of a MAP mastergrids data cube folder for a single variable and
     a single resolution and spatial / temporal aggregation statistic

     Based on a folder structure documented elsewhere, which in the present implementation must already exist
     and be populated (the TiffCube is currently readonly).
     In summary the given Masterfolder should contain subfolders for resolution: any / all of "1k", "5k", "10k".
     Each of these should contain subfolders named any/all of "Annual", "Monthly", "Synoptic", "n-Daily" (where n
     is a digit).
     Files within each of these folders should have base filenames with six dot-delimited tokens,
     with the following meanings:

    {VariableName}.{Year}.{Month}.{TemporalSummary}.{Res}.{SpatialSummary}.tif
    - The first token can contain arbitrary description information with other delimiters, such as hyphens
    - The second token can be the string "Synoptic" instead of a year.
    - The third token can be a month number, or the string "Overall" iif "Synoptic" is in pos 2, or a 3-digit julian
      day, or a 4 digit month day (0101 to 1231)
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
                    temporalsummary = TemporalAggregationStats.RAWDATA,
                    spatialsummary = ContinuousAggregationStats.RAWDATA,
                    allowGlobalCache = True,
                    cubelevel = None,
                    variablename = "*",  # for use in case there's multiple things in one folder tree differentiated in the first token
    ):
        self.MasterFolder = MasterFolder
        self.VariableName = variablename
        self.MonthlyDictionary = {}
        self.AnnualDictionary = {}
        self.SynopticDictionary = {}
        self.DailyDictionary = {}
        self.__HasMonthly = False
        self.__HasAnnual = False
        self.__HasSynoptic = False
        self.__HasStatic = False
        self.__HasDaily = False

        self.__DataCache = {"Filename": None, "FileObject": None}
        self.__CanCacheData = allowGlobalCache

        self.CubeLevel = cubelevel
        self.CubeResolution = resolution
        self.SpatialSummary = spatialsummary
        self.TemporalSummary = temporalsummary

        self.__InitialiseFiles()

    def log(self, msg):
        print(msg)

    def __InitialiseNDaily(self):
        pass
        
    def __InitialiseFiles(self):
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
        if self.CubeResolution== CubeResolutions.ONE_K:
            res = "1km"
        elif self.CubeResolution == CubeResolutions.FIVE_K:
            res = "5km"
        elif self.CubeResolution == CubeResolutions.TEN_K:
            res = "10km"
        else:
            raise ValueError("resolution of " + self.CubeResolution + " not recognised!")

        temporalsummary = self.TemporalSummary.value
        spatialsummary = self.SpatialSummary.value

        # this cube object represents one of the (maybe) available resolutions
        resFolderPath = os.path.join(self.MasterFolder, res)
        filenameTemplate = "{0!s}.{1!s}.{2!s}.{3!s}.{4!s}.{5!s}.tif"
        searchedPaths = []
        # this cube object can reflect various temporal summaries of the data, depending on what's avail.
        # Parse all that's available from the various cube levels as required

        dailyWildcardFolderPath = os.path.join(resFolderPath, CubeLevels.N_DAILY.value)
        dailyFolders = glob.glob(dailyWildcardFolderPath)
        if len(dailyFolders) > 0:
            # if any n-daily folders are present, store all matching files against a dictionary keyed by their date.
            # First search for files matching the specified spatial and temporal summary and if none are found (which
            # there won't be at the top level of the cube)
            if self.CubeLevel is None or self.CubeLevel == CubeLevels.N_DAILY:
                # we have at least one folder like 8-Daily. In all current cases that's the only match there will be.
                dailyFileWildcard = filenameTemplate.format(self.VariableName, # variablename
                                                            "*", "*", # year and julian-day or mmdd
                                                            temporalsummary, res, spatialsummary)
                dailyFilesWildcardPath = os.path.join(dailyWildcardFolderPath, dailyFileWildcard)
                allNDailyFiles = glob.glob(dailyFilesWildcardPath)
                searchedPaths.append(dailyFilesWildcardPath)
                if len(allNDailyFiles) == 0:
                    dailyFileWildcard = filenameTemplate.format(self.VariableName, # variablename
                                                            "*", "*", # year and julian-day or mmdd
                                                            TemporalAggregationStats.RAWDATA.value,
                                                            res,
                                                            ContinuousAggregationStats.RAWDATA.value)
                    dailyFilesWildcardPath = os.path.join(dailyWildcardFolderPath, dailyFileWildcard)
                    allNDailyFiles = glob.glob(dailyFilesWildcardPath)
                    searchedPaths.append(dailyFilesWildcardPath)
                if len(allNDailyFiles) == 0:
                    self.__HasDaily = False
                else:
                    self.__HasDaily = True
                for f in allNDailyFiles:
                    (variablename, yearOrSynoptic, dayOrDate, temporalType,
                     resolution, spatialType) = self._TryParseCubeFilename(f)
                    filedate = None
                    if len(dayOrDate) == 3:
                        # it is a julian day
                        daysFromNYD = relativedelta(days=int(dayOrDate)-1)
                        filedate = date(int(yearOrSynoptic), 1, 1) + daysFromNYD
                    elif len(dayOrDate) == 4:
                        mth = int(dayOrDate[0:2])
                        day = int(dayOrDate[2:])
                        filedate = date(int(yearOrSynoptic), mth, day)
                    else:
                        raise ValueError("Couldn't parse date of file " + f)
                    if self.DailyDictionary.has_key(filedate):
                        raise RuntimeError("Found more than one N-daily file for " + str(filedate) +
                                           " - this is not currently supported")
                    self.DailyDictionary[filedate] = f


        monthlyFolderPath = os.path.join(resFolderPath, CubeLevels.MONTHLY.value)
        if os.path.isdir(monthlyFolderPath):
            self.__HasMonthly = True
            if self.CubeLevel is None or self.CubeLevel == CubeLevels.MONTHLY:
            # if a monthly folder is present store all matching files against a dictionary keyed by a
            # date object (first of the month in question)
                monthlyWildcard = filenameTemplate.format(self.VariableName, # variablename
                                                          "*", "*", # yr, month
                                                          temporalsummary, res, spatialsummary)
                monthlyWildcardPath = os.path.join(monthlyFolderPath, monthlyWildcard)
                monthlyFiles = glob.glob(monthlyWildcardPath)
                searchedPaths.append(monthlyWildcardPath)
                if len(monthlyFiles) == 0:
                    self.__HasMonthly = False
                    # self.log('No monthly files found with wildcard ' + monthlyWildcardPath)
                for f in monthlyFiles:
                    (variablename, yearOrSynoptic, monthOrOverall, temporalType,
                     resolution, spatialType) = self._TryParseCubeFilename(f)
                    if self.__isInt(yearOrSynoptic) and self.__isInt(monthOrOverall):
                        filedate = date(int(yearOrSynoptic), int(monthOrOverall), 1)
                        self.MonthlyDictionary[filedate] = f
                    else:
                        self.log("File {0!s} is in monthly folder but filename does not match format"
                                 .format(f))
            else:
                pass
                # self.log("Monthly folder path of {0!s} not found".format(monthlyFolderPath))

        annualFolderPath = os.path.join(resFolderPath, CubeLevels.ANNUAL.value)
        if os.path.isdir(annualFolderPath):
            self.__HasAnnual = True # flag this so we don't identify as a static variable in the synoptic parser
            if self.CubeLevel is None or self.CubeLevel == CubeLevels.ANNUAL:
            # if an annual folder is present store all matching files against a dictionary keyed by an
            # int - the year in question
                annualWildcard = filenameTemplate.format(self.VariableName,
                                                         "*", "Annual",
                                                         temporalsummary, res, spatialsummary)
                annualWildcardPath = os.path.join(annualFolderPath, annualWildcard)
                annualFiles = glob.glob(annualWildcardPath)
                searchedPaths.append(annualWildcardPath)
                if len(annualFiles) == 0:
                    self.__HasAnnual = False # if for some reason the folder exists but empty, unflag
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

        if (self.CubeLevel is None or self.CubeLevel == CubeLevels.SYNOPTIC_MONTHLY or
            self.CubeLevel == CubeLevels.SYNOPTIC_OVERALL):
            # if a synoptic folder is present store all matching files against a dictionary keyed by a
            # string - the 2-digits month number or the constant "Overall"
            synopticFolderPath = os.path.join(resFolderPath, CubeLevels.SYNOPTIC_MONTHLY.value)
            if os.path.isdir(synopticFolderPath):
                synopticWildcard = filenameTemplate.format("*",
                                                           "Synoptic", "*",
                                                           temporalsummary, res, spatialsummary)
                synopticWildcardPath = os.path.join(synopticFolderPath, synopticWildcard)
                synopticFiles = glob.glob(synopticWildcardPath)
                searchedPaths.append(synopticWildcardPath)
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

        if not (self.__HasAnnual or self.__HasMonthly or self.__HasSynoptic or self.__HasStatic or self.__HasDaily):
            self.log('No matching files of any type have been located: aborting')
            self.log('Tried paths: ' + '\n'.join(searchedPaths))
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
        if self.VariableName == "*":
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
        if CubeLevel is None:
            CubeLevel = CubeLevels.STATIC

        rasterFilename = self.GetFilenameForDate(CubeLevel=CubeLevel, RequiredDate=RequiredDate,
                                                 useClosestAlternateYear=useClosestAvailableYear)

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

    def nearest(self, items, pivot):
        ''' Get the nearest item from iterable "items" to provided item "pivot", using default comparator '''
        # https://stackoverflow.com/a/32237949/4150190
        return min(items, key=lambda x: abs(x - pivot))

    def IsDateWithinRange(self, RequiredDate, AvailableDate, MaxRelativeDelta):
        # subtracting two datetimes gives a datetime.timedelta, which only has properties up to days (not weeks, months)
        secsFromNearest = abs((RequiredDate - AvailableDate).total_seconds())

        # datetime +- a relativeDelta gives a datetime.date
        # we need to look forward and back because in, say, March, 1 month prior is 28 days but 1 month
        # forward is 31 days, so the window is not necessarily symmetrical
        minAcceptable = RequiredDate - MaxRelativeDelta
        maxAcceptable = RequiredDate + MaxRelativeDelta

        # subtracting two datetimes gives a datetime.timedelta
        secsSinceMin = RequiredDate - minAcceptable
        secsToMax = maxAcceptable - RequiredDate

        # use whichever side of the window is greater (the nearest time image might be the other side,
        # and thus technically not in range, but i think this is a reasonable thing to do anyway)
        acceptableOffsetSeconds = max(secsSinceMin.total_seconds(), secsToMax.total_seconds())

        return secsFromNearest <= acceptableOffsetSeconds


    def GetFilenameForDate(self, CubeLevel, RequiredDate, maxDateOffset = None, useClosestAlternateYear = False):
        ''' Returns the filename of a tiff file of the corresponding temporal level (monthly, daily, etc)
        corresponding to the given date, if one is available.
        For Cubelevels.N_DAILY, a relativedelta maxDateOffset may be provided to specify
        alternative dates that can be returned. in which case if an exact match is not available the nearest available
        (forward or back) filename will be returned if it is within the specified time period.
        For Cubelevels.MONTHLY or Cubelevels.ANNUAL, a boolean useClosestAlternateYear may be provided in which case
        a date that is outside the timespan of the files available will return a file for the most recent occurrence
        or that month, or the most recent year, respectively. '''
        isAvail = self.CubeLevelIsAvailable(CubeLevel)
        if not isAvail:
            raise ValueError("No files are present of the requested CubeLevel type")
        if maxDateOffset is not None:
            if not isinstance(maxDateOffset, relativedelta):
                raise ValueError("maxDateOffset must be None or a dateutil.relativedelta object")

        rasterFilename = None

        if CubeLevel == CubeLevels.N_DAILY:
            # strip out any time portion so it doesn't throw the calcs off - no time is registered in any filename
            noTimeDate = date(RequiredDate.year, RequiredDate.month, RequiredDate.day)

            if self.DailyDictionary.has_key(noTimeDate):
                rasterFilename = self.DailyDictionary[noTimeDate]

            elif maxDateOffset is not None:
                nearestAvail = self.nearest(self.DailyDictionary.keys(), RequiredDate)
                isWithinRange = self.IsDateWithinRange(noTimeDate, nearestAvail, maxDateOffset)
                if isWithinRange:
                    rasterFilename = self.DailyDictionary[nearestAvail]
                else:
                    raise ValueError("No daily file is available within the specified offset from the given date")
            else:
                raise ValueError("maxDateOffset must be a dateutil.relativedelta object")

        elif CubeLevel == CubeLevels.MONTHLY:
            # strip out the day portion and check against the first of the month.
            # "March 2018" for a cube that runs monthly up to Dec 2017 would return "March 2017" if
            # useClosestAlternateYear is true.
            reqYear = RequiredDate.year
            reqMonth = RequiredDate.month
            firstOfMonth = date(reqYear, reqMonth, 1)
            if self.MonthlyDictionary.has_key(firstOfMonth):
                rasterFilename = self.MonthlyDictionary[firstOfMonth]
            elif useClosestAlternateYear:
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
            elif useClosestAlternateYear:
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


        return rasterFilename

    def __isInt(self, s):
        try:
            _ = int(s, 10)
            return True
        except:
            return False