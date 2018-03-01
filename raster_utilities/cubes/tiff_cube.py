import os, glob
from datetime import date
from collections import  defaultdict
from cube_constants import  CubeResolutions, CubeLevels
from ..aggregation.aggregation_values import TemporalAggregationStats, ContinuousAggregationStats
from ..io.tiff_management import ReadAOI_PixelLims, GetRasterProperties
from ..utils.geotransform_calcs import CalculatePixelLims

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
                 spatialsummary = ContinuousAggregationStats.MEAN):
        self.MasterFolder = MasterFolder
        self.VariableName = ""
        self.MonthlyDictionary = {}
        self.AnnualDictionary = {}
        self.SynopticDictionary = {}
        self.__InitialiseFiles(resolution, temporalsummary, spatialsummary)

    def log(self, msg):
        pass

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
            res = "1k"
        elif resolution == CubeResolutions.FIVE_K:
            res = "5k"
        elif resolution == CubeResolutions.TEN_K:
            res = "10k"
        else:
            raise ValueError("resoluion of " + resolution + " not recognised!")

        # this cube object represents one of the (maybe) available resolutions
        resFolderPath = os.path.join(self.MasterFolder, res)
        filenameTemplate = "{0!s}.{1!s}.{2!s}.{3!s}.{4!s}.{5!s}*.tif"

        # this cube object can reflect various temporal summaries of the data, depending on what's avail.
        # Parse all that's available from the various cube levels

        # if a monthly folder is present store all matching files against a dictionary keyed by a
        # date object (first of the month in question)
        monthlyFolderPath = os.path.join(resFolderPath, 'Monthly')
        if os.path.isdir(monthlyFolderPath):
            monthlyWildcard = filenameTemplate.format(self.VariableName,
                                                      "*", "*",
                                                      temporalsummary, res, spatialsummary)
            monthlyFiles = glob.glob(os.path.join(monthlyFolderPath, monthlyWildcard))
            for f in monthlyFiles:
                (variablename, yearOrSynoptic, monthOrOverall, temporalType,
                 resolution, spatialType) = self._TryParseCubeFilename(f)
                if self.__isInt(yearOrSynoptic) and self.__isInt(monthOrOverall):
                    filedate = date(int(yearOrSynoptic), int(monthOrOverall), 1)
                    self.MonthlyDictionary[filedate] = f
                else:
                    self.log("File {0!s} is in monthly folder but filename does not match format"
                             .format(f))

        # if an annual folder is present store all matching files against a dictionary keyed by an
        # int - the year in question
        annualFolderPath = os.path.join(resFolderPath, 'Annual')
        if os.path.isdir(annualFolderPath):
            annualWildcard = filenameTemplate.format(self.VariableName,
                                                     "*", "Annual",
                                                     temporalsummary, res, spatialsummary)
            annualFiles = glob.glob(os.path.join(annualFolderPath, annualWildcard))
            for f in annualFiles:
                (variablename, yearOrSynoptic, monthOrOverall, temporalType,
                 resolution, spatialType) = self._TryParseCubeFilename(f)
                if self.__isInt(yearOrSynoptic) and monthOrOverall == "Annual":
                    self.AnnualDictionary[int(yearOrSynoptic)] = f
                else:
                    self.log("File {0!s} is in annual folder but filename does not match format"
                             .format(f))

        # if a synoptic folder is present store all matching files against a dictionary keyed by a
        # string - the 2-digits month number or the constant "Overall"
        synopticFolderPath = os.path.join(resFolderPath, 'Synoptic')
        if os.path.isdir(synopticFolderPath):
            synopticWildcard = filenameTemplate.format(self.VariableName,
                                                       "Synoptic", "*",
                                                       temporalsummary, res, spatialsummary)
            synopticFiles = glob.glob(os.path.join(synopticFolderPath, synopticWildcard))
            for f in synopticFiles:
                (variablename, yearOrSynoptic, monthOrOverall, temporalType,
                 resolution, spatialType) = self._TryParseCubeFilename(f)
                if yearOrSynoptic == "Synoptic":
                    self.SynopticDictionary[monthOrOverall.zfill(2)] = f
                else:
                    self.log("File {0!s} is in synoptic folder but filename does not match format"
                             .format(f))


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

    def ReadMonthlyDataForDate(self, RequiredDate, latLims, lonLims):
        return self.ReadDataForDate(CubeLevels.MONTHLY, RequiredDate, latLims, lonLims)

    def ReadAnnualDataForDate(self, RequiredDate, latLims, lonLims):
        return self.ReadDataForDate(CubeLevels.ANNUAL, RequiredDate, latLims, lonLims)

    def ReadSynopticDataForDate(self, RequiredDate, latLims, lonLims):
        '''RequiredDate may be None for overall synoptic, or a date for the matching synoptic month'''
        return self.ReadDataForDate(CubeLevels.SYNOPTIC, RequiredDate, latLims, lonLims)

    def ReadDataForDate(self, CubeLevel, RequiredDate, latLims = None, lonLims = None):
        '''produces an array of data representing the content of the required term for this date

        This takes into account the temporal summary, anomaly, as appropriate for this term
        '''
        rasterFilename = None
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
            if self.SynopticDictionary.has_key("STATIC"):
                rasterFilename = self.SynopticDictionary["STATIC"]
        else:
            self.log("Unknown value for CubeLevel parameter")
        if rasterFilename is not None:
            self.log("Reading data from file " + rasterFilename)
            inGT, inProj, inNDV, inWidth, inHeight, inRes, inDT = GetRasterProperties(rasterFilename)
            pixelLims = CalculatePixelLims(inGT, lonLims, latLims)
            dataArr, subsetGT, _, _ = ReadAOI_PixelLims(rasterFilename, pixelLims[0], pixelLims[1])
            return (dataArr, subsetGT, inProj, inNDV)
        else:
            self.log("No matching filename found")
            return None

    def __isInt(self, s):
        try:
            _ = int(s, 10)
            return True
        except:
            return False