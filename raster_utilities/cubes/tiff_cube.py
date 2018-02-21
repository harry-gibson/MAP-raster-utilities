import os, glob
from datetime import date
from collections import  defaultdict
from cube_constants import  CubeResolutions
from ..aggregation.aggregation_values import TemporalAggregationStats, ContinuousAggregationStats

class TiffCube:
    '''

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
        # set the constants to be defined as numeric values, maybe could have done them as these strings
        if resolution == CubeResolutions.ONE_K:
            res = "1k"
        elif resolution == CubeResolutions.FIVE_K:
            res = "5k"
        else:
            res = "10k"

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
                    self.log("File {0!s} is in synoptic folder but filename does not matcch format"
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



    def __isInt(self, s):
        try:
            _ = int(s, 10)
            return True
        except:
            return False