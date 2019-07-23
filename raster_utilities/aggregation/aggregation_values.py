from enum import Enum

#from collections import namedtuple
#AggregationArgs = namedtuple("AggregationArgs", ["gt", "proj", "ndv", "width", "height", "res", "datatype"])

class AggregationModes(Enum):
    CONTINUOUS = "continuous"
    CATEGORICAL = "categorical"

class AggregationTypes(Enum):
    RESOLUTION = "resolution"
    FACTOR = "factor"
    SIZE = "size"

class SnapTypes(Enum):
    '''For use when calculating an adjusted geotransform that will align with mastergrids. Normally if we are adjusting
    e.g. a 30 arcsecond grid which has been stored with imprecise (rounded) resolution we would use NEAREST. However if
    we are aggregating a grid and want the output to be aligned then we would do TOWARDS_ORIGIN to ensure that the full
    extent is covered'''
    NONE = "none"
    NEAREST = "nearest"
    TOWARDS_ORIGIN = "towards_origin"

class TemporalAggregationStats(Enum):
    MIN = "min"
    MAX = "max"
    RANGE = "range"
    SUM = "sum"
    MEAN = "mean"
    SD = "SD"
    COUNT = "count"
    # these two are available in the cube naming convention but not as aggregation types per se
    RAWDATA = "Data"
    BALANCEDMEAN = "Balanced-mean"
    # hence they do not appear in ALL
    ALL = [MIN, MAX, RANGE, SUM, MEAN, SD, COUNT]

class ContinuousAggregationStats(Enum):
    MIN = "min"
    MAX = "max"
    RANGE = "range"
    SUM = "sum"
    MEAN = "mean"
    SD = "SD"
    COUNT = "count"
    RAWDATA = "Data"
    ALL = [MIN, MAX, RANGE, SUM, MEAN, SD, COUNT]

class CategoricalAggregationStats(Enum):
    MAJORITY = "majority-class"
    FRACTIONS = "fraction"
    PERCENTAGE = "percentage" # use to allow 8-bit int type if appropriate for the data
    LIKEADJACENCIES = "like-adjacency"
    ALL = [MAJORITY, FRACTIONS, LIKEADJACENCIES]
