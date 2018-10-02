from enum import Enum
class AggregationModes(Enum):
    CONTINUOUS = "continuous"
    CATEGORICAL = "categorical"

class AggregationTypes(Enum):
    RESOLUTION = "resolution"
    FACTOR = "factor"
    SIZE = "size"

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
