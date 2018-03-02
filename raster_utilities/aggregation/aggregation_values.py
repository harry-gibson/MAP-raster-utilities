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
    ALL = [MIN, MAX, RANGE, SUM, MEAN, SD, COUNT]

class ContinuousAggregationStats(Enum):
    MIN = "min"
    MAX = "max"
    RANGE = "range"
    SUM = "sum"
    MEAN = "mean"
    SD = "SD"
    COUNT = "count"
    ALL = [MIN, MAX, RANGE, SUM, MEAN, SD, COUNT]

class CategoricalAggregationStats(Enum):
    MAJORITY = "MajorityClass"
    FRACTIONS = "fraction"
    LIKEADJACENCIES = "like-adjacency"
    ALL = [MAJORITY, FRACTIONS, LIKEADJACENCIES]
