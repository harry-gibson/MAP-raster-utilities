class AggregationModes:
    CONTINUOUS = "continuous"
    CATEGORICAL = "categorical"

class AggregationTypes:
    RESOLUTION = "resolution"
    FACTOR = "factor"
    SIZE = "size"

class TemporalAggregationStats:
    MIN = "min"
    MAX = "max"
    RANGE = "range"
    SUM = "sum"
    MEAN = "mean"
    SD = "SD"
    COUNT = "count"
    ALL = [MIN, MAX, RANGE, SUM, MEAN, SD, COUNT]

class ContinuousAggregationStats:
    MIN = "min"
    MAX = "max"
    RANGE = "range"
    SUM = "sum"
    MEAN = "mean"
    SD = "SD"
    COUNT = "count"
    ALL = [MIN, MAX, RANGE, SUM, MEAN, SD, COUNT]

class CategoricalAggregationStats:
    MAJORITY = "MajorityClass"
    FRACTIONS = "fraction"
    LIKEADJACENCIES = "like-adjacency"
    ALL = [MAJORITY, FRACTIONS, LIKEADJACENCIES]
