from enum import Enum

class SpatialSummaryTypes(Enum):
    '''Summarises the spatial aggregation types we store'''
    MAX = "max"
    MEAN = "mean"
    MIN = "min"
    RANGE = "range"
    SD = "SD"
    RAWDATA = "Data"
    FRACTIONAL = "ss_fractionalbinary"

class CubeResolutions(Enum):
    ONE_K = 1.0 / 120.0
    FIVE_K = 1.0 / 24.0
    TEN_K = 1.0 / 12.0

class CubeLevels(Enum):
    MONTHLY = "monthly"
    ANNUAL = "annual"
    SYNOPTIC = "synoptic"
    STATIC = "static variable"
