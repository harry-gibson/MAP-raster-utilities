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
    ONE_K = "1km" # 1.0 / 120.0
    FIVE_K = "5km" # 1.0 / 24.0
    TEN_K = "10km" # 1.0 / 12.0

class CubeLevels(Enum):
    # string values of the enum give the names the folder must have
    MONTHLY = "Monthly"
    ANNUAL = "Annual"
    SYNOPTIC_MONTHLY = "Synoptic"
    SYNOPTIC_OVERALL = "Synoptic_Overall" # the overall files will actually be stored in the "Synoptic" folder too
    STATIC = "Static"
    N_DAILY = "*-Daily"
