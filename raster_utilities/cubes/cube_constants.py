class SpatialSummaryTypes:
    MAX = "max"
    MEAN = "mean"
    MIN = "min"
    RANGE = "range"
    SD = "SD"
    FRACTIONAL = "ss_fractionalbinary"

class TemporalSummaryTypes:
    D_MONTHLY = "Dynamic monthly"
    D_ANNUAL = "Dynamic annual"
    S_MONTHLY_MEAN = "Synoptic Monthly Mean"
    S_MONTHLY_SD = "Synoptic Monthly SD"
    S_ANNUAL_MEAN = "Synoptic Annual Mean"
    STATIC = "Static variable"

class CubeResolutions:
    ONE_K = 1.0 / 120.0
    FIVE_K = 1.0 / 24.0
    TEN_K = 1.0 / 12.0

class CubeLevels:
    MONTHLY = "monthly"
    ANNUAL = "annual"
    SYNOPTIC = "synoptic"
