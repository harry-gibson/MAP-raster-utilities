import math
from ..utils.logger import MessageLogger, LogLevels
from ..aggregation.aggregation_values import AggregationTypes, SnapTypes

def calcAggregatedProperties(method, inRasterProps,
                             aggregationSpecifier = None,
                             inputSnapType = SnapTypes.NONE,
                             outputSnapType = SnapTypes.NEAREST):
                             #aggFactor=None, outShape=None, outResolution=None, snapType = SnapTypes.NONE):
    ''' Given an input raster, get the post-aggregation geotransform and dimensions

    method should be one of AggregationTypes.Factor, AggregationTypes.Size, or
    AggregationTypes.Resolutions (or a string "factor", "size", or "resolution");
    the appropriate one of the other parameters should then also be set to determine
    either the cell size multiplication factor, a desired output shape, or a desired
    output resolution.

    For the resolution method, a numeric value can be provided, or a string. A string
    must be one of "1k", "5k", or "10k" to set the resolution to 30 arcseconds, 2.5
    arcminutes, or 5 arcminutes respectively (assuming that the geotransform units are in
    degrees).

    Resolution and size methods may result in a different aspect ratio from the input, meaning
    that the cell shapes may be changed. Resolution method will always result in square cells.

    Returns a 2-tuple where first item is the geotransform (itself a 6-tuple,
    according to the GDAL standard, and the second item is the shape (a 2-tuple,
    according to numpy standard)

    calcAggregatedProperties("size", (17400,43200), origGT, None, (4320, 8640), None)

    calcAggregatedProperties("resolution", (17400,43200), origGT, None, None, 0.04166666666665)

    calcAggregatedProperties("resolution", (17400, 43200), origGT, None, None, "5km")

    calcAggregatedProperties("factor", (21600,43200), origGT, 10, None, None)


    '''
    from ..io.tiff_management import RasterProps

    assert isinstance(inRasterProps, RasterProps)
    assert len(inRasterProps.gt) == 6

    # assert aggFactor is not None or outShape is not None or outResolution is not None
    logger = MessageLogger()
    if method == "factor" or method == AggregationTypes.FACTOR:
        # the simplest version, output is just n times coarser than before
        # if it's not a clean match then we will expand to get everything
        # Implies the same pixel shape as before just larger
        if not (isinstance(aggregationSpecifier, int) and aggregationSpecifier > 0):
            raise ValueError("For aggregation type 'factor' the specifier must be an integer factor > 0")
        outXSize = int(math.ceil(1.0 * inRasterProps.width / aggregationSpecifier))
        outYSize = int(math.ceil(1.0 * inRasterProps.height / aggregationSpecifier))
        if inRasterProps.width % aggregationSpecifier != 0 or inRasterProps.height % aggregationSpecifier != 0:
            # this will tend to trigger with irrational cell sizes such as 1/120 so we should probably
            # put a tolerance on it
            logger.logMessage("warning, input size was not clean multiple of factor, "+
                       "output will have 1 cell greater extent", LogLevels.WARNING)
        outputGT = (inRasterProps.gt[0], inRasterProps.gt[1] * aggregationSpecifier, 0.0,
                    inRasterProps.gt[3], 0.0, inRasterProps.gt[5] * aggregationSpecifier)

    elif method == "size" or method == AggregationTypes.SIZE:
        # we will specify the required pixel dimensions of the output: aspect ratio may change
        # this is generally the best method to use if we're working with non-integer cell resolutions
        if not (isinstance(aggregationSpecifier, tuple) and len(aggregationSpecifier)==2
                and isinstance(aggregationSpecifier[0], int) and isinstance(aggregationSpecifier[0], int)):
            raise ValueError("For aggregation type 'size' the specifier must be a 2-tuple of ints (height, width)")
        outXSize = aggregationSpecifier[1]
        outYSize = aggregationSpecifier[0]
        # now we need to calculate the resolution this implies (for a maintained extent)
        inputHeightPx = inRasterProps.height
        inputWidthPx = inRasterProps.width
        if 1.0 * inputHeightPx / inputWidthPx != 1.0 * outYSize / outXSize:
            # the coverage in ground extent will be the same, so the pixels themselves must change shape
            logger.logMessage("warning, output size is different proportion to input, "+
                       "cells will change shape", LogLevels.WARNING)
        inputXMin = inRasterProps.gt[0]
        inputYMax = inRasterProps.gt[3]
        inputXMax = inputXMin + inRasterProps.gt[1] * inputWidthPx
        # y resolution is negative as origin is top left
        inputYMin = inputYMax + inRasterProps.gt[5] * inputHeightPx
        inputHeightProj = inputYMax - inputYMin
        inputWidthProj = inputXMax - inputXMin
        outputResX = inputWidthProj / outXSize
        outputResY = inputHeightProj / outYSize
        outputGT = (inRasterProps.gt[0], outputResX, 0.0, inRasterProps.gt[3], 0.0, -outputResY)

    elif method == "resolution" or method == AggregationTypes.RESOLUTION:
        inGT = inRasterProps.gt
        inResX = inGT[1]
        inResY = inGT[5]
        inXSize = inRasterProps.width
        inYSize = inRasterProps.height
        xOrigin = inGT[0]
        yOrigin = inGT[3]
        if not inResY == -inResX:
            logger.logMessage("warning, input had non-square cells, "+
                       "resolution mode creates square cells so they will change shape", LogLevels.WARNING)
        if isinstance(aggregationSpecifier, str):
            # use the hardcoded resolutions for the 3 main MAP resolutions to avoid
            # cocking about with irrational numbers not multiplying / dividing cleanly
            inResX = SanitiseResolution(inResX)
            inResY = -(SanitiseResolution(-inResY))
            inResXRnd = round(inResX, 8)
            inResYRnd = round(inResY, 8)
            if inResXRnd == 0.00833333:
                inResX = 1.0 / 120.0
            elif inResXRnd == 0.04166667:
                inResX = 1.0 / 24.0
            elif inResXRnd == 0.08333333:
                inResX = 1.0 / 12.0
            if inResYRnd == -0.00833333:
                inResY = -1.0 / 120.0
            elif inResYRnd == -0.04166667:
                inResY = -1.0 / 24.0
            elif inResYRnd == -0.08333333:
                inResY = -1.0 / 12.0
            xExtent = inXSize * inResX
            yExtent = inYSize * -inResY
            if aggregationSpecifier.lower().startswith("1k"):
                outXSize = int(120 * xExtent)
                outYSize = int(120 * yExtent)
                outResolution = 1.0/120
            elif aggregationSpecifier.lower().startswith("5k"):
                outXSize = int(24 * xExtent)
                outYSize = int(24 * yExtent)
                outResolution = 1.0/24
            elif aggregationSpecifier.lower().startswith("10k"):
                outXSize = int(12 * xExtent)
                outYSize = int(12 * yExtent)
                outResolution = 1.0/12
            else:
                logger.logMessage("Unknown string resolution description!", LogLevels.ERROR)
                assert False
        else:
            # specify a desired output resolution, implies the same in both directions (square pixels)
            outResolution = SanitiseResolution(aggregationSpecifier)
            xFactor = round(1.0 * outResolution / inGT[1], 8)
            yFactor = round(-1.0 * outResolution / inGT[5], 8)
            outX_exact = 1.0 * inXSize / xFactor
            outY_exact = 1.0 * inYSize / yFactor
            outXSize = math.ceil(outX_exact)
            outYSize = math.ceil(outY_exact)
            if outX_exact != outXSize or outY_exact != outYSize:
                logger.logMessage("warning, specified resolution was not clean multiple of input, "+
                           "output will have 1 cell greater extent", LogLevels.WARNING)
        outputGT = (inGT[0], outResolution, 0.0, inGT[3], 0.0, -outResolution)
    else:
        logger.logMessage("Unknown aggregation type requested, valid values are 'size', 'resolution', 'factor'",
                   LogLevels.ERROR)
        assert False

    return (
        outputGT,
        (outYSize, outXSize)
    )


def CalculateClippedGeoTransform(inGT, xPixelLims, yPixelLims):
    '''Returns the GDAL geotransform of a clipped subset of an existing geotransform

    Where clipping coordinates are specified as pixel limits'''
    topLeftLongIn = inGT[0]
    topLeftLatIn = inGT[3]
    resX = inGT[1]
    resY = inGT[5]

    # the origin coord of the output is simply a whole number of pixels from the input origin
    topLeftLongOut = topLeftLongIn + xPixelLims[0] * resX
    topLeftLatOut = topLeftLatIn + yPixelLims[0] * resY

    clippedGT = (topLeftLongOut, resX, 0.0, topLeftLatOut, 0.0, resY)
    return clippedGT

def CalculateClippedGeoTransform_RoundedRes(inGT, xPixelLims, yPixelLims):
    '''Returns the GDAL geotransform of a clipped subset of an existing geotransform

    Where clipping coordinates are specified as pixel limits, and the resolution of the output file
    is one of the the older (inaccurate) versions used in MAP pre-2014 (e.g. 0.00833333 degrees), resulting
    in a geotransform that will line up with older "mastergrids" files'''
    topLeftLongIn = inGT[0]
    topLeftLatIn = inGT[3]
    resX = inGT[1]
    resY = inGT[5]

    # the input geotransform should have "true" resolution, i.e. sufficient decimal places to specify the
    # irrational number resolutions accurately enough to ensure alignment in a global grid. We will output in
    # the less-precise version used earlier where for example 1/120 ws specified as 0.00833333 rather than
    # 0.008333333333333
    roundedResX = round(resX, 8)
    roundedResY = round(resY, 8)
    assert roundedResX != resX
    assert roundedResY != resY

    # top left pixel coordinate of the input image relative to a fully global image with origin at -180, +90
    topLeftPixelX = (topLeftLongIn - -180.0) / resX
    topLeftPixelY = (90.0 - topLeftLatIn) / (-resY)

    # what would the lat/lon of this input pixel be in a grid with the rounded resolution
    # the rounded mastergrid "global" origin is at -180.0, 89.99994
    topLeftLongInMG = -180.0 + (topLeftPixelX * roundedResX)
    # NB y-res is negative
    topLeftLatInMG = 89.99994 + (topLeftPixelY * roundedResY)

    topLeftLongOut = topLeftLongInMG + xPixelLims[0] * roundedResX
    topLeftLatOut = topLeftLatInMG + yPixelLims[0] * resY

    clippedGT = (topLeftLongOut, roundedResX, 0.0, topLeftLatOut, 0.0, roundedResY)
    return clippedGT

def SanitiseResolution(resolutionValue):
    ''' Gets a "sanitised" version of the cell resolution.
      If the resolution is less than 1 then it's assumed to be a erroneously-rounded decimal fraction, probably of a degree, as
      will be the case with lat/lon grids. An "unrounded" version of this will be returned, e.g.
      0.00833333 -> 0.00833333333333333333.
      If the resolution is greater than 1 then it's assumed to be an erroneously-precise value that should be an integer,
      as will be the case with a grid where something's not been set quite right in the extent calculation. In this case
      it will just return the rounded int version of the passed resolution e.g. 30.00000103 -> 30'''
    if resolutionValue < 1:
        # round the number of cells per degree to the nearest integer
        intDiv = round(1.0 / resolutionValue)
        # then get a more accurate representation of that in degrees per cell,
        # this will transform e.g. 0.00833333 to 0.0083333333333333
        idealRes = round(1.0 / intDiv, 20)
        return idealRes
    return round(resolutionValue)

def SnapAndAlignGeoTransform(inGT, fixResolution = True, snapType=SnapTypes.NEAREST):
    ''' Makes up to two types of change to a geotransform to get it to align to the pixels of a global grid.

    First the resolution may be "sanitised" such that the cell resolution is a rational fraction of 1 - so
    assuming that the units are degrees, a resolution of 0.00833334 (imprecise specification of 30 arcsecond) would
    be set to exactly 1/120

    Second the origin point (top left corner) may be snapped to the location it would have if part of an entire global
    grid of this resolution (after sanitising the resolution, if chosen), such that the grid will line up with global
    ones'''

    assert isinstance(inGT, tuple) and len(inGT) == 6

    def round_to_nearest(n, m):
        r = n % m
        # if it's more than half a cell from the origin round "up" i.e. away from origin
        return n + m - r if r + r >= m else n - r

    def round_to_negative(n, m):
        r = n % m
        return n - r

    def round_to_positive(n, m):
        r = n % -m
        return n - r

    xRes = inGT[1]
    yRes = inGT[5]
    # we will only support square cells
    assert round(xRes, 10) == -(round(yRes, 10))
    xOrigin = inGT[0]
    yOrigin = inGT[3]

    idealRes_X = SanitiseResolution(xRes)
    # the function wants a value>0
    idealRes_Y = -(SanitiseResolution(-yRes))

    outRes_X = xRes
    outRes_Y = yRes
    new_xOrigin = xOrigin
    new_yOrigin = yOrigin
    logger = MessageLogger()
    if fixResolution:
        if outRes_X == idealRes_X and outRes_Y == idealRes_Y:
            logger.logMessage("Cellsize already ok, not altering")
            # return None
        else:
            logger.logMessage("Cellsize input (x,y): {}*{}, sanitised to cellsize of: {}*{}".format(xRes, yRes, idealRes_X, idealRes_Y))
            outRes_X = idealRes_X
            outRes_Y = idealRes_Y
      #  elif xRes != idealRes_X or yRes != idealRes_Y:
      #      logMessage("Cellsize in: {}*{}".format(xRes, yRes))
      #      new_xOrigin = round_to_nearest(xOrigin, outRes_X)
      #      new_yOrigin = round_to_nearest(yOrigin, -outRes_Y) # the rounding expects a +ve number to work
        if snapType == SnapTypes.NONE:
            new_xOrigin = xOrigin # redundant but left for clarity of working
            new_yOrigin = yOrigin

        elif snapType == SnapTypes.TOWARDS_ORIGIN:
            # for rounding to origin it differs in x and y direction, because y resolution is negative
            # for longitude (x), always round towards more negative
            # for latitude (y) always round towards more positive
            new_xOrigin = round_to_negative(xOrigin, outRes_X)
            new_yOrigin = round_to_positive(yOrigin, -outRes_Y) # the rounding expects a positive number to work

        elif snapType == SnapTypes.NEAREST:
            new_xOrigin = round_to_nearest(xOrigin, outRes_X)
            new_yOrigin = round_to_nearest(yOrigin, -outRes_Y)

        else:
            raise ValueError("Unknown value of snapType parameter")

        if snapType != SnapTypes.NONE:
            if (new_yOrigin != yOrigin) or (new_xOrigin != xOrigin):
                logger.logMessage("Snapped origin point from {} (x,y) to {} (x,y)".format((xOrigin, yOrigin), (new_xOrigin, new_yOrigin)))
            else:
                logger.logMessage("Origin point was already correctly aligned at {} (x,y)".format((xOrigin, yOrigin)))
    else:
        if snapType != SnapTypes.NONE:
            logger.logMessage("Does not make sense to snap geotransform without also sanitising resolution, output will be unchanged",
                              LogLevels.WARNING)
        else:
            logger.logMessage("No correction to resolution or alignment was requested, output is unchanged", LogLevels.WARNING)
    return (new_xOrigin, outRes_X, 0, new_yOrigin, 0, outRes_Y)

def CalculatePixelLims_GlobalRef(inGT, longitudeLims, latitudeLims):
    '''Returns pixel coords of a given AOI in degrees, as they *would* be in a global image

    The resolution is taken from the input geotransform but the origin relative to which the
    coords are calculated is taken as -180,90 regardless of what is in the geotransform.

    Returns pixel coordinates as ((xMin, xMax), (yMax, yMin)); these represent the pixel window on a
    global image of the matching resolution that includes the request bbox.

    The main mastergrid resolutions of 30 arcsecond, 2.5 arcminute, 5 arcminute are treated
    explicitly to avoid rounding errors.'''

    assert isinstance(longitudeLims, tuple) and len(longitudeLims) == 2
    assert isinstance(latitudeLims, tuple) and len(latitudeLims) == 2
    assert isinstance(inGT, tuple) and len(inGT) == 6

    EastLimitOut = longitudeLims[1]
    WestLimitOut = longitudeLims[0]
    NorthLimitOut = latitudeLims[0]
    SouthLimitOut = latitudeLims[1]

    assert EastLimitOut > WestLimitOut
    assert NorthLimitOut > SouthLimitOut

    xRes = inGT[1]
    yRes = -(inGT[5])
    # check for the commonly-used imprecise resolutions and recalculate to ensure we don't end up missing a pixel
    inResXRnd = round(xRes, 8)
    inResYRnd = round(yRes, 8)
    if inResXRnd == 0.00833333:
        xRes = 1.0 / 120.0
    elif inResXRnd == 0.04166667:
        xRes = 1.0 / 24.0
    elif inResXRnd == 0.08333333:
        xRes= 1.0 / 12.0
    if inResYRnd == 0.00833333:
        yRes= 1.0 / 120.0
    elif inResYRnd == 0.04166667:
        yRes= 1.0 / 24.0
    elif inResYRnd == 0.08333333:
        yRes= 1.0 / 12.0

    OverallNorthLimit = 90.
    OverallWestLimit = -180.
    x0 = int((WestLimitOut - OverallWestLimit) / xRes)
    x1 = int(((EastLimitOut - OverallWestLimit) / xRes) + 0.5)
    y0 = int((OverallNorthLimit - NorthLimitOut) / yRes)
    y1 = int(((OverallNorthLimit - SouthLimitOut) / yRes) + 0.5)

    return ((x0, x1), (y0, y1))

def CalculatePixelLims(inGT, longitudeLims, latitudeLims):
    '''Returns pixel coords of a given AOI in degrees, in an image with the given geotransform

    Returns pixel coordinates as ((xMin, xMax), (yMax, yMin)); these represent the pixel window on a
    global image of the matching resolution that includes the request bbox.

    The main mastergrid resolutions of 30 arcsecond, 2.5 arcminute, 5 arcminute are treated
    explicitly to avoid rounding errors.'''

    assert isinstance(longitudeLims, tuple) and len(longitudeLims) == 2
    assert isinstance(latitudeLims, tuple) and len(latitudeLims) == 2
    assert isinstance(inGT, tuple) and len(inGT) == 6

    EastLimitOut = longitudeLims[1]
    WestLimitOut = longitudeLims[0]
    NorthLimitOut = latitudeLims[0]
    SouthLimitOut = latitudeLims[1]

    assert EastLimitOut > WestLimitOut
    assert NorthLimitOut > SouthLimitOut

    xRes = inGT[1]
    yRes = -(inGT[5])
    # check for the commonly-used imprecise resolutions and recalculate to ensure we don't end up missing a pixel
    inResXRnd = round(xRes, 8)
    inResYRnd = round(yRes, 8)
    if inResXRnd == 0.00833333:
        xRes = 1.0 / 120.0
    elif inResXRnd == 0.04166667:
        xRes = 1.0 / 24.0
    elif inResXRnd == 0.08333333:
        xRes = 1.0 / 12.0
    if inResYRnd == 0.00833333:
        yRes = 1.0 / 120.0
    elif inResYRnd == 0.04166667:
        yRes = 1.0 / 24.0
    elif inResYRnd == 0.08333333:
        yRes = 1.0 / 12.0

    OverallNorthLimit = inGT[3]
    OverallWestLimit = inGT[0]

    x0 = int((WestLimitOut - OverallWestLimit) / xRes)
    x1 = int(((EastLimitOut - OverallWestLimit) / xRes) + 0.5)
    y0 = int((OverallNorthLimit - NorthLimitOut) / yRes)
    y1 = int(((OverallNorthLimit - SouthLimitOut) / yRes) + 0.5)

    return ((x0, x1), (y0, y1))
