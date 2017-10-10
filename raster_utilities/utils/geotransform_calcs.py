import math
from ..utils.logger import logMessage
from ..aggregation.aggregation_values import AggregationTypes
def calcAggregatedProperties(method, inRasterProps,
                             aggFactor=None, outShape=None, outResolution=None):
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

    assert aggFactor is not None or outShape is not None or outResolution is not None

    if method == "factor" or method == AggregationTypes.FACTOR:
        # the simplest version, output is just n times coarser than before
        # if it's not a clean match then we will expand to get everything
        # Implies the same pixel shape as before just larger
        assert isinstance(aggFactor, int) and aggFactor > 0
        outXSize = int(math.ceil(1.0 * inRasterProps.width / aggFactor))
        outYSize = int(math.ceil(1.0 * inRasterProps.height / aggFactor))
        if inRasterProps.width % aggFactor != 0 or inRasterProps.height % aggFactor != 0:
            # this will tend to trigger with irrational cell sizes such as 1/120 so we should probably
            # put a tolerance on it
            logMessage("warning, input size was not clean multiple of factor, "+
                       "output will have 1 cell greater extent", "warning")
        outputGT = (inRasterProps.gt[0], inRasterProps.gt[1] * aggFactor, 0.0,
                    inRasterProps.gt[3], 0.0, inRasterProps.gt[5] * aggFactor)

    elif method == "size" or method == AggregationTypes.SIZE:
        # we will specify the required pixel dimensions of the output: aspect ratio may change
        # this is generally the best method to use if we're working with non-integer cell resolutions
        outXSize = outShape[1]
        outYSize = outShape[0]
        # now we need to calculate the resolution this implies (for a maintained extent)
        inputHeightPx = inRasterProps.height
        inputWidthPx = inRasterProps.width
        if 1.0 * inputHeightPx / inputWidthPx != 1.0 * outYSize / outXSize:
            # the coverage in ground extent will be the same, so the pixels themselves must change shape
            logMessage("warning, output size is different proportion to input, "+
                       "cells will change shape", "warning")
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
            logMessage("warning, input had non-square cells, "+
                       "resolution mode creates square cells so they will change shape", "warning")
        if isinstance(outResolution, str):
            # use the hardcoded resolutions for the 3 main MAP resolutions to avoid
            # cocking about with irrational numbers not multiplying / dividing cleanly
            xExtent = inXSize * inResX
            yExtent = inYSize * -inResY
            if outResolution.lower().startswith("1k"):
                outXSize = int(120 * xExtent)
                outYSize = int(120 * yExtent)
                outResolution = 0.008333333333333
            elif outResolution.lower().startswith("5k"):
                outXSize = int(24 * xExtent)
                outYSize = int(24 * yExtent)
                outResolution = 0.041666666666667
            elif outResolution.lower().startswith("10k"):
                outXSize = int(12 * xExtent)
                outYSize = int(12 * yExtent)
                outResolution = 0.08333333333333
            else:
                logMessage("Unknown string resolution description!", "error")
                assert False
        else:
            # specify a desired output resolution, implies the same in both directions (square pixels)
            xFactor = round(1.0 * outResolution / inGT[1], 8)
            yFactor = round(-1.0 * outResolution / inGT[5], 8)
            outX_exact = 1.0 * inXSize / xFactor
            outY_exact = 1.0 * inYSize / yFactor
            outXSize = math.ceil(outX_exact)
            outYSize = math.ceil(outY_exact)
            if outX_exact != outXSize or outY_exact != outYSize:
                logMessage("warning, specified resolution was not clean multiple of input, "+
                           "output will have 1 cell greater extent", "warning")
        outputGT = (inGT[0], outResolution, 0.0, inGT[3], 0.0, -outResolution)
    else:
        logMessage("Unknown aggregation type requested, valid values are 'size', 'resolution', 'factor'",
                   "error")
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

def CalculatePixelLims_GlobalRef(inGT, longitudeLims, latitudeLims):
    '''Returns pixel coords of a given AOI in degrees, as they *would* be in a global image

    The resolution is taken from the input geotransform but the origin relative to which the
    coords are calculated is taken as -180,90 regardless of what is in the geotransform'''

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
    OverallNorthLimit = 90
    OverallWestLimit = -180
    x0 = int((WestLimitOut - OverallWestLimit) / xRes)
    x1 = int(((EastLimitOut - OverallWestLimit) / xRes) + 0.5)
    y0 = int((OverallNorthLimit - NorthLimitOut) / yRes)
    y1 = int(((OverallNorthLimit - SouthLimitOut) / yRes) + 0.5)

    return ((x0, x1), (y0, y1))

def CalculatePixelLims(inGT, longitudeLims, latitudeLims):
    '''Returns pixel coords of a given AOI in degrees, in an image with the given geotransform

    Returns pixel coordinates as ((xMin, xMax), (yMax, yMin))'''

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
    OverallNorthLimit = inGT[3]
    OverallWestLimit = inGT[0]

    x0 = int((WestLimitOut - OverallWestLimit) / xRes)
    x1 = int(((EastLimitOut - OverallWestLimit) / xRes) + 0.5)
    y0 = int((OverallNorthLimit - NorthLimitOut) / yRes)
    y1 = int(((OverallNorthLimit - SouthLimitOut) / yRes) + 0.5)

    return ((x0, x1), (y0, y1))
