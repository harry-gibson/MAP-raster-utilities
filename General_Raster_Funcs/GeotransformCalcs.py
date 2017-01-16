import math
def calcAggregatedProperties(method, inShape, inGT,
                             aggFactor=None, outShape=None, outResolution=None):
    ''' Given an input raster, get the post-aggregation geotransform and dimensions

    The aggregation can be specified as either a factor, a desired output shape,
    or a desired output resolution.

    Returns a 2-tuple where first item is the geotransform (itself a 6-tuple,
    according to the GDAL standard, and the second item is the shape (a 2-tuple,
    according to numpy standard)

    calcAggregatedProperties("size", (17400,43200), origGT, None, (4320, 8640), None)

    calcAggregatedProperties("resolution", (17400,43200), origGT, None, None, 0.04166666666665)

    calcAggregatedProperties("factor", (21600,43200), origGT, 10, None, None)


    '''
    assert len(inShape) == 2
    assert len(inGT) == 6
    assert aggFactor is not None or outShape is not None or outResolution is not None

    if method == "factor":
        # the simplest version, output is just n times coarser than before
        # if it's not a clean match then we will expand to get everything
        # Implies the same pixel shape as before just larger
        assert isinstance(aggFactor, int) and aggFactor > 0
        outXSize = int(math.ceil(1.0 * inShape[1] / aggFactor))
        outYSize = int(math.ceil(1.0 * inShape[0] / aggFactor))
        if inShape[1] % aggFactor != 0 or inShape[0] % aggFactor != 0:
            # this will tend to trigger with irrational cell sizes such as 1/120 so we should probably
            # put a tolerance on it
            print "warning, input size was not clean multiple of factor, output will have 1 cell greater extent"
        outputGT = (inGT[0], inGT[1] * aggFactor, 0.0, inGT[3], 0.0, inGT[5] * aggFactor)

    elif method == "size":
        # we will specify the required pixel dimensions of the output: aspect ratio may change
        # this is generally the best method to use if we're working with non-integer cell resolutions
        outXSize = outShape[1]
        outYSize = outShape[0]
        # now we need to calculate the resolution this implies (for a maintained extent)
        inputHeightPx = inShape[0]
        inputWidthPx = inShape[1]
        if 1.0 * inputHeightPx / inputWidthPx != 1.0 * outYSize / outXSize:
            # the coverage in ground extent will be the same, so the pixels themselves must change shape
            print "warning, output size is different proportion to input, cells will change shape"
        inputXMin = inGT[0]
        inputYMax = inGT[3]
        inputXMax = inputXMin + inGT[1] * inputWidthPx
        # y resolution is negative as origin is top left
        inputYMin = inputYMax + inGT[5] * inputHeightPx
        inputHeightProj = inputYMax - inputYMin
        inputWidthProj = inputXMax - inputXMin
        outputResX = inputWidthProj / outXSize
        outputResY = inputHeightProj / outYSize
        outputGT = (inGT[0], outputResX, 0.0, inGT[3], 0.0, -outputResY)

    elif method == "resolution":
        # specify a desired output resolution, implies the same in both directions (square pixels)
        xFactor = round(1.0 * outResolution / inGT[1], 8)
        yFactor = round(-1.0 * outResolution / inGT[5], 8)
        outX_exact = 1.0 * inShape[1] / xFactor
        outY_exact = 1.0 * inShape[0] / yFactor
        outXSize = math.ceil(outX_exact)
        outYSize = math.ceil(outY_exact)
        if outX_exact != outXSize or outY_exact != outYSize:
            print "warning, specified resolution was not clean multiple of input, output will have 1 cell greater extent"
        outputGT = (inGT[0], outResolution, 0.0, inGT[3], 0.0, -outResolution)
    else:
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

    topLeftLongOut = topLeftLongIn + xPixelLims[0] * resX
    topLeftLatOut = topLeftLatIn + yPixelLims[0] * resY

    clippedGT = (topLeftLongOut, resX, 0.0, topLeftLatOut, 0.0, resY)
    return clippedGT


def CalculatePixelLims(inGT, longitudeLims, latitudeLims):
    '''Return the pixel coords of a given AOI in degrees

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
