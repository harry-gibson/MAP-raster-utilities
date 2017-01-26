import numpy as np
from libc.math cimport sqrt
cimport cython

cpdef matchToCoastline(long long[:,::1] data, char[:,::1] landMask, long long _NDV = -9999,
                        char applyClip = 1, char applyFill = 1,
                        int useNearestNPixels=1,
                        int searchPixelRadius = 10):
    '''
    Matches a data image to a mask image by removing and/or adding pixels from the data.

    Fills in gaps in a data grid, relative to a coastline or other limits layer to create an
    output dataset that is flush to the coast, and clips (removes) data outside of the coast.
    Both filling and clipping steps are optional - i.e. either or both can be done.

    For use when an input dataset has been created with a different coastline and now we require
    a dataset that matches MAP's One True Coastline (TM).

    Clipping, if done, is done before filling so the values removed do not participate in calculating
    a fill value for missing inland pixels. The clipping step replaces any data pixels that fall
    outside of the mask's "land" area with NoData.

    The function returns an array representing the locations where the fill process failed as no data was
    found within range. The data array itself is modified in-place (ref parameter).

    Gaps are filled with the mean of the nearest n data-containing pixels, where n is the
    useNearestNPixels parameter. By default n=1 to just the nearest value, for integer data.
    These data pixels must be found within the specified searchPixelRadius. If fewer pixels are
    found within the specified radius,  a fill is still created if at least one is found.

    The fill process is iterative i.e. fill values for one pixel can then be used in generating
    a fill for adjacent pixels. This will lead to "smearing" (down and right) and so this algorithm
    isn't appropriate for filling large areas, and generally it isn't appropriate for continuous
    data of any kind. Rather, it's intended for matching things like administrative units to a
    different coastline template.

    The mask should have value 1 for "land" (areas that should have data), and 0 for "sea"
    (areas that should have nodata). The data and mask arrays must be of the same shape.

    See also populationReallocator which has a similar, but somewhat different purpose, for population
    data where totals must be maintained. This version doesn't maintain totals.
    '''

    cdef:
        Py_ssize_t xShapeIn, yShapeIn
        Py_ssize_t xIn, yIn, xNbr, yNbr, nbrIndex_prv
        float localValue
        int[:,::1] nbrIntCoords
        char[:,::1] failedLocs
        int filledCells, failedFills, clippedCells
        char reallocatedOK
        int _MAX_NEIGHBOURS_TO_CHECK
        int reallocationSum_prv
        int reallocationCount_prv

    yShapeIn = data.shape[0]
    xShapeIn = data.shape[1]
    assert landMask.shape[0] == yShapeIn
    assert landMask.shape[1] == xShapeIn

    # Generate the table of coordinate pairs which when followed in order specify the offsets for a spiral
    # search pattern.
    diam = searchPixelRadius * 2 + 1
    inds = np.indices([diam,diam]) - searchPixelRadius
    distTmp = np.sqrt((inds ** 2).sum(0))
    npTmpTable = ((inds.T).reshape(diam**2, 2))
    npTmpTable = np.append(npTmpTable, distTmp.ravel()[:,None], axis=1)
    # sort the table by distance then x then y (the arguments are last-sort-first)
    order = np.lexsort((npTmpTable[:,1], npTmpTable[:,0], npTmpTable[:,2]))

    # npTmpTable has shape ((radius*2+1)**2, 3) and represents the offset coordinates of all cells in
    # a square of size radius*2+1 around a central point, sorted by increasing distance from the centre.
    # npTmpTable[:,0] is x-offsets, npTmpTable[:,1] is y-offsets,
    # npTmpTable[:,2] is distance.
    npTmpTable = np.take(npTmpTable,order,axis=0)

    # Transpose this into an array in C-contiguous layout and with three rows and many columns,
    # which makes it more cache-friendly for the accesses we will be making.
    # Only transfer the values relating to the cells within the desired radius (cut the corners
    # off the square to make a circle).
    nbrTable = np.copy((npTmpTable[npTmpTable[:,2] <= searchPixelRadius]).T, order='c')

    # transfer this numpy array to  to a native C object (memoryview)
    # and while doing so, cast the columns that will be used as array indices
    # to int type once here. This saves a great deal of time compared to
    # casting repeatedly inside the inner loop
    nbrIntCoords = np.asarray(nbrTable[0:2,:]).astype(np.int32)
    _MAX_NEIGHBOURS_TO_CHECK = nbrIntCoords.shape[1]

    filledCells = 0
    failedFills = 0
    clippedCells = 0

    failedLocs = np.zeros_like(landMask)

    # run the clip step, if required
    if applyClip:
        for yIn in range(yShapeIn):
            for xIn in range(xShapeIn):
                if landMask[yIn, xIn] == 0:
                    if data[yIn, xIn] != _NDV:
                        clippedCells += 1
                        data[yIn, xIn] = _NDV

    # run the spread-and-fill step, if required
    if applyFill:
        for yIn in range(yShapeIn):
            for xIn in range(xShapeIn):
                if landMask[yIn, xIn] == 0 or data[yIn, xIn] != _NDV:
                    # there is nothing to do, because either we are in the sea (and clipping
                    # data out of here was optionally done previously), or
                    # we are on land with good data
                    continue
                # otherwise, we are on land but have a nodata pixel: we need to fill it
                reallocatedOK = 0
                reallocationCount_prv = 0
                reallocationSum_prv = 0
                for nbrIndex_prv in range(1, _MAX_NEIGHBOURS_TO_CHECK):
                    if reallocationCount_prv == useNearestNPixels:
                        break

                    # use int-type coords array to avoid cast op in tight loop: yes this does become significant
                    xNbr = xIn + nbrIntCoords[0, nbrIndex_prv]
                    yNbr = yIn + nbrIntCoords[1, nbrIndex_prv]
                    if (xNbr >= 0 and xNbr < xShapeIn and
                        yNbr >= 0 and yNbr < yShapeIn and
                        data[yNbr, xNbr] != _NDV):
                        # NB we allow a sea pixel with data to supply a fill value
                        # unless they have been clipped out previously
                        reallocationSum_prv += data[yNbr, xNbr]
                        reallocationCount_prv += 1
                if reallocationCount_prv > 0:
                    # we modify the input. So, this fill value may be found and used
                    # when looking for neighbours for adjacent pixels. This will lead
                    # to "smearing" of values in an easterly and southerly direction.
                    # Hence, not suitable for filling large areas!
                    data[yIn, xIn] = reallocationSum_prv / reallocationCount_prv
                    filledCells += 1
                else:
                    failedFills += 1
                    failedLocs[yIn, xIn] = 1

    print ("Clipped {0!s} data cells that were outside provided limits".format(
        clippedCells))
    print ("Filled {0!s} total cells within limits from nearby data".format(
        filledCells))
    print ("Failed to fill {0!s} total cells within limits due to no data cells in range".format(
        failedFills))
    return failedLocs



cpdef reallocateToUnmasked(float[:,::1] data, char[:,::1] landMask, float _NDV = -9999,
                           char fillLandWithZero = 1,
                           char clipZerosAtSea = 1,
                           int searchPixelRadius = 100,
                           char deleteDespiteFailure = 0):
    '''
    Reallocates data falling in masked area to nearest non-masked pixel

    For use in preparing population datasets for MAP use with standardised land-sea template.

    When matching to a coastline template, for population data we cannot just clip / spread the raster, as
    this will change the total population.

    Instead, any population falling in pixels that are "sea" according to MAP's One True Coastline (TM)
    must be forcibly relocated Bikini-Atoll-style to the nearest "land" pixel according to MAP's
    One True Coastline (TM), in order to maintain population counts.

    Input data must be a float array. Input mask must be a byte array of the same shape as
    the data array, with a value of 1 on "land" (unmasked areas), and any other value
    elsewhere.

    The input data array is modified in-place. The returned object is a new array flagging
    (with a value of 1) locations where the reallocation failed because there was no
    unmasked (land) pixel within the search radius. At these locations, the data will be
    unmodified despite being in the sea.
    '''
    cdef:
        Py_ssize_t xShapeIn, yShapeIn
        Py_ssize_t xIn, yIn, xNbr, yNbr, nbrIndex_prv
        float localValue
        int[:,::1] nbrIntCoords
        char[:,::1] failedLocs
        int reallocatedCells, failedReallocations, clippedZeros 
        float failedReallocationPop,  reallocatedTotalPop
        char reallocatedOK
        int _MAX_NEIGHBOURS_TO_CHECK

    yShapeIn = data.shape[0]
    xShapeIn = data.shape[1]
    assert landMask.shape[0] == yShapeIn
    assert landMask.shape[1] == xShapeIn

    # Generate the neighbour spiral search table out to "a bit" further than needed
    diam = searchPixelRadius * 2 + 1
    inds = np.indices([diam,diam]) - searchPixelRadius
    distTmp = np.sqrt((inds ** 2).sum(0))
    npTmpTable = ((inds.T).reshape(diam**2, 2))
    npTmpTable = np.append(npTmpTable, distTmp.ravel()[:,None], axis=1)
    # sort the table by distance then x then y (the arguments are last-sort-first)
    order = np.lexsort((npTmpTable[:,1],npTmpTable[:,0],npTmpTable[:,2]))

    # npTmpTable has shape ((radius*2+1)**2, 3) and represents the offset coordinates of all cells in
    # a square of size radius*2+1 around a central point, sorted by increasing distance from the centre.
    # npTmpTable[:,0] is x-offsets, npTmpTable[:,1] is y-offsets,
    # npTmpTable[:,2] is distance.
    npTmpTable = np.take(npTmpTable,order,axis=0)

     # Transpose this into an array in C-contiguous layout and with three rows and many columns,
    # which makes it more cache-friendly for the accesses we will be making.
    # Only transfer the values relating to the cells within the desired radius (cut the corners
    # off the square to make a circle).
    nbrTable = np.copy((npTmpTable[npTmpTable[:,2] <= searchPixelRadius]).T,order='c')

    # transfer this numpy array to  to a native C object (memoryview)
    # and while doing so, cast the columns that will be used as array indices
    # to int type once here. This saves a great deal of time compared to
    # casting repeatedly inside the inner loop
    nbrIntCoords = np.asarray(nbrTable[0:2,:]).astype(np.int32)
    _MAX_NEIGHBOURS_TO_CHECK = nbrIntCoords.shape[1]

    reallocatedCells = 0
    clippedZeros = 0
    reallocatedTotalPop = 0
    failedReallocations = 0
    failedReallocationPop = 0

    failedLocs = np.zeros_like(landMask)

    for yIn in range (yShapeIn):
        for xIn in range(xShapeIn):
            if landMask[yIn, xIn] == 1:
                # we are on land, so no need to reallocate anything
                # however we might want to set it to zero to stop the R-boys getting
                # stressed about seeing nodata inland
                if data[yIn, xIn] == _NDV and fillLandWithZero:
                    data[yIn, xIn] = 0
                continue
            if data[yIn, xIn] == _NDV:
                # we're in the sea, but there's nothing to reallocate
                continue
            if data[yIn, xIn] == 0:
                # we're in the sea, and there's nothing to reallocate but...
                if clipZerosAtSea:
                    # ... we might want to replace zeros at-sea with no-data to stop the R-boys
                    # getting stressed about seeing people at sea even though the number of
                    # people in question is zero
                    data[yIn, xIn] = _NDV
                    clippedZeros += 1
                continue
            # otherwise, we're at sea but have non-zero data. Find nearest land pixel to reallocate it to.
            reallocatedOK = 0
            for nbrIndex_prv in range(1, _MAX_NEIGHBOURS_TO_CHECK):
                # use int-type coords array to avoid cast op in tight loop
                xNbr = xIn + nbrIntCoords[0, nbrIndex_prv]
                yNbr = yIn + nbrIntCoords[1, nbrIndex_prv]
                if (xNbr >= 0 and xNbr < xShapeIn and
                    yNbr >= 0 and yNbr < yShapeIn and
                    landMask[yNbr, xNbr] == 1):
                    # we've found a land pixel, move the value from here to there
                    if data[yNbr, xNbr] == _NDV or data[yNbr, xNbr] < 0:
                        data[yNbr, xNbr] = data[yIn, xIn]
                    else:
                        data[yNbr, xNbr] += data[yIn, xIn]
                    data[yIn, xIn] = _NDV
                    reallocatedOK = 1
                    reallocatedCells += 1
                    reallocatedTotalPop += data[yNbr, xNbr]
                    break
            if reallocatedOK == 0:
                failedReallocations += 1
                failedReallocationPop += data[yIn, xIn]
                if deleteDespiteFailure:
                    data[yIn, xIn] = _NDV
                failedLocs[yIn, xIn] = 1

    print ("Reallocated {0!s} total pop from {1!s} cells to nearby land cell".format(
        reallocatedTotalPop,reallocatedCells))
    print ("Clipped (set to nodata) {0!s} zero-value cells in the sea".format(clippedZeros))
    print ("Failed to reallocate {0!s} total pop from {1!s} cells to nearby land cell".format(
        failedReallocationPop, failedReallocations))
    return np.asarray(failedLocs)
















































