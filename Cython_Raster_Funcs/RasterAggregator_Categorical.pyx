#cython: /openmp
# %%cython --compile-args=/openmp --link-args=/openmp --force
# author: Harry Gibson
# date: 2017-01-06
# description: Cython class for aggregating categorical rasters


cimport cython
import numpy as np
cimport openmp
from cython.parallel import parallel, prange

cdef class RasterAggregator_Categorical:
    ''' Aggregates a categorical raster grid by a specified factor (e.g. 10 to aggegate 500m to 5km data)

     Returns a dictionary containing two three dimensional grids and one two dimensional grid
     at the aggregated resolution, each representing a different summary of the source
     pixels covered by each output pixel, namely:
        {
          "fractions" : (3D array, one z-level for each category,
                          containing fraction of that category as an
                          integer (rounded) percentage),
          "likeadjacencies" : (3D array, one z-level for each category,
                          containing like-adjacency of that category),
          "majority" : (2D array, containing the value of the modal category
                          at that location),
          "valuemap" : (1D array, containing the category value of each corresponding
                      z-level in the fractions and likeadjacencies arrays. For
                      example [7, 1, 4] means the 2d array at the 0th position of the
                      fractions array z dimension, contains the fraction of the cells
                      that were covered by category 7)
        }

     The fraction and like adjacency outputs have one image (in the z dimension) for each category
     (value) of the input raster.

     Note that this means the output will have as many "layers" in the z dimension as there are unique
     values in the input data. Thus this is only recommended where there are a small number of unique
     values, or the output grids are small, due to the large memory implications. This is the main reason
     why the input data type is restricted to an 8-bit unsigned integer raster.

     The input grid dimensions do not necessarily need to be exact multiples of the output
     dimensions, but this is generally recommended as otherwise the output pixels will be
     comprised of varying numbers of input pixels (each input is allocated to only one output
     based on the location of its left/top corner).

     The output majority grid will use 255 as nodata value unless another value is provided
     so this value should not appear in the input data.

     Like-adjacencies are calculated using the double-count method. That is, to calculate the
     contribution for a given pixel, the pixels above, below, left, and right of it are checked.
     But when the contributions of those pixels themselves are also checked, the original pixel will
     be a contributing neighbour of them, too.
    '''

    cdef:
       Py_ssize_t xShapeIn, yShapeIn, tileXShapeIn, tileYShapeIn
       Py_ssize_t xShapeOut, yShapeOut
       double xFact, yFact
       float proportion

    cdef:
        Py_ssize_t xIn, yIn, xOut, yOut, catNum
        int yBelow, yAbove, xLeft, xRight


    cdef:
        unsigned char nCategories
        float _fltNDV
        unsigned char _byteNDV
        unsigned char _hasNDV
        float[:,:,::1] outputFracArr
        float[:,:,::1] outputLikeAdjArr
        unsigned char [:,::1] outputMajorityArr
        int [::1] valueMap

    cdef:
        float [:,::1] tmpMajorityPropArr
        char[:,::1] _coverageArr

    def __cinit__(self,
                  Py_ssize_t xSizeIn, Py_ssize_t ySizeIn,
                  Py_ssize_t xSizeOut, Py_ssize_t ySizeOut,
                  unsigned char nCategories, float fltNDV = -9999, byteNDV = None):
        assert xSizeIn > xSizeOut
        assert ySizeIn > ySizeOut

        self.xShapeIn = xSizeIn
        self.yShapeIn = ySizeIn
        self.xShapeOut = xSizeOut
        self.yShapeOut = ySizeOut

        self.xFact = <double>self.xShapeIn / self.xShapeOut
        self.yFact = <double>self.yShapeIn / self.yShapeOut

        self._fltNDV = fltNDV
        # categorical rasters often don't contain nodata
        if byteNDV is not None and isinstance(byteNDV, int) and 0 <= byteNDV <= 255:
            self._byteNDV = byteNDV
            self._hasNDV = 1
        else:
            self._hasNDV = 0

        self.nCategories = nCategories
        self.valueMap = np.zeros(shape = (nCategories), dtype = np.int32)
        self.valueMap[:] = -1
        self.outputFracArr = np.zeros(shape = (nCategories, self.yShapeOut, self.xShapeOut),
                                      dtype = np.float32)
        self.outputLikeAdjArr = np.zeros(shape = (nCategories, self.yShapeOut, self.xShapeOut),
                                         dtype = np.float32)
        self.outputMajorityArr = np.zeros(shape = (self.yShapeOut, self.xShapeOut),
                                          dtype = np.uint8)
        self.tmpMajorityPropArr = np.zeros(shape = (self.yShapeOut, self.xShapeOut),
                                           dtype = np.float32)
        self._coverageArr = np.zeros(shape = (self.yShapeOut, self.xShapeOut), dtype = np.byte)

    @cython.boundscheck(False)
    @cython.cdivision(True)
    @cython.wraparound(False)
    cpdef addTile(self, unsigned char[:,::1] data, Py_ssize_t xOffset, Py_ssize_t yOffset):
        cdef:
            # shape of the data we receive
            Py_ssize_t tileYShapeIn, tileXShapeIn
            # coords of neighbours in global grid
            int yBelowGlobal, yAboveGlobal, xLeftGlobal, xRightGlobal
            # coords of neighbours in the received tile
            int yBelowTile, yAboveTile, xLeftTile, xRightTile
            # current pixel coords in global and tile coords
            Py_ssize_t xInGlobal, xInTile, yInGlobal, yInTile
            unsigned char localValue
            unsigned char nNeighbours
            unsigned char catNum
            float likeAdjProp
            Py_ssize_t xOut, yOut
            # how much of an output cell does each input cell account for
            float proportion
            unsigned char catsOk
            int valuePos
            Py_ssize_t i

        tileXShapeIn = data.shape[1]
        tileYShapeIn = data.shape[0]

        # how much of an output cell does each input cell account for
        proportion = 1.0 / (self.xFact * self.yFact)

        with nogil, parallel():
            for yInTile in prange (tileYShapeIn):
                catsOk = 1
                yInGlobal = yInTile + yOffset

                yAboveGlobal = yInGlobal - 1
                if yInGlobal == 0:
                    yAboveGlobal = -1
                yAboveTile = yInTile - 1
                if yInTile == 0:
                    yAboveTile = -1

                yBelowGlobal = yInGlobal + 1
                if yInGlobal == self.yShapeIn - 1:
                    yBelowGlobal = -1
                yBelowTile = yInTile + 1
                if yInTile == tileYShapeIn - 1:
                    yBelowTile = -1

                yOut = <int> (yInGlobal / self.yFact)
                # yeah i know that it's an unsigned char but we won't actually use this value,
                # these assignments are to cause the cython converter to make these thread-local
                localValue = -1
                xOut = -1
                valuePos = -1
                i = 0

                for xInTile in range(tileXShapeIn):
                    xInGlobal = xInTile + xOffset
                    # make these private by assignment
                    nNeighbours = 0
                    likeAdjProp = 0

                    xLeftGlobal = xInGlobal - 1
                    if xInGlobal == 0:
                        xLeftGlobal = -1
                    xLeftTile = xInTile - 1
                    if xInTile == 0:
                        xLeftTile = -1

                    xRightGlobal = xInGlobal + 1
                    if xInGlobal == self.xShapeIn - 1:
                        xRightGlobal = -1
                    xRightTile = xInTile + 1
                    if xInTile == tileXShapeIn - 1:
                        xRightTile = -1

                    xOut = <int> (xInGlobal / self.xFact)

                    localValue = data[yInTile, xInTile]
                    if self._hasNDV == 1 and localValue == self._byteNDV:
                        # bit 1 indicates covered by some input grid even if it was nodata
                        self._coverageArr[yOut, xOut] = self._coverageArr[yOut, xOut] | 1
                        # don't do anything
                        continue

                    # determine the z-position in the output stack for the fractional / like-adj
                    # grids for this category value
                    valuePos = -1
                    for i in range(self.nCategories):
                        if self.valueMap[i] == localValue:
                            valuePos = i
                            break
                    # if we haven't seen this value yet, add it
                    if valuePos == -1:
                        for i in range(self.nCategories):
                            if self.valueMap[i] == -1:
                                self.valueMap[i] = localValue
                                valuePos = i
                                break
                    # if we failed to add it, we must have more categories in the data
                    # than we bargained for
                    if valuePos == -1:
                        catsOk = 0
                        break

                    # bit 2 indicates covered by a data pixel
                    self._coverageArr[yOut, xOut] = self._coverageArr[yOut, xOut] | 2

                    # the fraction is straightforward, just the proportion of the output cell
                    # covered by this one
                    # reminder: inplace operator (+=) causes it to be treated as a reduction
                    # (shared) variable by the cython translation
                    self.outputFracArr[valuePos, yOut, xOut] += proportion

                    # the like adjacency contribution of a given incoming cell
                    # is less straightforward because it depends on neighbours and at
                    # edges of a tile we don't have access to all neighbours of incoming data
                    # so we just calculate it based on how many neighbours we do have
                    # (this isn't totally accurate of course)
                    if yAboveTile >= 0:
                        nNeighbours = nNeighbours + 1
                        if data[yAboveTile, xInTile] == localValue:
                            likeAdjProp += 1
                    if xRightTile >= 0:
                        nNeighbours = nNeighbours + 1
                        if data[yInTile, xRightTile] == localValue:
                            likeAdjProp += 1
                    if yBelowTile >= 0:
                        nNeighbours = nNeighbours + 1
                        if data[yBelowTile, xInTile] == localValue:
                            likeAdjProp += 1
                    if xLeftTile >= 0:
                        nNeighbours = nNeighbours + 1
                        if data[yInTile, xLeftTile] == localValue:
                            likeAdjProp += 1

                    self.outputLikeAdjArr[valuePos, yOut, xOut] += (
                        (likeAdjProp / nNeighbours) * proportion)
                if catsOk == 0:
                    break

        if catsOk == False:
            raise Exception("More category values were encountered than were specified. Cannot continue!")

    @cython.boundscheck(False)
    @cython.cdivision(True)
    @cython.wraparound(False)
    cdef finalise(self):
        cdef:
            Py_ssize_t xOut, yOut
            float iscomplete = 1

        # replace output values in fraction and like-adjacency stacks with nodata where appropriate
        for catNum in range (self.nCategories):
            # catNum here is actually just the position in the stack not the actual value
            yOut = -1
            for yOut in range (self.yShapeOut):
                xOut = -1
                for xOut in range (self.xShapeOut):
                    # check whether there's anywhere that hasn't been covered by an input grid
                    # (no matter whether that was data or nodata)
                    if self._coverageArr[yOut, xOut] & 3 == 0:
                        iscomplete = 0

                    # update the majority-class grid if this class' fraction is the highest yet
                    if self.outputFracArr[catNum, yOut, xOut] > self.tmpMajorityPropArr[yOut, xOut]:
                        self.tmpMajorityPropArr[yOut, xOut] = self.outputFracArr[catNum, yOut, xOut]
                        self.outputMajorityArr[yOut, xOut] = self.valueMap[catNum]

                    # for the like adjacency grids, output a value if we've had any input data here;
                    # but output nodata if we haven't.
                    # The arrays are initialised to zero so we need to change those positions to ndv.
                    # Use the fraction (class proportion) grids to check this
                    # (non-zero implies we had a data pixel)
                    if self.outputFracArr[catNum, yOut, xOut] > 0:
                        # we've had at least one data pixel of this class at this location
                        # so we can calculate a like adjacency
                        self.outputLikeAdjArr[catNum, yOut, xOut] = (
                            self.outputLikeAdjArr[catNum, yOut, xOut] / self.outputFracArr[catNum, yOut, xOut]
                        )
                        # also, convert the fraction to a percentage so we can return as int type
                        self.outputFracArr[catNum, yOut, xOut] *= 100
                        # so we can cast (truncate) rather than round
                        self.outputFracArr[catNum, yOut, xOut] += 0.5
                    else:
                        # no input pixels of this class at this output loc: output ndv for like-adjacency
                        self.outputLikeAdjArr[catNum, yOut, xOut] = self._fltNDV
                        # for the fraction (class proportion) grids, output ndv if we haven't
                        # had any input data of _any_ class here (if we have, but of a different
                        # class, then the default 0 is correct for this fraction grid)
                        if self._coverageArr[yOut, xOut] & 2 == 0:
                            self.outputFracArr[catNum, yOut, xOut] = self._fltNDV

        # replace output values in majority class grid with nodata where appropriate
        for yOut in range (self.yShapeOut):
            for xOut in range(self.xShapeOut):
                # if an output pixel has not been covered by any true data then set it to ndv
                if self._coverageArr[yOut, xOut] & 2 == 0:
                    if self._hasNDV:
                        self.outputMajorityArr[yOut, xOut] = self._byteNDV
                    else:
                        self.outputMajorityArr[yOut, xOut] = 255

        if not iscomplete:
            print "Warning, generating a result without having received input data for full extent"
            return False
        return True

    cpdef GetResults(self):
        '''If input is complete, returns an object containing 'fractions', 'likeadjacencies', 'majority' '''
        if not self.finalise():
            return None
        return {
            "fractions": np.asarray(self.outputFracArr).astype(np.uint8),
            "likeadjacencies": np.asarray(self.outputLikeAdjArr),
            "majority": np.asarray(self.outputMajorityArr), #.astype(np.uint8)
            "valuemap": np.asarray(self.valueMap)
        }
