#cython: /openmp
# %%cython --compile-args=/openmp --link-args=/openmp --force
# author: Harry Gibson
# date: 2017-01-06
# description: Cython class for aggregating categorical rasters


# This file contains Cython code so must be translated (to C) and compiled before it can be used in Python.
# To do this simply run "python setup.py build_ext --inplace" (assuming you have cython installed)


cimport cython
import numpy as np
cimport openmp
from cython.parallel import parallel, prange
from raster_utilities.aggregation.aggregation_values import CategoricalAggregationStats as catstats
from ....utils.logger import MessageLogger
cdef class Categorical_Aggregator:
    ''' Aggregates a categorical raster grid to a coarser resolution (i.e. smaller pixel dimensions)

     Returns a dictionary containing two three dimensional grids and one two dimensional grid
     at the aggregated resolution, each representing a different summary of the source
     pixels covered by each output pixel, namely:
        {
          "fractions" : (3D integer array, one z-level for each category value,
                          containing fraction of that category in each cell as an
                          integer (rounded) percentage),
          "likeadjacencies" : (OPTIONAL) (3D float array, one z-level for each category value,
                          containing like-adjacency of that category in each cell),
          "majority" : (2D integer array, containing the value of the modal category
                          at that location),
          "valuemap" : (1D array, containing the category value of each corresponding
                      z-level in the fractions and likeadjacencies arrays. For
                      example [7, 1, 4] means the 2d array at the 0th position of the
                      fractions array z dimension, contains the fraction of the cells
                      that were covered by category 7, the one at the 1st position is
                      category 1, and the 2nd position is the category 4)
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

     The aggregation is built up by adding tiles of the input data using the addTile method. This way a grid
     can be aggregated that is much too large to fit in memory. The required output and total input sizes must be
     specified at instantiation. Input tiles should be provided to cover the whole extent of the specified output -
     if no data is available for some areas then just add a tile of nodata. If this isn't done then no output will
     be generated (GetResults will return None).

    '''

    cdef:
        Py_ssize_t xShapeOut, yShapeOut
        Py_ssize_t xShapeIn, yShapeIn, tileXShapeIn, tileYShapeIn
        Py_ssize_t xSubpixelOffsetIn, ySubpixelOffsetIn

        double xFact, yFact, xFactCalc, yFactCalc
        float proportion

        Py_ssize_t xIn, yIn, xOut, yOut
        int yBelow, yAbove, xLeft, xRight

        unsigned char nCategories
        unsigned char gotCategories
        float _fltNDV
        unsigned char _byteNDV
        unsigned char _hasNDV
        float[:,:,::1] outputFracArr
        float[:,:,::1] outputLikeAdjArr
        unsigned char [:,::1] outputMajorityArr
        int [::1] valueMap
        int [::1] sortedValueMap
        float [:,::1] tmpMajorityPropArr
        char[:,::1] _coverageArr
        short[:,::1] _countArr

        unsigned char _doFractions, _doLikeAdj

    def __cinit__(self,
                  Py_ssize_t ySizeIn, Py_ssize_t xSizeIn,
                  Py_ssize_t ySizeOut, Py_ssize_t xSizeOut,
                  Py_ssize_t ySubpixelOffsetIn, Py_ssize_t xSubpixelOffsetIn,
                  double yFactor, double xFactor,
                  categories,
                  unsigned char doLikeAdjacency = 1,
                  float fltNdv = -9999, byteNdv = None):
        assert xSizeIn > xSizeOut
        assert ySizeIn > ySizeOut

        self.xShapeIn = xSizeIn
        self.yShapeIn = ySizeIn
        self.xShapeOut = xSizeOut
        self.yShapeOut = ySizeOut
        self.xFactCalc = <double>self.xShapeIn / self.xShapeOut
        self.yFactCalc = <double>self.yShapeIn / self.yShapeOut
        self.xFact = xFactor
        self.yFact = yFactor

        if xSubpixelOffsetIn < 0 or xSubpixelOffsetIn >= xFactor or ySubpixelOffsetIn < 0 or ySubpixelOffsetIn >= yFactor:
            raise ValueError("sub-pixel offsets must be between zero and the cell aggregation factor")
        self.xSubpixelOffsetIn = xSubpixelOffsetIn
        self.ySubpixelOffsetIn = ySubpixelOffsetIn

        xSizeCheck = (xSizeIn + xSubpixelOffsetIn) / xFactor
        ySizeCheck = (ySizeIn + ySubpixelOffsetIn) / yFactor
        if xSizeCheck > xSizeOut or ySizeCheck > ySizeOut:
            raise ValueError("specified output size is too small")
        if xSizeCheck < xSizeOut-1 or ySizeCheck < ySizeOut-1:
            raise ValueError("specified output size is too big")

        self._fltNDV = fltNdv
        # categorical rasters often don't contain nodata, that's fine
        if byteNdv is not None and (int(byteNdv) == byteNdv) and 0 <= byteNdv <= 255:
            self._byteNDV = byteNdv
            self._hasNDV = 1
        else:
            self._hasNDV = 0

        logger = MessageLogger()
        # categories parameter can be an int number of categories, in which case we will derive them, or
        # a list / ndarray of the values, which must be between 0 and 255
        if (isinstance(categories,int)):
            assert 0 < categories <= 255
            self.nCategories = categories
            self.valueMap = np.zeros(shape = (categories), dtype = np.int32)
            self.valueMap[:] = -1
            logger.logMessage("Building category list, max expected number is "+str(categories))
        else:
            if isinstance(categories, list):
                categories = np.asarray(categories)
            else:
                assert isinstance(categories, np.ndarray)
            assert 0 <= categories.min()
            assert 255 >= categories.max()
            self.valueMap = categories
            self.gotCategories = 1
            self.nCategories = len(categories)
            logger.logMessage("Using provided category list of "+str(len(categories))+" values: "+str(categories))

        self.sortedValueMap = np.zeros(shape=(self.nCategories), dtype=np.int32)
        self.sortedValueMap[:] = -1
        self.outputFracArr = np.zeros(shape = (self.nCategories, self.yShapeOut, self.xShapeOut),
                                      dtype = np.float32)
        if doLikeAdjacency:
            self.outputLikeAdjArr = np.zeros(shape = (self.nCategories, self.yShapeOut, self.xShapeOut),
                                             dtype = np.float32)
            self._doLikeAdj = True
        self.outputMajorityArr = np.zeros(shape = (self.yShapeOut, self.xShapeOut),
                                          dtype = np.uint8)
        self.tmpMajorityPropArr = np.zeros(shape = (self.yShapeOut, self.xShapeOut),
                                           dtype = np.float32)
        self._coverageArr = np.zeros(shape = (self.yShapeOut, self.xShapeOut), dtype = np.byte)
        self._countArr = np.zeros(shape = (self.yShapeOut, self.xShapeOut), dtype = np.int16)

        if self._hasNDV:
            logger.logMessage("Input nodata value is "+str(self._byteNDV))
        else:
            logger.logMessage("Nodata not defined")

    @cython.boundscheck(True)
    @cython.cdivision(True)
    @cython.wraparound(False)
    cpdef addTile(self, unsigned char[:,::1] data, Py_ssize_t xTileOffset, Py_ssize_t yTileOffset):
        '''Add a tile of the input data to the aggregation. Provide the offset of the top left pixel.'''
        cdef:
            # shape of the data we receive
            Py_ssize_t tileYShapeIn, tileXShapeIn

            # coords of neighbours in global grid
            int yBelowGlobal, yAboveGlobal, xLeftGlobal, xRightGlobal
            # coords of neighbours in the received tile
            int yBelowTile, yAboveTile, xLeftTile, xRightTile

            # current pixel coords in global and tile coords
            Py_ssize_t xInGlobal, xInTile, yInGlobal, yInTile, xSubCellOffset, ySubCellOffset
            unsigned char localValue, nNeighbours
            float likeAdjProp
            Py_ssize_t xOut, yOut
            # how much of an output cell does each input cell account for
            #float proportion
            unsigned char catsOk
            int errorCategory
            int valuePos
            Py_ssize_t i

        tileYShapeIn = data.shape[0]
        tileXShapeIn = data.shape[1]
        xSubCellOffset = self.xSubpixelOffsetIn
        ySubCellOffset = self.ySubpixelOffsetIn

        # how much of an output cell does each input cell account for
        #proportion = 1.0 / (self.xFact * self.yFact)
        catsOk = 1
        errorCategory = -1
        with nogil, parallel():
            for yInTile in prange (tileYShapeIn):

                yInGlobal = yInTile + yTileOffset

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

                yOut = <int> ((yInGlobal + ySubCellOffset) / self.yFact)
                # yeah i know that it's an unsigned char but we won't actually use this value,
                # these assignments are to cause the cython converter to make these thread-local
                localValue = -1
                xOut = -1
                valuePos = -1
                i = 0

                for xInTile in range(tileXShapeIn):
                    xInGlobal = xInTile + xTileOffset
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

                    xOut = <int> ((xInGlobal + xSubCellOffset) / self.xFact)

                    localValue = data[yInTile, xInTile]

                    # we will count the number of incoming pixels that count towards each
                    # output cell. Because if the cell sizes aren't a clean multiple, then the
                    # number of inputs per output will vary cyclically.
                    # We count the incoming cells regardless of whether or not they have data.
                    self._countArr[yOut, xOut] += 1

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
                    if valuePos == -1:
                        if self.gotCategories:
                            # we have been given the expected category values but have encountered a different one
                            # for now, error out - though actually we might want to just continue here i.e. ignore
                            # cells with unspecified values maybe treating as nodata
                            catsOk = 0
                            errorCategory = localValue
                            break
                        # if we haven't seen this value yet, and we haven't pre-loaded the list of expected values,
                        # add it to the next available position in valuemap
                        # todo think about whether this is truly safe. not sure - can we end up with valuemap
                        # inconsistent?
                        # i.e. is the with gil sufficient to ensure consistent access to valueMap. I'm not sure it is
                        # as threads can still be preempted. See the finalise method for a backstop check.
                        with gil:
                            for i in range(self.nCategories):
                                if self.valueMap[i] == -1:
                                    self.valueMap[i] = localValue
                                    valuePos = i
                                    break
                    # if we failed to add it, we must have more categories in the data
                    # than we bargained for
                    if valuePos == -1:
                        errorCategory = localValue
                        catsOk = 0
                        break

                    # bit 2 indicates covered by a data pixel
                    self._coverageArr[yOut, xOut] = self._coverageArr[yOut, xOut] | 2

                    # the fraction is straightforward, just the proportion of the output cell
                    # covered by this one. Track as a count and convert to a fraction afterwards
                    # once we know the overall count.
                    # reminder: inplace operator (+=) causes it to be treated as a reduction
                    # (shared) variable by the cython translation
                    self.outputFracArr[valuePos, yOut, xOut] += 1

                    # the like adjacency contribution of a given incoming cell
                    # is less straightforward because it depends on neighbours and at
                    # edges of a tile we don't have access to all neighbours of incoming data
                    # so we just calculate it based on how many neighbours we do have
                    # (this isn't totally accurate of course)
                    if self._doLikeAdj:
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
                            (likeAdjProp / nNeighbours) )
                if catsOk == 0:
                    break

        if catsOk == False:
            if self.gotCategories:
                raise Exception("Encountered unexpected category value of {} - expected {} categories are {}".format(
                    str(errorCategory), self.nCategories, str(np.asarray(self.valueMap))))
            else:
                raise Exception("More category values were encountered than were specified. Cannot continue! "+
                            str(np.asarray(self.valueMap)))

    @cython.boundscheck(False)
    @cython.cdivision(True)
    @cython.wraparound(False)
    cdef finalise(self):
        cdef:
            Py_ssize_t xOut, yOut
            float iscomplete = 1
            float proportion
            int i, catValue, catPosition

        # because i'm not certain that we can validly build the valuemap from the data in a threadsafe way,
        # here we check for errors that might have occurred. The potential risk is that the same value could have
        # been added to the valuemap twice. So check that the valuemap is in fact unique.
        vmNumpy = np.asarray(self.valueMap)
        logger = MessageLogger()
        logger.logMessage("Value map is "+str(vmNumpy))
        vmNumpyExclBlank = vmNumpy[vmNumpy!=-1]
        if len(np.unique(vmNumpyExclBlank)) < len(vmNumpyExclBlank):
            raise ValueError("messed up the value map, that's an error. Please try re-running with the expected" 
                             " value map passed as input. Produced: "+ str(np.asarray(self.valueMap)))

        # normalise the fraction and like-adjacency arrays according to then number of data pixels
        # that went into each output location
        for i in range(self.nCategories):
            yOut = -1
            with nogil, parallel():
                for yOut in prange(self.yShapeOut):
                    xOut = -1
                    for xOut in range(self.xShapeOut):
                        if self._countArr[yOut, xOut] > 0:
                            proportion = 1.0 / self._countArr[yOut, xOut]
                            self.outputFracArr[i, yOut, xOut] *= proportion
                            if self._doLikeAdj:
                                self.outputLikeAdjArr[i, yOut, xOut] *= proportion

        self.sortedValueMap = np.array(sorted(vmNumpy))
        # replace output values in fraction and like-adjacency stacks with nodata where appropriate
        for i in range (self.nCategories):
            # For the majority class output, if there is a tie then we will output whichever class
            # with that tied fraction we hit first.
            # However the order that items got added to valuemap is non-deterministic as we are
            # running with multiple threads. Therefore, if we just go through valuemap in whatever
            # order it is and pick the majority based on what we hit first, the output majority grid
            # can vary between repeated runs.
            # So we will always determine the majority output in sorted order, i.e. output the lowest
            # one first.
            # the value to output
            catValue = self.sortedValueMap[i]
            if catValue == -1:
                continue
            # the position in thje stack
            catPosition = np.argwhere(vmNumpy==catValue)[0]
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
                    if self.outputFracArr[catPosition, yOut, xOut] > self.tmpMajorityPropArr[yOut, xOut]:
                        self.tmpMajorityPropArr[yOut, xOut] = self.outputFracArr[catPosition, yOut, xOut]
                        self.outputMajorityArr[yOut, xOut] = catValue

                    # for the like adjacency grids, output a value if we've had any input data here;
                    # but output nodata if we haven't.
                    # The arrays are initialised to zero so we need to change those positions to ndv.
                    # Use the fraction (class proportion) grids to check this
                    # (non-zero implies we had a data pixel)
                    if self.outputFracArr[catPosition, yOut, xOut] > 0:
                        # we've had at least one data pixel of this class at this location
                        # so we can calculate a like adjacency
                        if self._doLikeAdj:
                            self.outputLikeAdjArr[catPosition, yOut, xOut] = (
                                self.outputLikeAdjArr[catPosition, yOut, xOut] / self.outputFracArr[catPosition, yOut, xOut]
                            )
                        # also, convert the fraction to a percentage so we can return as int type
                        self.outputFracArr[catPosition, yOut, xOut] *= 100
                        # so we can cast (truncate) rather than round
                        self.outputFracArr[catPosition, yOut, xOut] += 0.5
                    else:
                        # no input pixels of this class at this output loc: output ndv for like-adjacency
                        if self._doLikeAdj:
                            self.outputLikeAdjArr[catPosition, yOut, xOut] = self._fltNDV
                        # for the fraction (class proportion) grids, output ndv if we haven't
                        # had any input data of _any_ class here (if we have, but of a different
                        # class, then the default 0 is correct for this fraction grid)
                        if self._coverageArr[yOut, xOut] & 2 == 0:
                            self.outputFracArr[catPosition, yOut, xOut] = self._fltNDV

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
            logger.logMessage("Warning, cannot generate a result without \
            having received input data for full extent")
            return False
        return True

    cpdef GetResults(self):
        '''If input is complete, returns a dict containing the results plus "valuemap" for the z-order '''
        if not self.finalise():
            return None
        returnObj = {
            catstats.FRACTIONS: np.asarray(self.outputFracArr).astype(np.int16),
            catstats.MAJORITY: np.asarray(self.outputMajorityArr), #.astype(np.uint8)
            "valuemap": np.asarray(self.valueMap)
        }
        if self._doLikeAdj:
            returnObj[catstats.LIKEADJACENCIES] = np.asarray(self.outputLikeAdjArr)
        return returnObj
