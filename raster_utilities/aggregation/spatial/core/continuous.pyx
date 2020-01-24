cimport cython
import numpy as np
from libc.math cimport sqrt
from raster_utilities.aggregation.aggregation_values import ContinuousAggregationStats as contstats
from ....utils.logger import MessageLogger
# This file contains Cython code so must be translated (to C) and compiled before it can be used in Python.
# To do this simply run "python setup.py build_ext --inplace" (assuming you have cython installed)


@cython.boundscheck(True)
@cython.cdivision(True)
@cython.wraparound(False)
cdef class Continuous_Aggregator_Flt:
    ''' Aggregates a continuous float raster grid to smaller dimensions, i.e. coarser resolution

    Returns a dictionary containing up to seven items, each being a different summary of the source
    pixels covered by each output pixel. The item key is the string name of the statistic/summary
    type and the value is the grid at the aggregated resolution for that statistic.
    The statistics available are "count", "min", "max", "range", "sum", "mean", "sd"
    
    The aggregation is built up by adding tiles of the input data using the addTile method. This way a grid
    can be aggregated that is much too large to fit in memory. The required output and total input sizes must be
    specified at instantiation. Input tiles should be provided to cover the whole extent of the specified output -
    if no data is available for some areas then just add a tile of nodata. (At present only a warning is generated
    if this isn't done: you can still continue).

    The aggregation is run in optimised C code generated using Cython. This will need compiling / translating on
    first use.
    '''

    cdef:
        Py_ssize_t xShapeOut, yShapeOut
        Py_ssize_t xShapeIn, yShapeIn, tileXShapeIn, tileYShapeIn
        int xSubpixelOffsetIn, ySubpixelOffsetIn

    cdef:

        float _NDV
        char minMaxRangeSumOnly

        float[:,::1] outputMeanArr
        float[:,::1] outputMinArr
        float[:,::1] outputMaxArr
        float[:,::1] outputRangeArr
        float[:,::1] outputSumArr
        float[:,::1] outputSDArr

        float[:,::1] _oldSDArr
        float[:,::1] _oldMeanArr

        int[:,::1] outputCountArr

        char[:,::1] _coverageArr

        double variance
        double xFact, yFact, xFactCalc, yFactCalc
        # things we need to calculate
        unsigned char _doMean, _doSD, _doMin, _doMax, _doSum, _doRange
        # things we want to output - e.g. we may need to track mean for the SD calculation, 
        # but not want to output it
        unsigned char _outputMean, _outputMax, _outputMin, _outputSum

    @cython.boundscheck(True)
    @cython.cdivision(True)
    @cython.wraparound(False)
    def __cinit__(self,
                  Py_ssize_t ySizeIn, Py_ssize_t xSizeIn,
                  Py_ssize_t ySizeOut, Py_ssize_t xSizeOut,
                  int ySubpixelOffsetIn, int xSubpixelOffsetIn,
                  double yFactor, double xFactor,
                  _NDV,
        stats):
        '''
        Initialise an instance of the class for aggregating a float-type dataset.

        ySizeIn: the overall y dimension of the input file being aggregated
        xSizeIn: the overall x dimension of the input file being aggregated
        ySizeOut: the overall y dimension of the desired output file (ideally but not
            necessarily a multiple of ySizeIn)
        xSizeOut: the overall x dimension of the desired output file (ideally but not
            necessarily a multiple of xSizeIn
        ySubpixelOffsetIn: how many input pixels from the left edge of the origin output pixel is the input origin
        xSubpixelOffsetIn: how many input pixels from the top edge of the origin output pixel is the input origin
            The output may be aligned e.g. to a "5km global" grid and the input coming from a "1k global" grid does
            not necessarily originate at a point coincident with the edge of a 5km pixel. So we need to allow for this
            to maintain proper alignment.
        '''
        assert xSizeIn > xSizeOut
        assert ySizeIn > ySizeOut

        self.xShapeIn = xSizeIn
        self.yShapeIn = ySizeIn
        self.xShapeOut = xSizeOut
        self.yShapeOut = ySizeOut

        self.minMaxRangeSumOnly = contstats.MEAN not in stats and  contstats.SD not in stats

        # the factor can be non-integer, which means the output cells will have 
        # varying number of input cells
        # If the input size is not a clean multiple of the user-requested factor then the output size will have
        # been rounded up. Also if snapping has caused the extent to be less it will have been rounded up.
        # So we can't correctly calculate the factor here and must use what is passed in.
        self.xFactCalc = <double>self.xShapeIn / self.xShapeOut
        self.yFactCalc = <double>self.yShapeIn / self.yShapeOut

        self.xFact = xFactor
        self.yFact = yFactor

        if xSubpixelOffsetIn >= xFactor or ySubpixelOffsetIn >= yFactor:
            raise ValueError("sub-pixel offsets must be less than the cell aggregation factor")
        self.xSubpixelOffsetIn = xSubpixelOffsetIn
        self.ySubpixelOffsetIn = ySubpixelOffsetIn

        xSizeCheck = int(((xSizeIn + xSubpixelOffsetIn) / xFactor) + 0.5)
        ySizeCheck = int(((ySizeIn + ySubpixelOffsetIn) / yFactor) + 0.5)
        if xSizeCheck > xSizeOut or ySizeCheck > ySizeOut:
            raise ValueError("specified output size is too small")
        if xSizeCheck < xSizeOut-1 or ySizeCheck < ySizeOut-1:
            raise ValueError("specified output size is too big")
        self._NDV = _NDV

        # initialise the output arrays
        # always keep a coverage array which tells us when all tiles have been added
        self._coverageArr = np.zeros(shape=(ySizeOut, xSizeOut), dtype = np.byte)
        self.outputCountArr = np.zeros(shape=(ySizeOut, xSizeOut), dtype = np.int32)
        # but only create as many of the other arrays as we actually need, to save memory
        if contstats.MIN in stats:
            self._doMin = 1
            self._outputMin = 1
        if contstats.MAX in stats:
            self._doMax = 1
            self._outputMax = 1
        if contstats.RANGE in stats:
            self._doRange = 1
            self._doMax = 1
            self._doMin = 1
        if contstats.SUM in stats:
            self._doSum = 1
            self._outputSum = 1
        if contstats.MEAN in stats or contstats.SD in stats:
            self._doMean = 1
            if contstats.MEAN in stats:
                self._outputMean = 1
                self._doSum = 1
            self.outputMeanArr = np.zeros(shape=(ySizeOut, xSizeOut), dtype = np.float32)
            self._oldMeanArr = np.zeros(shape=(ySizeOut, xSizeOut), dtype = np.float32)
            self.outputMeanArr[:] = _NDV
            self._oldMeanArr[:] = _NDV
        if contstats.SD in stats:
            self._doSD = 1
            self.outputSDArr = np.zeros(shape=(ySizeOut, xSizeOut), dtype = np.float32)
            self._oldSDArr = np.zeros(shape=(ySizeOut, xSizeOut), dtype = np.float32)
            self.outputSDArr[:] = _NDV
            self._oldSDArr[:] = _NDV
        if self._doSum:
            self.outputSumArr = np.zeros(shape=(ySizeOut, xSizeOut), dtype = np.float32)
        if self._doMax:
            self.outputMaxArr = np.zeros(shape=(ySizeOut, xSizeOut), dtype = np.float32)
            self.outputMaxArr[:] = -np.inf
        if self._doMin:
            self.outputMinArr = np.zeros(shape=(ySizeOut, xSizeOut), dtype = np.float32)
            self.outputMinArr[:] = np.inf


        #self.outputRangeArr = np.zeros(shape=(ySizeOut, xSizeOut), dtype = np.float32)
            
    @cython.boundscheck(True)
    @cython.cdivision(True)
    @cython.wraparound(False)
    cpdef addTile(self, float[:,::1] data, Py_ssize_t xTileOffset, Py_ssize_t yTileOffset, float ndv_In):
        ''' Add a tile of the input data to the aggregation. Provide the offset of the top-left pixel.
        float[:,::1] data, int xOffset, int yOffset
        '''
        cdef:
            Py_ssize_t tileYShapeIn, tileXShapeIn
            Py_ssize_t xInGlobal, xInTile, yInGlobal, yInTile
            int xSubCellOffset, ySubCellOffset
            float localValue
            Py_ssize_t xOut, yOut
        tileYShapeIn = data.shape[0]
        tileXShapeIn = data.shape[1]
        xSubCellOffset = self.xSubpixelOffsetIn
        ySubCellOffset = self.ySubpixelOffsetIn
        # unlike the categorical aggregation, this code isn't parallelised because
        # most of the work (checking and updating for max / min etc) would need
        # to lock the arrays anyway
        for yInTile in range(tileYShapeIn):
            yInGlobal = yInTile + yTileOffset
            # cast to int, means round down, means output row is based on the top of the input cell
            yOut = <int> ((yInGlobal + ySubCellOffset) / self.yFact)
            localValue=-1
            xOut = -1
            if yOut >= self.yShapeOut:
                    raise ValueError("yOut is {}, yShape is {}".format(yOut, self.yShapeOut))

            for xInTile in range(tileXShapeIn):
                xInGlobal = xInTile + xTileOffset
                # output col is based on the left of the input cell
                xOut = <int> ((xInGlobal+xSubCellOffset) / self.xFact)
                if xOut >= self.xShapeOut:
                    raise ValueError("xOut is {}, xShape is {}".format(xOut, self.xShapeOut))
                self._coverageArr[yOut, xOut] = 1

                localValue = data[yInTile, xInTile]
                if localValue == ndv_In:
                    continue
                # always track count
                self.outputCountArr[yOut, xOut] += 1
                # Max, Min, Sum - the easy ones
                if self._doMax:
                    if localValue > self.outputMaxArr[yOut, xOut]:
                        self.outputMaxArr[yOut, xOut] = localValue
                if self._doMin:
                    if localValue < self.outputMinArr[yOut, xOut]:
                        self.outputMinArr[yOut, xOut] = localValue
                if self._doSum:
                    self.outputSumArr[yOut, xOut] += localValue
                # Running mean and SD
                if self.outputCountArr[yOut, xOut] == 1:
                    if self._doMean:
                        self._oldMeanArr[yOut, xOut] = localValue
                        self.outputMeanArr[yOut, xOut] = localValue
                    if self._doSD:
                        self._oldSDArr[yOut, xOut] = 0
                        self.outputSDArr[yOut, xOut] = 0
                else:
                    if self._doMean:
                        self.outputMeanArr[yOut, xOut] = (self._oldMeanArr[yOut, xOut] +
                                                     ((localValue - self._oldMeanArr[yOut, xOut]) /
                                                          self.outputCountArr[yOut, xOut]))
                    if self._doSD:
                        self.outputSDArr[yOut, xOut] = (self._oldSDArr[yOut, xOut] +
                                                   ((localValue - self._oldMeanArr[yOut, xOut]) *
                                                    (localValue - self.outputMeanArr[yOut, xOut])
                                                    ))
                    # the SD calc above uses the old and new mean, so have to repeat the check now
                    # rather than move this line up
                    if self._doMean:
                        self._oldMeanArr[yOut, xOut] = self.outputMeanArr[yOut, xOut]
                    if self._doSD:
                        self._oldSDArr[yOut, xOut] = self.outputSDArr[yOut, xOut]

    @cython.boundscheck(True)
    @cython.cdivision(True)
    @cython.wraparound(False)
    cdef finalise(self):
        cdef:
            Py_ssize_t xOut, yOut
            float variance
            float iscomplete = 1
        # free the mem from the arrays we're done with before attempting to allocate one for range
        self._oldMeanArr = None
        self._oldSDArr = None
        if self._doRange:
            self.outputRangeArr = np.zeros(shape=(self.yShapeOut, self.xShapeOut), dtype = np.float32)
                    
        for yOut in range(self.yShapeOut):
            xOut = -1
            for xOut in range(self.xShapeOut):
                if self._coverageArr[yOut, xOut] == 0:
                    iscomplete = 0
                if self.outputCountArr[yOut, xOut] == 0:
                    if self._doMin:
                        self.outputMinArr[yOut, xOut] = self._NDV
                    if self._doMax:
                        self.outputMaxArr[yOut, xOut] = self._NDV
                    if self._doRange:
                        self.outputRangeArr[yOut, xOut] = self._NDV
                    if self._doSum:
                        self.outputSumArr[yOut, xOut] = self._NDV
                    if self._doMean:
                        self.outputMeanArr[yOut, xOut] = self._NDV
                    #continue
                else:
                    # min, max, sum, count don't need any further processing
                    if self._doRange:
                        self.outputRangeArr[yOut, xOut] = (
                            self.outputMaxArr[yOut, xOut] - self.outputMinArr[yOut, xOut])
                    if self._doSD:
                        variance = self.outputSDArr[yOut, xOut] / self.outputCountArr[yOut, xOut]
                        self.outputSDArr[yOut, xOut] = sqrt(variance)
                    if self._outputMean:
                        # re-calculate the mean using simple sum/n as the running mean method is more
                        # likely to have accumulated (slight) fp errors (in practice they seem to match
                        # to around 1e-6 but this will depend on the size of the values)
                        self.outputMeanArr[yOut, xOut] = (
                            self.outputSumArr[yOut, xOut] / self.outputCountArr[yOut, xOut])
        if not iscomplete:
            logger = MessageLogger()
            logger.logMessage("Warning, generating a result without having received input data for full extent")

    @cython.boundscheck(False)
    @cython.cdivision(True)
    @cython.wraparound(False)
    cpdef GetResults(self):
        self.finalise()
        returnObj = {
            contstats.COUNT: np.asarray(self.outputCountArr)
        }
        if self._outputMin:
            returnObj[contstats.MIN] = np.asarray(self.outputMinArr)
        if self._outputMax:
            returnObj[contstats.MAX] = np.asarray(self.outputMaxArr)
        if self._doRange:
            returnObj[contstats.RANGE] = np.asarray(self.outputRangeArr)
        if self._outputMean:
            returnObj[contstats.MEAN] = np.asarray(self.outputMeanArr)
        if self._doSD:
            returnObj[contstats.SD] = np.asarray(self.outputSDArr)
        if self._outputSum:
            returnObj[contstats.SUM] = np.asarray(self.outputSumArr)

        return returnObj
        

