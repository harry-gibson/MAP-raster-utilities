cimport cython
import numpy as np
from libc.math cimport sqrt


@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
cdef class RasterAggregator_float:
    ''' Aggregates a continuous raster grid to smaller dimensions, i.e. coarser resolution

     Returns a tuple containing up to seven grids at the aggregated resolution,
     each representing a different summary of the source pixels covered by each output
     pixel, namely:
        (
          Count,
          Max,
          Mean (or None),
          Min,
          Range,
          Sum,
          SD (or None)
        )
    '''

    cdef:
        Py_ssize_t xShapeOut, yShapeOut
        Py_ssize_t xShapeIn, yShapeIn, tileXShapeIn, tileYShapeIn

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

        float proportion
        double variance
        double xFact, yFact

    @cython.boundscheck(False)
    @cython.cdivision(True)
    @cython.wraparound(False)
    def __cinit__(self, xSizeIn, ySizeIn, xSizeOut, ySizeOut, _NDV, minMaxRangeSumOnly):
        '''
        Initialise an instance of the class for aggregating a float-type dataset.

        xSizeIn: the overall x dimension of the input file being aggregated
        ySizeIn: the overall y dimension of the input file being aggregated
        xSizeOut: the overall x dimension of the desired output file (ideally but not
            necessarily a multiple of xSizeIn
        ySizeOut: the overall y dimension of the desired output file (ideally but not
            necessarily a multiple of ySizeIn)
        '''
        assert xSizeIn > xSizeOut
        assert ySizeIn > ySizeOut

        self.xShapeIn = xSizeIn
        self.yShapeIn = ySizeIn
        self.xShapeOut = xSizeOut
        self.yShapeOut = ySizeOut

        self.minMaxRangeSumOnly = minMaxRangeSumOnly

        self.xFact = <double>self.xShapeIn / self.xShapeOut
        self.yFact = <double>self.yShapeIn / self.yShapeOut

        # how much of an output cell does each input cell account for
        self.proportion = 1.0 / (self.xFact * self.yFact)

        self._NDV = _NDV

        # initialise the output arrays
        self.outputMinArr = np.zeros(shape=(ySizeOut, xSizeOut), dtype = np.float32)
        self.outputMaxArr = np.zeros(shape=(ySizeOut, xSizeOut), dtype = np.float32)
        self.outputSumArr = np.zeros(shape=(ySizeOut, xSizeOut), dtype = np.float32)
        self.outputRangeArr = np.zeros(shape=(ySizeOut, xSizeOut), dtype = np.float32)
        self.outputCountArr = np.zeros(shape=(ySizeOut, xSizeOut), dtype = np.int32)
        self.outputMinArr[:] = np.inf
        self.outputMaxArr[:] = -np.inf
        self._coverageArr = np.zeros(shape=(ySizeOut, xSizeOut), dtype = np.byte)

        if not minMaxRangeSumOnly:
            self.outputMeanArr = np.zeros(shape=(ySizeOut, xSizeOut), dtype = np.float32)
            self.outputSDArr = np.zeros(shape=(ySizeOut, xSizeOut), dtype = np.float32)

            self._oldSDArr = np.zeros(shape=(ySizeOut, xSizeOut), dtype = np.float32)
            self._oldMeanArr = np.zeros(shape=(ySizeOut, xSizeOut), dtype = np.float32)

            self.outputMeanArr[:] = _NDV
            self.outputSDArr[:] = _NDV

            self._oldMeanArr[:] = _NDV
            self._oldSDArr[:] = _NDV

    @cython.boundscheck(False)
    @cython.cdivision(True)
    @cython.wraparound(False)
    cpdef addTile(self, float[:,::1] data, Py_ssize_t xOffset, Py_ssize_t yOffset):
        ''' Add a tile of the input data to the aggregation. Provide the offset of the top-left pixel.
        float[:,::1] data, int xOffset, int yOffset
        '''
        cdef:
            Py_ssize_t tileYShapeIn, tileXShapeIn
            int yBelow, yAbove, xLeft, xRight
            Py_ssize_t xInGlobal, xInTile, yInGlobal, yInTile
            float localValue
            Py_ssize_t xOut, yOut
        tileYShapeIn = data.shape[0]
        tileXShapeIn = data.shape[1]

        # unlike the categorical aggregation, this code isn't parallelised because
        # most of the work (checking and updating for max / min etc) would need
        # to lock the arrays anyway
        for yInTile in range(tileYShapeIn):
            yInGlobal = yInTile + yOffset
            yAbove = yInGlobal - 1
            if yInGlobal == 0:
                yAbove = -1
            yBelow = yInGlobal+1
            if yInGlobal == self.yShapeIn-1:
                yBelow = -1
            yOut = <int> (yInGlobal / self.yFact)
            localValue=-1
            xOut = -1
            for xInTile in range(tileXShapeIn):
                xInGlobal = xInTile + xOffset
                xLeft = xInGlobal - 1
                if xInGlobal == 0:
                    xLeft = -1
                xRight = xInGlobal + 1
                if xInGlobal == self.xShapeIn - 1:
                    xRight = -1
                xOut = <int> (xInGlobal / self.xFact)

                self._coverageArr[yOut, xOut] = 1

                localValue = data[yInTile, xInTile]
                if localValue == self._NDV:
                    continue
                # Max and Min
                if localValue > self.outputMaxArr[yOut, xOut]:
                    self.outputMaxArr[yOut, xOut] = localValue
                if localValue < self.outputMinArr[yOut, xOut]:
                    self.outputMinArr[yOut, xOut] = localValue
                # Sum and Count
                self.outputSumArr[yOut, xOut] += localValue
                self.outputCountArr[yOut, xOut] += 1
                if not self.minMaxRangeSumOnly:
                    # Running mean and SD
                    if self.outputCountArr[yOut, xOut] == 1:
                        self._oldMeanArr[yOut, xOut] = localValue
                        self.outputMeanArr[yOut, xOut] = localValue
                        self._oldSDArr[yOut, xOut] = 0
                        self.outputSDArr[yOut, xOut] = 0
                    else:
                        self.outputMeanArr[yOut, xOut] = (self._oldMeanArr[yOut, xOut] +
                                                     ((localValue - self._oldMeanArr[yOut, xOut]) /
                                                          self.outputCountArr[yOut, xOut]))
                        self.outputSDArr[yOut, xOut] = (self._oldSDArr[yOut, xOut] +
                                                   ((localValue - self._oldMeanArr[yOut, xOut]) *
                                                    (localValue - self.outputMeanArr[yOut, xOut])
                                                    ))
                        self._oldMeanArr[yOut, xOut] = self.outputMeanArr[yOut, xOut]
                        self._oldSDArr[yOut, xOut] = self.outputSDArr[yOut, xOut]

    @cython.boundscheck(False)
    @cython.cdivision(True)
    @cython.wraparound(False)
    cdef finalise(self):
        cdef:
            Py_ssize_t xOut, yOut
            float variance
            float iscomplete = 1
        for yOut in range(self.yShapeOut):
            xOut = -1
            for xOut in range(self.xShapeOut):
                if self._coverageArr[yOut, xOut] == 0:
                    iscomplete = 0
                if self.outputCountArr[yOut, xOut] == 0:
                    self.outputMinArr[yOut, xOut] = self._NDV
                    self.outputMaxArr[yOut, xOut] = self._NDV
                    self.outputRangeArr[yOut, xOut] = self._NDV
                    self.outputSumArr[yOut, xOut] = self._NDV
                    if not self.minMaxRangeSumOnly:
                        self.outputMeanArr[yOut, xOut] = self._NDV
                    #continue
                else:
                    # min, max, sum, count are already set
                    self.outputRangeArr[yOut, xOut] = (
                        self.outputMaxArr[yOut, xOut] - self.outputMinArr[yOut, xOut])
                    if not self.minMaxRangeSumOnly:
                        variance = self.outputSDArr[yOut, xOut] / self.outputCountArr[yOut, xOut]
                        self.outputSDArr[yOut, xOut] = sqrt(variance)
                        # re-calculate the mean using simple sum/n as the running mean method is more
                        # likely to have accumulated (slight) fp errors (in practice they seem to match
                        # to around 1e-6 but this will depend on the size of the values)
                        self.outputMeanArr[yOut, xOut] = (
                            self.outputSumArr[yOut, xOut] / self.outputCountArr[yOut, xOut])
        if not iscomplete:
            print "Warning, generating a result without having received input data for full extent"

    @cython.boundscheck(False)
    @cython.cdivision(True)
    @cython.wraparound(False)
    cpdef GetResults(self):
        self.finalise()
        if not self.minMaxRangeSumOnly:
            return { # count, max, mean, min, range, sum
                "count": np.asarray(self.outputCountArr),
                "max": np.asarray(self.outputMaxArr),
                "mean": np.asarray(self.outputMeanArr).astype(np.float32),
                "min": np.asarray(self.outputMinArr),
                "range": np.asarray(self.outputRangeArr),
                "sum": np.asarray(self.outputSumArr),
                "sd": np.asarray(self.outputSDArr).astype(np.float32),
            }
        else:
            return {
                "count": np.asarray(self.outputCountArr),
                "max": np.asarray(self.outputMaxArr),
                "min": np.asarray(self.outputMinArr),
                "range": np.asarray(self.outputRangeArr),
                "sum": np.asarray(self.outputSumArr),
            }

