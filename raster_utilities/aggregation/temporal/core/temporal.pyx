cimport cython, openmp
from cython.parallel cimport parallel, prange
import numpy as np
from libc.math cimport sqrt
from raster_utilities.aggregation.aggregation_values import TemporalAggregationStats as tempstats


cdef class TemporalAggregator_Dynamic:
    '''Calculates running local statistics of a series of incoming arrays

    For a given array shape, determined at initialisation, this class creates arrays, for
    each requested statistic, of the local values of that statistic in the incoming arrays.
    Two arrays are created for each statistic, representing a subtotal (which can be retrieved and
    reset) and an overall (synoptic) total. The incoming arrays should then be supplied sequentially.

    The variance (and thus SD) are computed using the method of Donald Knuth which is robust against
    numerical errors that can otherwise occur for large values with small differences.

    Output arrays are generated simultaneously for both all data received, and for all data
    received since the value of the "month" parameter changed. This enables the class to be
    used to calculate monthly and overall statistics in a single pass.

    The calculation is implemented in optimised multithreaded C code generated using Cython. By
    default this will use 6 threads although this can be changed in the code.

    This class has been used to generate "synoptic" outputs from the MODIS variables used in MAP
    but of course can be used for any other suitable series of arrays where efficient flattening to
    mean/sd/count is required.
    '''

    cdef:
        short[:, ::1] step_n, tot_n
        double[:, ::1] step_oldMean, step_newMean, step_oldSD, step_newSD
        double[:, ::1] tot_oldMean, tot_newMean, tot_oldSD, tot_newSD
        float[:, ::1] step_Max, step_Min, step_Sum, tot_Max, tot_Min, tot_Sum
        double ndv
        Py_ssize_t height, width
        unsigned char _startNewStep
        # track with native vars so we don't have to search stats list in the loop
        unsigned char _doMean, _doSD, _doMin, _doMax, _doSum
        unsigned char _outputCount, _outputMean
        # allow skipping of synoptic outputs to only track one set of data and halve memory use
        unsigned char _generateSynoptic

    def __cinit__(self, Py_ssize_t height, Py_ssize_t width, double ndv, 
                stats, generateSynoptic = True):
        '''Height and width specify the shape of the arrays to be provided. NDV refers to outputs.

        The class requires approximately 80 bytes RAM per pixel for calculating mean, sd and count,
        as the outputs are tracked using double precision; plus more if additional stats are chosen.
        So choose a size that will be manageable in the RAM available.'''

        # initialise arrays as required to track both totals and subtotals
        self.tot_n = np.zeros((height, width), dtype='Int16')
        self.step_n = np.zeros((height, width),dtype='Int16')
        if tempstats.COUNT in stats:
            self._outputCount = 1
        if tempstats.MEAN in stats or tempstats.SD in stats:
            self._doMean = 1 # we need mean to do sd, even if we don't output it
            if tempstats.MEAN in stats:
                self._outputMean = 1
            if generateSynoptic:
                # initialise arrays to track totals
                self.tot_oldMean = np.zeros((height, width), dtype='float64')
                self.tot_newMean = np.zeros((height, width), dtype='float64')
                # don't have zero but no data instead because zero is valid. Counts remain at zero.
                self.tot_oldMean[:] = ndv
                self.tot_newMean[:] = ndv
            # initialise arrays to track subtotals
            self.step_oldMean = np.zeros((height, width),dtype='float64')
            self.step_newMean = np.zeros((height, width),dtype='float64')
            self.step_oldMean[:] = ndv
            self.step_newMean[:] = ndv
        if tempstats.SD in stats:
            self._doSD = 1
            if generateSynoptic:
                self.tot_oldSD = np.zeros((height, width), dtype='float64')
                self.tot_newSD = np.zeros((height, width), dtype='float64')
                self.tot_oldSD[:] = ndv
                self.tot_newSD[:] = ndv
            self.step_oldSD = np.zeros((height, width),dtype='float64')
            self.step_newSD = np.zeros((height, width),dtype='float64')
            self.step_oldSD[:] = ndv
            self.step_newSD[:] = ndv

        if tempstats.MIN in stats:
            self._doMin = 1
            if generateSynoptic:
                self.tot_Min = np.zeros((height, width), dtype='float32')
                self.tot_Min[:] = np.inf
            self.step_Min = np.zeros((height, width), dtype='float32')
            self.step_Min[:] = np.inf
        if tempstats.MAX in stats:
            self._doMax = 1
            if generateSynoptic:
                self.tot_Max = np.zeros((height, width), dtype='float32')
                self.tot_Max[:] = -np.inf
            self.step_Max = np.zeros((height, width), dtype='float32')
            self.step_Max[:] = -np.inf
        if tempstats.SUM in stats:
            self._doSum = 1
            if generateSynoptic:
                self.tot_Sum = np.zeros((height, width), dtype='float32')
            self.step_Sum = np.zeros((height, width), dtype='float32')

        self.ndv = ndv
        self.height = height
        self.width = width
        self._startNewStep = 1
        if generateSynoptic:
            self._generateSynoptic = 1

    @cython.boundscheck(False)
    @cython.cdivision(True)
    cpdef addFile(self, float[:,::1] data, float thisNdv):
        '''Add a tile of data to the running summaries. Data array must be of shape specified in startup.

        Specify the value in the data that should be treated as nodata (does not have to be the same
        as the output nodata value specified at startup)

        This should be called once for each incoming data file, calling emitStep when appropriate
        during the sequential process, and emitTotal when finished. E.g. add each file representing
        a "January" in turn, call emitStep to get the synoptic January results, then continue for
        each of the other calendar months, and call emitTotal to get the overall synoptic total when
        finished.

        At each pixel location (and for each of the month and total outputs), tracks the count
        of non-nodata values ever seen, and calculates the mean and the variance of those values
        (which will be converted to SD later).

        Variance is calculated using the numerically-robust method of Donald Knuth as described at
        http://www.johndcook.com/blog/standard_deviation/
        '''
        cdef:
            double value
            double test_ndv
            double sd
            Py_ssize_t y, x, yShape, xShape, t, b

        assert self.height == data.shape[0]
        assert self.width == data.shape[1]

        if self._startNewStep:
            #initialise arrays to track this month
            self.step_n = np.zeros((self.height, self.width),dtype='Int16')
            if self._doMean: # will be set even if only sd was set
                self.step_oldMean = np.zeros((self.height, self.width),dtype='float64')
                self.step_newMean = np.zeros((self.height, self.width),dtype='float64')
                self.step_oldMean[:] = self.ndv
                self.step_newMean[:] = self.ndv
            if self._doSD:
                self.step_oldSD = np.zeros((self.height, self.width),dtype='float64')
                self.step_newSD = np.zeros((self.height, self.width),dtype='float64')
                # don't have zero but no data instead because zero is valid. Counts remain at zero.
                self.step_oldSD[:] = self.ndv
                self.step_newSD[:] = self.ndv
            if self._doMin:
                self.step_Min = np.zeros((self.height, self.width), dtype='float32')
                self.step_Min[:] = np.inf
            if self._doMax:
                self.step_Max = np.zeros((self.height, self.width), dtype='float32')
                self.step_Max[:] = -np.inf
            if self._doSum:
                self.step_Sum = np.zeros((self.height, self.width), dtype='float32')
            self._startNewStep = 0

        # hardcode to 6 threads which is fast enough for anything that can 
        # feasibly fit in RAM anyway and is ok on most machines
        with nogil, cython.wraparound(False), parallel(num_threads=6):
            for y in prange (self.height, schedule='static'):
                value = thisNdv
                x = -1
                for x in range(0, self.width):
                    value = data[y, x]
                    if value == thisNdv:
                        continue
                    # always track a count
                    self.tot_n[y, x] +=1
                    self.step_n[y, x] += 1
                    
                    # do all the calcs for the overall result, if required
                    if self._generateSynoptic:
                        if self.tot_n[y, x] == 1:
                            if self._doMean: # nb this is set even if we're only doing sd as mean is needed for that
                                self.tot_oldMean[y, x] = value
                                self.tot_newMean[y, x] = value
                            if self._doSD:
                                self.tot_oldSD[y, x] = 0
                                self.tot_newSD[y, x] = 0
                        else:
                            if self._doMean:
                                self.tot_newMean[y,x] = (self.tot_oldMean[y,x] +
                                    ((value - self.tot_oldMean[y,x]) / self.tot_n[y,x]))
                            if self._doSD:
                                self.tot_newSD[y,x] = (self.tot_oldSD[y,x] +
                                    ((value - self.tot_oldMean[y,x]) *
                                     (value - self.tot_newMean[y,x])
                                      ))
                            # the SD calc above uses the old and new mean, so have to repeat the check now
                            # rather than move this line up
                            if self._doMean:
                                self.tot_oldMean[y,x] = self.tot_newMean[y,x]
                            if self._doSD:
                                self.tot_oldSD[y,x] = self.tot_newSD[y,x]
                        if self._doMax:
                            if value > self.tot_Max[y,x]:
                                self.tot_Max[y,x] = value
                        if self._doMin:
                            if value < self.tot_Min[y,x]:
                                self.tot_Min[y,x] = value
                        if self._doSum:
                            self.tot_Sum[y,x] += value

                    # do it all again for the subtotals
                    if self.step_n[y, x] == 1:
                        if self._doMean: # nb this is set even if we're only doing sd as mean is needed for that
                            self.step_oldMean[y, x] = value
                            self.step_newMean[y, x] = value
                        if self._doSD:
                            self.step_oldSD[y, x] = 0
                            self.step_newSD[y, x] = 0
                    else:
                        if self._doMean:
                            self.step_newMean[y, x] = (self.step_oldMean[y, x] +
                                            ((value - self.step_oldMean[y, x]) / self.step_n[y, x]))
                        if self._doSD:
                            self.step_newSD[y, x] = (self.step_oldSD[y, x] +
                                         ((value - self.step_oldMean[y, x]) *
                                          (value - self.step_newMean[y, x])
                                          ))
                        if self._doMean:
                            self.step_oldMean[y, x] = self.step_newMean[y, x]
                        if self._doSD:
                            self.step_oldSD[y, x] = self.step_newSD[y, x]
                    if self._doMax:
                        if value > self.step_Max[y,x]:
                            self.step_Max[y,x] = value
                    if self._doMin:
                        if value < self.step_Min[y,x]:
                            self.step_Min[y,x] = value
                    if self._doSum:
                        self.step_Sum[y,x] += value

    @cython.boundscheck(False)
    @cython.cdivision(True)
    cpdef emitStep(self):
        ''' Return the stats arrays accumulated *since this method was last called* '''
        cdef:
            double variance
            Py_ssize_t x, y
        if self._doSD:
            with nogil, cython.wraparound(False), parallel(num_threads=6):
                for y in prange(self.height, schedule='static'):
                    x = -1
                    for x in range (0, self.width):
                        if self.step_n[y, x] <=1:
                            continue
                        variance = self.step_newSD[y, x] / (self.step_n[y, x] - 1)
                        self.step_newSD[y, x] = sqrt(variance)
        self._startNewStep = 1
        returnObj = {
            tempstats.COUNT: np.asarray(self.step_n)
        }
        if self._outputMean:
            returnObj[tempstats.MEAN] = np.asarray(self.step_newMean).astype('float32')
        if self._doSD:
            returnObj[tempstats.SD] = np.asarray(self.step_newSD).astype('float32')
        if self._doMin:
            # get rid of the infinities
            with nogil, cython.wraparound(False), parallel(num_threads=6):
                for y in prange(self.height, schedule='static'):
                    x = -1
                    for x in range(0, self.width):
                        if self.step_n[y, x] == 0:
                            self.step_Min[y, x] = self.ndv
            returnObj[tempstats.MIN] = np.asarray(self.step_Min)
        if self._doMax:
            # get rid of the infinities
            with nogil, cython.wraparound(False), parallel(num_threads=6):
                for y in prange(self.height, schedule='static'):
                    x = -1
                    for x in range(0, self.width):
                        if self.step_n[y, x] == 0:
                            self.step_Max[y, x] = self.ndv
            returnObj[tempstats.MAX] = np.asarray(self.step_Max)
        if self._doSum:
            returnObj[tempstats.SUM] = np.asarray(self.step_Sum)

        return returnObj

    @cython.boundscheck(False)
    @cython.cdivision(True)
    cpdef emitTotal(self):
        ''' Return the stats arrays accumulated *since the class was instantiated* '''
        if not self._generateSynoptic:
            return {
                "error": "Generation of synoptic outputs was not done!"
            }
        cdef:
            double variance
            Py_ssize_t x, y
        if self._doSD:
            with nogil, cython.wraparound(False), parallel(num_threads=6):
                for y in prange(self.height, schedule='static'):
                    x = -1
                    for x in range(0, self.width):
                        if self.tot_n[y, x] <= 1:
                            continue
                        variance = self.tot_newSD[y, x] / (self.tot_n[y, x] - 1)
                        self.tot_newSD[y, x] = sqrt(variance)
        returnObj = {
            "count": np.asarray(self.tot_n)
        }
        if self._outputMean:
            returnObj[tempstats.MEAN] = np.asarray(self.tot_newMean).astype('float32')
        if self._doSD:
            returnObj[tempstats.SD] = np.asarray(self.tot_newSD).astype('float32')
        if self._doMin:
            # get rid of the infinities
            with nogil, cython.wraparound(False), parallel(num_threads=6):
                for y in prange(self.height, schedule='static'):
                    x = -1
                    for x in range(0, self.width):
                        if self.tot_n[y, x] == 0:
                            self.tot_Min[y, x] = self.ndv
            returnObj[tempstats.MIN] = np.asarray(self.tot_Min)
        if self._doMax:
            # get rid of the infinities
            with nogil, cython.wraparound(False), parallel(num_threads=6):
                for y in prange(self.height, schedule='static'):
                    x = -1
                    for x in range(0, self.width):
                        if self.tot_n[y, x] == 0:
                            self.tot_Max[y, x] = self.ndv
            returnObj[tempstats.MAX] = np.asarray(self.tot_Max)
        if self._doSum:
            returnObj[tempstats.SUM] = np.asarray(self.tot_Sum)

        return returnObj


