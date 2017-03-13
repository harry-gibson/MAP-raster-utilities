cimport cython, openmp
from cython.parallel cimport parallel, prange
import numpy as np
from libc.math cimport sqrt

cdef class MonthlyStatCalculator:
    '''Calculates a running mean, SD, and count of local values of a series of incoming arrays

    For a given array shape, determined at initialisation, this class creates an array of the local
    mean, SD, and count of the values in the incoming arrays, which will then be supplied sequentially.

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
        short[:, ::1] mth_n, tot_n_Days, tot_n_Months
        double[:, ::1] mth_oldM, mth_newM, mth_oldS, mth_newS
        double[:, ::1] tot_oldM_Days, tot_newM_Days, tot_oldS_Days, tot_newS_Days
        double ndv
        Py_ssize_t height, width
        unsigned char  _startNewMonth

    def __cinit__(self, Py_ssize_t height, Py_ssize_t width, double ndv):
        '''Height and width specify the shape of the arrays to be provided. NDV refers to outputs.

        The class requires approximately 80 bytes RAM per pixel as the outputs are tracked using
        double precision - choose a size that will be manageable in the RAM available.'''

        self.tot_n_Days = np.zeros((height, width), dtype='Int16')
        self.tot_oldM_Days = np.zeros((height, width), dtype='float64')
        self.tot_newM_Days = np.zeros((height, width), dtype='float64')
        self.tot_oldS_Days = np.zeros((height, width), dtype='float64')
        self.tot_newS_Days = np.zeros((height, width), dtype='float64')
        self.tot_oldM_Days[:] = ndv
        self.tot_newM_Days[:] = ndv
        self.tot_oldS_Days[:] = ndv
        self.tot_newS_Days[:] = ndv

        #initialise arrays to track this month
        self.mth_n = np.zeros((height, width),dtype='Int16')
        self.mth_oldM = np.zeros((height, width),dtype='float64')
        self.mth_newM = np.zeros((height, width),dtype='float64')
        self.mth_oldS = np.zeros((height, width),dtype='float64')
        self.mth_newS = np.zeros((height, width),dtype='float64')
        # don't have zero but no data instead because zero is valid. Counts remain at zero.
        self.mth_oldM[:] = ndv
        self.mth_newM[:] = ndv
        self.mth_oldS[:] = ndv
        self.mth_newS[:] = ndv

        self.ndv = ndv
        self.height = height
        self.width = width
        self._startNewMonth = 1

    @cython.boundscheck(False)
    @cython.cdivision(True)
    cpdef addFile(self, float[:,::1] data, float thisNdv):
        '''Add a tile of data to the running summaries. Data array must be of shape specified in startup.

        Specify the value in the data that should be treated as nodata (does not have to be the same
        as the output nodata value specified at startup)

        This should be called once for each incoming data file, calling emitMonth when appropriate
        during the sequential process, and emitTotal when finished. E.g. add each file representing
        a "January" in turn, call emitMonth to get the synoptic January results, then continue for
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

        if self._startNewMonth:
            #initialise arrays to track this month
            self.mth_n = np.zeros((self.height, self.width),dtype='Int16')
            self.mth_oldM = np.zeros((self.height, self.width),dtype='float64')
            self.mth_newM = np.zeros((self.height, self.width),dtype='float64')
            self.mth_oldS = np.zeros((self.height, self.width),dtype='float64')
            self.mth_newS = np.zeros((self.height, self.width),dtype='float64')
            # don't have zero but no data instead because zero is valid. Counts remain at zero.
            self.mth_oldM[:] = self.ndv
            self.mth_newM[:] = self.ndv
            self.mth_oldS[:] = self.ndv
            self.mth_newS[:] = self.ndv

            self._startNewMonth = 0


        with nogil, cython.wraparound(False), parallel(num_threads=6):
            for y in prange (self.height, schedule='static'):
                value = thisNdv
                x = -1
                for x in range(0, self.width):
                    value = data[y, x]
                    if value == thisNdv:
                        continue
                    self.tot_n_Days[y, x] +=1
                    self.mth_n[y, x] += 1

                    if self.tot_n_Days[y, x] == 1:
                        self.tot_oldM_Days[y, x] = value
                        self.tot_newM_Days[y, x] = value
                        self.tot_oldS_Days[y, x] = 0
                        self.tot_newS_Days[y, x] = 0
                    else:
                        self.tot_newM_Days[y,x] = (self.tot_oldM_Days[y,x] +
                                ((value - self.tot_oldM_Days[y,x]) / self.tot_n_Days[y,x]))
                        self.tot_newS_Days[y,x] = (self.tot_oldS_Days[y,x] +
                                ((value - self.tot_oldM_Days[y,x]) *
                                 (value - self.tot_newM_Days[y,x])
                                  ))
                        self.tot_oldM_Days[y,x] = self.tot_newM_Days[y,x]
                        self.tot_oldS_Days[y,x] = self.tot_newS_Days[y,x]

                    if self.mth_n[y, x] == 1:
                        self.mth_oldM[y, x] = value
                        self.mth_newM[y, x] = value
                        self.mth_oldS[y, x] = 0
                        self.mth_newS[y, x] = 0
                    else:
                        #update monthly stats
                        self.mth_newM[y, x] = (self.mth_oldM[y, x] +
                                         ((value - self.mth_oldM[y, x]) / self.mth_n[y, x]))
                        self.mth_newS[y, x] = (self.mth_oldS[y, x] +
                                         ((value - self.mth_oldM[y, x]) *
                                          (value - self.mth_newM[y, x])
                                          ))
                        self.mth_oldM[y, x] = self.mth_newM[y, x]
                        self.mth_oldS[y, x] = self.mth_newS[y, x]



    @cython.boundscheck(False)
    @cython.cdivision(True)
    cpdef emitMonth(self):
        ''' Return the count, mean, and sd arrays accumulated *since this method was last called* '''
        cdef:
            double variance
            Py_ssize_t x, y
        with nogil, cython.wraparound(False), parallel(num_threads=6):
            for y in prange(self.height, schedule='static'):
                x = -1
                for x in range (0, self.width):
                    if self.mth_n[y, x] <=1:
                        continue
                    variance = self.mth_newS[y, x] / (self.mth_n[y, x] - 1)
                    self.mth_newS[y, x] = sqrt(variance)
        self._startNewMonth = 1
        return {
            "count": np.asarray(self.mth_n),
            "mean": np.asarray(self.mth_newM).astype('float32'),
            "sd": np.asarray(self.mth_newS).astype('float32')

        }

    @cython.boundscheck(False)
    @cython.cdivision(True)
    cpdef emitTotal(self):
        ''' Return the count, mean, and sd arrays accumulated *since the class was instantiated* '''
        cdef:
            double variance
            Py_ssize_t x, y
        with nogil, cython.wraparound(False), parallel(num_threads=6):
            for y in prange(self.height, schedule='static'):
                x = -1
                for x in range(0, self.width):
                    if self.tot_n_Days[y, x] <= 1:
                        continue
                    variance = self.tot_newS_Days[y, x] / (self.tot_n_Days[y, x] - 1)
                    self.tot_newS_Days[y, x] = sqrt(variance)
        return {
            "count": np.asarray(self.tot_n_Days),
            "mean": np.asarray(self.tot_newM_Days).astype('float32'),
            "sd": np.asarray(self.tot_newS_Days).astype('float32')
        }

