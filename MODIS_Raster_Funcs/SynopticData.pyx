cimport cython, openmp
from cython.parallel cimport parallel, prange
import numpy as np
from libc.math cimport sqrt

cdef class MonthlyStatCalculator:

    cdef:
        short[:, ::1] mth_n, tot_n_Days, tot_n_Months
        double[:, ::1] mth_oldM, mth_newM, mth_oldS, mth_newS
        double[:, ::1] tot_oldM_Days, tot_newM_Days, tot_oldS_Days, tot_newS_Days
        double ndv
        Py_ssize_t height, width
        unsigned char  _startNewMonth

    def __cinit__(self, Py_ssize_t height, Py_ssize_t width, double ndv):

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
    cpdef addFile(self, float[:,::1] data, unsigned char month, float thisNdv):
        cdef:
            double value
            double test_ndv
            double sd
            Py_ssize_t y, x, yShape, xShape, t, b

        assert self.height == data.shape[0]
        assert self.width == data.shape[1]
        assert 0 < month <= 12

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

