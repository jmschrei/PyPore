import numpy as np
cimport numpy as np
cimport cython
import time

from cython.view cimport array as cvarray

cdef inline double double_max( double a, double b ): return a if a >= b else b
cdef inline double double_min( double a, double b ): return a if a <= b else b

cdef inline int double_argmax( double [:] x ):
	cdef int i, argmax
	cdef double maximum = -1
	for i in xrange( len(x) ):
		if x[i] > maximum:
			argmax = i
			maximum = x[i]
	return argmax

cdef class cSegmentAligner:
	cdef object model, seq 
	cdef double [:] model_means, model_stds, model_dur, c_model_dur
	cdef double skip_penalty, backslip_penalty
	
	def __init__( self, model_means, model_stds, model_dur, skip_penalty, backslip_penalty ):
		self.model_means = model_means
		self.model_stds = model_stds
		self.model_dur = model_dur
		self.c_model_dur = np.cumsum( self.model_dur )
		self.skip_penalty = skip_penalty
		self.backslip_penalty = backslip_penalty

	def align( self, seq_means, seq_stds, seq_durs ):
		return self._align( seq_means, seq_stds, seq_durs )

	cdef tuple _align( self, double [:] seq_means, double [:] seq_stds, double [:] seq_durs ):
		cdef unsigned int i, j, k, s=len( seq_means ), m=len(self.model_means)
		cdef int NEGINF = -99999999
		cdef double prev_score, max_err, skip_test, prev_test, back_test
		cdef double [:,:] match = cvarray( shape=(s, m), itemsize=sizeof(double), format="d" )
		cdef double [:,:] score = cvarray( shape=(s, m), itemsize=sizeof(double), format="d" )
		cdef double [:,:] skip_score = cvarray( shape=(s, m), itemsize=sizeof(double), format="d" )
		cdef double [:,:] backslip_score = cvarray( shape=(s, m), itemsize=sizeof(double), format="d" )
		cdef double [:] backtrace = cvarray( shape=(s,), itemsize=sizeof(double), format="d" )

		for i in xrange( s ):
			for j in xrange( m ):
				match[i, j] = -( seq_means[i]-self.model_means[j] ) ** 2 / ( seq_stds[i]*self.model_stds[j] )

		for j in xrange( m ):
			score[0,j] = match[0,j]*seq_durs[0] - self.skip_penalty*(self.c_model_dur[j]-self.model_dur[j])

		for i in xrange( 1, s ):
			skip_score[i, 0] = NEGINF
			for j in xrange( 1, m ):
				skip_score[i,j] = double_max( skip_score[i,j-1], score[i-1, j-1] ) - self.model_dur[j]*self.skip_penalty
			backslip_score[i,-1] = NEGINF
			for j in xrange( m-2, 0, -1 ):
				backslip_score[i,j] = double_max( backslip_score[i,j+1], score[i-1,j+1]) - self.model_dur[j+1]*self.backslip_penalty
			backslip_score[i,0] = double_max( backslip_score[i, 1], score[i-1, 1]) - self.model_dur[1]*self.backslip_penalty

			for j in xrange( m ):
				prev_score = score[i-1, j]
				if j > 0:
					prev_score = max( prev_score, score[i-1, j-1], skip_score[i, j-1])
				if j < m-1:
					prev_score = double_max( prev_score, backslip_score[i,j] )
				score[i,j] = prev_score + match[i,j]*seq_durs[i]

		j = double_argmax( score[s-1] )

		for i in xrange( s-1, 0, -1 ):
			backtrace[i] = j
			prev_test = score[i, j]-match[i,j]*seq_durs[i]
			max_err = 1e-6*abs(prev_test)
			if abs(prev_test-score[i-1,j-1] ) <= max_err:
				j -= 1
				continue
			if abs(prev_test-score[i-1,j] ) <= max_err:
				continue
			if j < m-1:
				k = j
				back_test = prev_test
				while k < m-1 and abs(back_test-backslip_score[i,k]) <= max_err:
					k += 1
					back_test += self.model_dur[k]*self.backslip_penalty
				if k > j:
					j = k
					continue
			if j > 0:
				k = j
				skip_test = prev_test
				while k >= 1 and abs(skip_test-skip_score[i,k-1]) <= max_err:
					k -= 1
					skip_test += self.model_dur[k]*self.skip_penalty
				if k < j:
					j = k-1
					continue

		backtrace[0] = j
		return ( score[s-1,m-1] / np.sum(seq_durs), np.array(backtrace) )