# cparsers.pyx
# Contact: Jacob Schreiber
#          jmschreiber91@gmail.com

'''
This contains cython implementations of ionic current parsers which are in
parsers.py. Currently the only parser is StatSplit, which is implemented
as FastStatSplit.
'''

import numpy as np
cimport numpy as np

from libc.math cimport log
cimport cython

from itertools import tee, izip, chain
from core import Segment

# Implement the max and min functions as cython
cdef inline int int_max( int a, int b ): return a if a >= b else b
cdef inline int int_min( int a, int b ): return a if a <= b else b


# Calculate the mean of a segment of current
@cython.boundscheck(False)
cdef inline double mean_c( int start, int end, double [:] c ): 
	return ( c[end-1] - c[start-1] ) / ( end-start) if start != 0 else c[end-1]/end if start != end else 0

# Calculate the variance of a segment of current
@cython.boundscheck(False)
cdef inline double var_c( int start, int end, double [:] c, double [:] c2 ):
	if start == end:
		return 0
	if start == 0:
		return c2[end-1]/end - (c[end-1]/end) ** 2
	return (c2[end-1]-c2[start-1])/(end-start) - \
		((c[end-1]-c[start-1])/(end-start))**2

def pairwise(iterable):
	a, b = tee(iterable)
	next(b, None)
	return izip(a, b)

cdef class FastStatSplit:
	'''
	A cython implementation of the segmenter written by Kevin Karplus. Sped up approximately 50-100x
	compared to the Python implementation depending on parameters.
	'''

	cdef int min_width, max_width, window_width, sampling_freq
	cdef public double min_gain
	cdef double [:] c, c2

	def __init__( self, min_width=100, max_width=1000000, window_width=10000,
		min_gain_per_sample=None, false_positive_rate=None,
		prior_segments_per_second=None, sampling_freq=1.e5, cutoff_freq=None ):

		self.min_width = min_width
		self.max_width = max_width
		self.window_width = window_width
		self.sampling_freq = sampling_freq

		if not false_positive_rate:
			false_positive_rate = sampling_freq
		if not prior_segments_per_second: 
			prior_segments_per_second = sampling_freq / 2.

		assert self.max_width >= self.min_width, "Maximum width must be greater\
			than minimum width."
		assert self.window_width >= 2*self.min_width, "Window width must be\
			greater than twice the minimum width."

		if cutoff_freq:
			assert cutoff_freq <= 0.5*sampling_freq, "Cutoff freq must be\
				less than half the sampling frequency."

		# Now set min_gain appropriately, either using the old method or
		# by calculating a new one in a Bayesian manner as described here:
		# http://gasstationwithoutpumps.wordpress.com/2014/02/01/more-on-
		# segmenting-noisy-signals/
		if min_gain_per_sample:
			# Use old method for setting gain (DEPRECATED)
			self.min_gain = min_gain_per_sample * self.window_width

		else:
			# Set the ratio between the cutoff frequency and the Nyquist
			# frequency.
			k = cutoff_freq / ( 0.5 * sampling_freq ) if cutoff_freq else 1
			
			# Shorten the name
			sps = prior_segments_per_second

			# Set the gain threshold in a Bayesian manner
			self.min_gain = \
				( -log( sps / ( sampling_freq - sps ) ) \
				  -log( false_positive_rate / sampling_freq ) ) / k 

		# Convert from sigma to variance, since this is in log space multiply
		# by two instead of square.
		self.min_gain *= 2

	def parse( self, current ):
		'''
		Wrapper function for the segmentation, which is implemented in cython.
		'''

		cdef list break_points
		cdef list paired
		self.c = np.cumsum( current )
		self.c2 = np.cumsum( np.multiply( current, current ) )

		breakpoints = self._recursive_split( 0, int(len(current)) )

		segments = [ Segment( current=current[start:end], start=start, duration=(end-start),
			end=end ) for start, end in pairwise( chain([0],breakpoints,[len(current)]) ) ]

		return segments

	def best_single_split( self, current ):
		'''
		Wrapper for a single call to _best_single_split. It will find the
		single best split in a series of current, and return the index of
		that split. Returns a tuple of ( gain, index ) where gain is the
		gain in variance by splitting there, and index is the index at which
		the split should occur in the current array. 
		'''

		self.c = np.cumsum( current )
		self.c2 = np.cumsum( np.multiply( current, current ) )

		return self._best_single_split()

	cdef tuple _best_single_split( self ):
		'''
		A slghtly modification of _best_split_stepwise, ensuring that the
		single best split is returned instead of only ones which meet a
		threshold.
		'''

		cdef int start = 0, end = len( self.c ) - 1, i, x = -1
		cdef double var_summed, low_var_summed, high_var_summed, gain
		cdef double min_gain = 0.

		var_summed = end * log( var_c( 0, end, self.c, self.c2))
		
		for i in xrange( 2, end-2):
			low_var_summed = i * log( var_c( 0, i, self.c, self.c2 ) )
			high_var_summed = ( end-i ) * log( var_c( i, end, self.c, self.c2 ) )
			gain = var_summed-( low_var_summed+high_var_summed )
			if gain > min_gain:
				min_gain = gain
				x = i

		return min_gain, x

	@cython.boundscheck(False)
	cdef int _best_split_stepwise( self, int start, int end ):
		'''
		Find the best split in a segment between start and end. Calculate best
		split by maximizing the change in variance.
		'''

		if end-start <= 2*self.min_width:
			return -1 
		cdef double var_summed = (end - start) * log( var_c(start, end, self.c, self.c2) )
		cdef double min_gain = self.min_gain
		cdef int i, x = -1
		cdef double low_var_summed, high_var_summed, gain

		for i in xrange( start+self.min_width, end+1-self.min_width ):
			low_var_summed = ( i-start ) * log( var_c( start, i, self.c, self.c2 ) )
			high_var_summed = ( end-i ) * log( var_c( i, end, self.c, self.c2 ) )
			gain = var_summed-( low_var_summed+high_var_summed )
			if gain > min_gain:
				min_gain = gain
				x = i
		return x

	cdef list _recursive_split( self, int start, int end ):
		'''
		Find the best splits recursively in the current until you get to the
		minimum width possible, and have looked at the entire 
		'''

		cdef int pseudostart, pseudoend, split_at = -1

		for pseudostart in xrange( start, end-2*self.min_width, self.window_width//2 ):
			if pseudostart > start + self.max_width:
				split_at = int_min( start+self.max_width, end-self.min_width )
				return [ split_at ] + self._recursive_split( split_at, end )

			pseudoend = int_min( end, pseudostart+self.window_width )
			split_at = self._best_split_stepwise( pseudostart, pseudoend )
			if split_at >= 0:
				break

		if split_at == -1:
			if end-start <= self.max_width:
				return []
			split_at = int_min( start+self.max_width, end-self.min_width )
		return self._recursive_split( start, split_at ) + [ split_at ] + \
			self._recursive_split( split_at, end )

	def score_samples( self, current, no_split=False ):
		'''
		Return a series of lists scoring each sample. However, this isn't just
		scoring each sample once. Every time a split is detected, it will return
		a new list with newly scored samples. In essence, it returns one list
		per scan of the current using the recursive method.
		'''

		self.c = np.cumsum( current )
		self.c2 = np.cumsum( np.multiply( current, current ) )
		return self._recursive_split_scoring( 0, len(current), no_split )

	@cython.boundscheck(False)
	cdef tuple _best_split_stepwise_score( self, int start, int end ):
		'''
		Find the best split in a segment between start and end. Calculate best
		split by maximizing the change in variance, and return the log score of
		each sample which has been viewed. 
		'''

		if end-start <= 2*self.min_width:
			return -1, []
		cdef double var_summed = (end - start) * log( var_c(start, end, self.c, self.c2) )
		cdef double min_gain = self.min_gain
		cdef int i, x = -1
		cdef double low_var_summed, high_var_summed, gain
		cdef np.ndarray score = np.zeros( len(self.c) )

		for i in xrange( start+self.min_width, end+1-self.min_width ):
			low_var_summed = ( i-start ) * log( var_c( start, i, self.c, self.c2 ) )
			high_var_summed = ( end-i ) * log( var_c( i, end, self.c, self.c2 ) )
			gain = var_summed-( low_var_summed+high_var_summed )
			score[i] = gain
			if gain > min_gain:
				min_gain = gain
				x = i
		return x, score

	cdef list _recursive_split_scoring( self, int start, int end, int no_split ):
		'''
		A copy of the _recursive_split method, but returning the score of the
		samples for each recursive sweep across the data.
		'''

		cdef int pseudostart, pseudoend, split_at = -1
		cdef np.ndarray score
		cdef list scores = [] 

		if no_split:
			split_at, score = self._best_split_stepwise_score( start, end )
			return list(score) 

		for pseudostart in xrange( start, end-2*self.min_width, self.window_width//2 ):
			if pseudostart > start + self.max_width:
				split_at = int_min( start+self.max_width, end-self.min_width )
				return scores + self._recursive_split_scoring( split_at, end, 0 )

			pseudoend = int_min( end, pseudostart+self.window_width )
			split_at, score = self._best_split_stepwise_score( pseudostart, pseudoend )
			scores.append( score )

			if split_at >= 0:
				break

		if split_at == -1:
			if end-start <= self.max_width:
				return scores
			split_at = int_min( start+self.max_width, end-self.min_width )

		return scores + self._recursive_split_scoring( start, split_at, 0 ) + \
			self._recursive_split_scoring( split_at, end, 0 )