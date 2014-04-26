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
	return 0 if start == end else (c2[end-1]/end - (c[end-1]/end)**2) if start == 0 else ((c2[end-1]-c2[start-1])/(end-start) - ((c[end-1]-c[start-1])/(end-start))**2)

def pairwise(iterable):
	a, b = tee(iterable)
	next(b, None)
	return izip(a, b)

cdef class FastStatSplit:
	'''
	A cython implementation of the segmenter written by Kevin Karplus. Sped up approximately 50-100x
	compared to the Python implementation depending on parameters.
	'''

	cdef int min_width, max_width, window_width
	cdef double min_gain_per_sample
	cdef double [:] c, c2

	def __init__( self, min_width=1000, max_width=1000000, window_width=10000, min_gain_per_sample=0.05 ):
		self.min_width = min_width
		self.max_width = max_width
		self.window_width = window_width
		self.min_gain_per_sample = min_gain_per_sample

	def parse( self, current ):
		'''
		Wrapper function for the segmentation, which is implemented in cython.
		'''

		cdef list break_points
		cdef list paired
		self.c = np.cumsum( current )
		self.c2 = np.cumsum( np.multiply( current, current ) )

		breakpoints = self._segment_cumulative( 0, int(len(current)) )

		segments = [ Segment( current=current[start:end],
            start=start,
            duration=(end-start)/100000 ) for start, end in pairwise( chain([0],breakpoints,[len(current)]) ) ]

		return segments

	@cython.boundscheck(False)
	cdef int _best_split_stepwise( self, int start, int end ):
		'''
		Find the best split in a segment between start and end. Calculate best
		split by maximizing the change in variance, preferably by splitting a
		segment containing two segments into the two segments.
		'''

		if end-start <= 2*self.min_width:
			return -1 
		cdef double var_summed = (end - start) * log( var_c(start, end, self.c, self.c2) )
		cdef double max_gain = self.min_gain_per_sample * self.window_width
		cdef int i, x = -1
		cdef double low_var_summed, high_var_summed, gain

		for i in xrange( start+self.min_width, end+1-self.min_width ):
			low_var_summed = ( i-start ) * log( var_c( start, i, self.c, self.c2 ) )
			high_var_summed = ( end-i )  * log( var_c( i, end, self.c, self.c2 ) )
			gain = var_summed-( low_var_summed+high_var_summed )
			if gain > max_gain:
				max_gain = gain
				x = i
		return x

	cdef list _segment_cumulative( self, int start, int end ):
		'''
		Find the best splits recursively in the current until you get to the
		minimum width possible, and have looked at the entire 
		'''

		cdef int pseudostart, pseudoend, split_at = -1

		for pseudostart in xrange( start, end-2*self.min_width, self.window_width//2 ):
			if pseudostart > start + self.max_width:
				split_at = int_min( start+self.max_width, end-self.min_width )
				return [ split_at ] + self._segment_cumulative( split_at, end )

			pseudoend = int_min( end, pseudostart+self.window_width )
			split_at = self._best_split_stepwise( pseudostart, pseudoend )
			if split_at >= 0:
				break

		if split_at == -1:
			if end-start <= self.max_width:
				return []
			split_at = int_min( start+self.max_width, end-self.min_width )
		return self._segment_cumulative( start, split_at ) + [ split_at ] + self._segment_cumulative( split_at, end )
