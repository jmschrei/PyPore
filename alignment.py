#!/usr/bin/env python
# Contact: Jacob Schreiber
#          jacobtribe@soe.ucsc.com
# alignment.py

'''
This module focuses on sequence alignment methods, including both dicrete alignment and
continuous alignment inputs, for pairwise and multiple sequence alignment. 
'''

import numpy as np
import math
import itertools as it 

from scipy.cluster.hierarchy import linkage
from core import *
from parsers import *
from DataTypes import *

import pyximport
pyximport.install()
from PyPore.calignment import cSegmentAligner

NEGINF = -999999999

class SegmentScoreMixin( object ):
	def _score( self, x, y ):
		if x == '-' or y == '-':
			return 0
		t = np.max( ( x.mean-y.mean )**2 / ( x.std*y.std ) )
		return 3 - t ** 2

class DifferenceScoreMixin( object ):
	def _score( self, x, y ):
		return x == y 

class SegmentAligner( object ):
	'''
	An aligner made to align ionic current segments based on mean, variance, and
	duration. The algorithm currently used is written by Dr. Kevin Karplus as a
	semi-local alignment along the sequence. 
	'''
	def __init__( self, model, skip_penalty, backslip_penalty ):
		self.model = model
		means = np.array([ seg.mean for seg in model.states ])
		stds = np.array([ seg.std for seg in model.states ])
		durs = np.array([ seg.duration for seg in model.states ])
		self.aligner = cSegmentAligner( means, stds, durs, skip_penalty, backslip_penalty )

	def align( self, seq ):
		'''
		Unpack the mean, standard deviation, and duration from the segment sequence.
		If those are not properties of the segment, the alignment cannot be done.
		'''
		try:
			means = np.array([ seg.mean for seg in seq.states ])
			stds = np.array([ seg.std for seg in seq.states ])
			durs = np.array([ seg.duration for seg in seq.states ])
			return self.aligner.align( means, stds, durs )
		except ValueError:
			return None, None

	def transform( self, seq, order ):
		'''
		Transform all of the sequences to align to the model better. Lengthens
		the duration of some segments to match the model. Mean and std remain 
		the same.
		'''
		if seq == None or order == None:
			return None
		model = self.model 
		segments = []
		start_time = 0
		m_start = 0
		s_start = 0
		while s_start < seq.n and m_start < model.n:
			s_state = seq.states[s_start]
			m = order[s_start]
			m_state = model.states[m]
			next_s = s_start + 1

			while next_s < seq.n and order[next_s] == m:
				next_s += 1

			if m < m_start:
				duration = -sum( [ seg.duration for seg in model.states[m:m_start] ] )
				segments.append( (start_time, duration, seq.states[s_start-1].mean, s_state.mean, 0 ) )
				start_time += duration
				m_start = m

			total_seq_dur = sum( [ seg.duration for seg in seq.states[s_start:next_s] ] )
			sum_model_dur = sum( [ seg.duration for seg in model.states[m_start:m] ] )

			dur = sum_model_dur + s_state.duration / total_seq_dur * m_state.duration
			segments.append( ( start_time, dur, s_state.mean, s_state.mean, s_state.std ) )
			start_time += dur

			remaining_m_dur = sum_model_dur + m_state.duration - dur
			remaining_s_dur = total_seq_dur - s_state.duration

			for s_start in xrange( s_start+1, next_s ):
				s_state = seq.states[s_start]
				dur = s_state.duration / remaining_s_dur * remaining_m_dur
				segments.append( ( start_time, dur, s_state.mean, s_state.mean, s_state.std ) )
				start_time += dur

			m_start = m+1
			s_start = next_s
		return segments

class PairwiseAligner( object ):
	'''
	This object will take in two sequences, and be able to perform pairwise
	sequence alignment using several different algorithms.
	'''
	def __init__( self, x, y ):
		if isinstance( x[0], Segment ) or isinstance( x[0], MetaSegment ):
			self.__class__ = type( 'PairwiseSegmentAligner', ( PairwiseAligner, SegmentScoreMixin ), {} )
		elif type( x[0] ) == str:
			self.__class__ = type( 'PairwiseProteinAligner', ( PairwiseAligner, ProteinScoreMixin ), {} )
		else:
			self.__class__ = type( 'PairwiseDifferenceAligner', ( PairwiseAligner, DifferenceScoreMixin), {} )
		self.x = x
		self.y = y
		self.m = len(self.x)
		self.n = len(self.y)

	def dotplot( self ):
		score = np.zeros(( self.m+1, self.n+1 ))

		for i in xrange( 1, self.m+1 ):
			for j in xrange( 1, self.n+1 ):
				score[i, j] = self._score( self.x[i-1], self.y[j-1] )

		return score 

	def _global_alignment_matrix( self, penalty=-1 ):
		'''
		Creates an alignment matrix according to the Needleman-Wunch global alignment algorithm
		between two sequences. It will return the raw score matrix and the pointer matrix. 
		'''
		# Initialize the matrix
		score = np.zeros( ( self.m+1, self.n+1 ) )
		pointer = np.zeros( ( self.m+1, self.n+1 ) )

		# Initialize the gaps in the matrix
		score[0, :] = np.arange(self.n+1) * penalty
		score[:, 0] = np.arange(self.m+1) * penalty
		pointer[0, :] = np.ones(self.n+1)
		pointer[:, 0] = np.ones(self.m+1)*2
		pointer[0, 0] = -1
		# Using Dynamic Programming to fill in the matrix-- with pointers as a trail of breadcrumbs
		for i in xrange( 1, self.m+1 ):
			for j in xrange( 1, self.n+1 ):
				scores = ( score[i-1, j-1] + self._score( self.x[i-1], self.y[j-1] ),
						   score[i, j-1] + penalty,
						   score[i-1, j] + penalty 
						  )
				score[i, j] = max( scores )
				pointer[i, j] = scores.index( score[i, j] )
		return score, pointer

	def _global_alignment_traceback( self, score, pointer ):
		'''
		Follows a traceback, starting at the bottom right corner and working back, according to
		global alignment. 
		'''
		i, j = pointer.shape[0] - 1, pointer.shape[1] - 1
		seq_score = score[i, j]
		xalign, yalign = [], []

		while i > 0 and j > 0:
			if pointer[i][j] == 0:
				xalign.append( self.x[i-1] )
				yalign.append( self.y[j-1] )
				i -= 1
				j -= 1
			elif pointer[i][j] == 1:
				xalign.append( '-' )
				yalign.append( self.y[j-1] )
				j -= 1
			elif pointer[i][j] == 2:
				xalign.append( self.x[i-1] )
				yalign.append( '-' ) 
				i -= 1
		return ( xalign, yalign ), seq_score 


	def global_alignment( self, penalty=-1 ):
		'''
		Client access to the global alignment methods. This will take in a penalty term
		and align the two sequences passed in upon initialization.
		'''
		score, pointer = self._global_alignment_matrix( penalty=penalty )
		return self._global_alignment_traceback( score, pointer )


	def _local_alignment_matrix( self, penalty ):
		'''
		Creates an alignment matrix according to the Smith-Waterman global alignment algorithm
		between two sequences. It will return the raw score matrix and the pointer matrix. 
		'''
		score = np.zeros( ( self.m+1, self.n+1 ) )
		pointer = np.zeros( ( self.m+1, self.n+1 ) )

		for i in xrange( 1, self.m+1 ):
			for j in xrange( 1, self.n+1 ):
				idx_scores = (     0,
								   score[i-1, j-1] + self._score( self.x[i-1], self.y[j-1] ),
								   score[i, j-1] + penalty,
								   score[i-1, j] + penalty 
							 )
				score[i, j] = max( idx_scores )
				pointer[i, j] = idx_scores.index( score[i, j] )

		return score, pointer

	def _local_alignment_traceback( self, score, pointer ):
		'''
		Follows a traceback, starting at the highest score function anywhere in the matrix, working
		back until it hits a 0 in the score matrix.
		'''
		xalign, yalign = [], []

		argmax = np.argmax( score )
		i, j = argmax/(self.n+1), argmax%(self.n+1)

		seq_score = score[i, j]

		while pointer[i, j] != 0:
			p = pointer[i, j]
			pointer[i, j] = 0
			pointer[j, i] = 0
			score[i, j] = NEGINF
			score[j, i] = NEGINF
			if p == 1:
				xalign.append( self.x[i-1] )
				yalign.append( self.y[j-1] )
				i -= 1
				j -= 1
			elif p == 2:
				xalign.append( '-' )
				yalign.append( self.y[j-1] )
				j -= 1
			elif p == 3:
				xalign.append( self.x[i-1] )
				yalign.append( '-' )
				i -= 1

		while xalign[-1] == '-' or yalign[-1] == '-':
			xalign = xalign[:-1]
			yalign = yalign[:-1]
		return (xalign, yalign), seq_score

	def _local_alignment_repeated_traceback( self, score, pointer, min_length ):
		'''
		Follows a traceback, starting at the highest score function anywhere in the matrix, working
		back until it hits a 0 in the score matrix. It will repeat this process, zeroing out every
		alignment that it pulls out, pulling sequences in order of score.
		'''
		while True:
			xalign, yalign = [], []

			argmax = np.argmax( score )
			i, j = argmax/(self.n+1), argmax%(self.n+1)
			if pointer[i, j] == 0:
				break

			seq_score = score[i, j]

			while pointer[i, j] != 0:
				p = pointer[i, j]
				pointer[i, j] = 0
				pointer[j, i] = 0
				score[i, j] = NEGINF
				score[j, i] = NEGINF
				if p == 1:
					xalign.append( self.x[i-1] )
					yalign.append( self.y[j-1] )
					i -= 1
					j -= 1
				elif p == 2:
					xalign.append( '-' )
					yalign.append( self.y[j-1] )
					j -= 1
				elif p == 3:
					xalign.append( self.x[i-1] )
					yalign.append( '-' )
					i -= 1

			if len(xalign) < min_length:
				continue
			else:
				while xalign[-1] == '-' or yalign[-1] == '-':
					xalign = xalign[:-1]
					yalign = yalign[:-1]
				yield (xalign, yalign), seq_score


	def local_repeated_alignment( self, penalty=-1, min_length=2 ):
		'''
		Client function for the local repeated alignment. Performs Smith-Waterman on the two
		stored sequences, then returns all alignments in order of score.
		'''
		score, pointer = self._local_alignment_matrix( penalty )
		return self._local_alignment_repeated_traceback( score, pointer, min_length )

	def local_alignment( self, penalty=-1 ):
		'''
		Client function for the local repeated alignment. Performs Smith-Waterman on the two
		stored sequences, then returns the best alignment anywhere in the matrix.
		'''
		score, pointer = self._local_alignment_matrix( penalty )
		return self._local_alignment_traceback( score, pointer )

class RepeatFinder():
	def __init__( self, event ):
		self.segments = event.segments
		self.event = event
		self.n = event.n

	def _local_repeated_traceback_arg( self, s, p ):
		while True:
			argmax = np.argmax( score )
			i, j = argmax/(self.n+1), argmax%(self.m+1)
			if pointer[i, j] == 0:
				break

			seq_score = score[i, j]

			while pointer[i, j] != 0:
				p = pointer[i, j]
				pointer[i, j] = 0
				pointer[j, i] = 0
				score[i, j] = NEGINF
				score[j, i] = NEGINF
				if p == 1:
					i -= 1
					j -= 1
				elif p == 2:
					j -= 1
				elif p == 3:
					i -= 1

			if len(xalign) < min_length:
				continue
			else:
				yield (i, j), seq_score


	def make_consensus( self ):
		aligner = PairwiseAligner( self.event, self.event )
		s, p = aligner._local_alignment_matrix()


class PhylogeneticTree( object ):
	def __init__( self, seqs ):
		self.seqs = seqs
		self.n = len( self.seqs )
	def grow( self ):
		matrix = np.zeros( ( self.n,self.n ) )
		for i in xrange( self.n ):
			for j in xrange( self.n ):
				if j < i:
					_, matrix[i, j] = PairwiseAligner( self.seqs[i], self.seqs[j] ).local_alignment()
					matrix[j, i] = matrix[i, j]

		for xid, yid, dist, n in linkage( matrix, method='weighted' ):
			self.seqs.append( PairwiseSegmentConsensus( self.seqs[ int(xid) ], self.seqs[ int(yid) ] ) )
		
		print self.seqs[-1]

def PairwiseSegmentConsensus( x, y ):
	assert x.__class__ == y.__class__
	consensus = []
	for xseg, yseg in it.izip( x, y ):
		if isinstance( xseg, Segment ) and isinstance( yseg, Segment ):
			consensus.append( MetaSegment( mean=(xseg.mean+yseg.mean) / 2,
						  				   std=math.sqrt(xseg.std**2+yseg.std**2),
						  				   duration=(xseg.duration+yseg.duration) / 2 )  )
		elif not isinstance( xseg, Segment ):
			consensus.append( yseg )
		else:
			consensus.append( xseg )
	return consensus 
