#!/usr/bin/env python
# Contact: Jacob Schreiber
#          jacobtribe@yahoo.com
# alignment.py

'''
This module focuses on sequence alignment methods, including both dicrete alignment and
continuous alignment inputs, for pairwise and multiple sequence alignment. 
'''

import numpy as np
import math
import itertools as it 
from core import Segment
from parsers import *
from DataTypes import *
from read_abf import *

import pyximport
pyximport.install()
from PyPore.calignment import cSegmentAligner

def read_blosum_matrix( filename ):
	matrix = {}
	with open( filename, 'r' ) as infile:
		letters = infile.readline().strip().split()
		for i, line in enumerate( infile ):
			line = line.strip().split()
			if line[0] in letters:
				line = line[1:]
			matrix[ letters[i] ] = { key: float(val) for key, val in zip( letters, line ) }
	return matrix 	

class ProteinScoreMixin( object ):
	BLOSUM60 = read_blosum_matrix( "C:\Anaconda\Lib\site-packages\PyPore\\blosum60.txt")
	def _score( self, x, y, matrix='BLOSUM60' ):
		return getattr( ProteinScoreMixin, matrix )[x][y] 

class NucleotideScoreMixin( object ):
	def _score( self, x, y ):
		return None

class SegmentScoreMixin( object ):
	def _score( self, x, y ):
		t = math.fabs( x.mean - y.mean ) / math.sqrt( x.std * y.std )
		return 4 -  t

class DifferenceScoreMixin( object ):
	def _score( self, x, y ):
		return x == y 

class SegmentAligner( object ):
	def __init__( self, model, skip_penalty, backslip_penalty ):
		self.model = model
		means = np.array([ seg.mean for seg in model.states ])
		stds = np.array([ seg.std for seg in model.states ])
		durs = np.array([ seg.duration for seg in model.states ])
		self.aligner = cSegmentAligner( means, stds, durs, skip_penalty, backslip_penalty )
	def align( self, seq ):
		try:
			means = np.array([ seg.mean for seg in seq.states ])
			stds = np.array([ seg.std for seg in seq.states ])
			durs = np.array([ seg.duration for seg in seq.states ])
			return self.aligner.align( means, stds, durs )
		except ValueError:
			return None, None

	def transform( self, seq, order ):
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
		if type( x[0] ) == str:
			if len(x[0]) > 1:
				self.__class__ = type( 'PairwiseDifferenceAligner', ( PairwiseAligner, DifferenceScoreMixin) )
			else:
				self.__class__ = type( 'PairwiseProteinAligner', ( PairwiseAligner, ProteinScoreMixin ), {} )
		elif isinstance( x[0], Segment ):
			self.__class__ = type( 'PairwiseSegmentAligner', ( SegmentAligner ), {} )
		self.x = x
		self.y = y

	def global_alignment( self ):
		m,n = len(self.x), len(self.y)
		penalty = -1 
		# Initialize the matrix
		score = np.zeros( ( m+1, n+1 ) )
		pointer = np.zeros( ( m+1, n+1 ) )

		# Initialize the gaps in the matrix
		score[0, :] = np.arange(n+1) * penalty
		score[:, 0] = np.arange(m+1) * penalty
		pointer[0, :] = np.ones(n+1)
		pointer[:, 0] = np.ones(m+1)*2
		pointer[0, 0] = -1
		# Using Dynamic Programming to fill in the matrix-- with pointers as a trail of breadcrumbs
		for i in xrange( 1, m+1 ):
			for j in xrange( 1, n+1 ):
				scores = [ score[i-1][j-1] + self._score( self.x[i-1], self.y[j-1] ),
						   score[i][j-1] + penalty,
						   score[i-1][j] + penalty ]
				score[i][j] = max( scores )
				pointer[i][j] = scores.index( score[i][j] )
		alignment = self.traceback( pointer, self.x, self.y )
		return self._traceback( pointer, self.x, self.y), score[i][j]

	def _traceback( self, pointer, x, y ):
		i, j = pointer.shape[0] - 1, pointer.shape[1] - 1
		xalign, yalign = [], []
		while i > 0 and j > 0:
			if pointer[i][j] == 0:
				xalign.append( x[i-1] )
				yalign.append( y[j-1] )
				i -= 1
				j -= 1
			elif pointer[i][j] == 1:
				xalign.append( '-' )
				yalign.append( y[j-1] )
				j -= 1
			elif pointer[i][j] == 2:
				xalign.append( x[i-1] )
				yalign.append( '-' ) 
				i -= 1
		return xalign, yalign

if __name__ == '__main__':
	data = File( "C:\Users\Jacob\Desktop\Abada\\12811001-s06.abf" )
	print "parsing file"
	data.parse()
	print "parsing events"
	for event in data.events:
		event.parse( parser=StatSplit() )
	aligner = PairwiseAligner( data.events[0].states, data.events[1].states )
	(xalign, yalign), score = aligner.global_alignment()
	print xalign[0:5]
	print yalign[0:5]