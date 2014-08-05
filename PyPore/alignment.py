#!/usr/bin/env python
# Contact: Jacob Schreiber
#          jacobtribe@soe.ucsc.com
# alignment.py

'''
This module focuses on sequence alignment methods, including both dicrete alignment and
continuous alignment inputs, for pairwise and multiple sequence alignment. 
'''

from sklearn.neighbors import KernelDensity

import numpy as np
import math
import itertools as it 

from core import *
from parsers import *
from DataTypes import *

from PyPore.calignment import cSegmentAligner
from yahmm import *

NEGINF = -999999999

class SegmentAligner( object ):
	'''
	An aligner made to align ionic current segments based on mean, variance, and
	duration. The algorithm currently used is written by Dr. Kevin Karplus as a
	semi-local alignment along the sequence. 
	'''

	def __init__( self, model_means, model_stds, model_durs, skip_penalty, backslip_penalty ):
		self.aligner = cSegmentAligner( model_means, model_stds, model_durs, 
			skip_penalty, backslip_penalty )

	def align( self, seq_means, seq_stds, seq_durs ):
		'''
		Unpack the mean, standard deviation, and duration from the segment sequence.
		If those are not properties of the segment, the alignment cannot be done.
		'''

		try:
			return self.aligner.align( seq_means, seq_stds, seq_durs )
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
			s_segment = seq.segments[s_start]
			m = order[s_start]
			m_segment = model.segments[m]
			next_s = s_start + 1

			while next_s < seq.n and order[next_s] == m:
				next_s += 1

			if m < m_start:
				duration = -sum( [ seg.duration for seg in model.segments[m:m_start] ] )
				segments.append( (start_time, duration, seq.segments[s_start-1].mean, s_segment.mean, 0 ) )
				start_time += duration
				m_start = m

			total_seq_dur = sum( [ seg.duration for seg in seq.segments[s_start:next_s] ] )
			sum_model_dur = sum( [ seg.duration for seg in model.segments[m_start:m] ] )

			dur = sum_model_dur + s_segment.duration / total_seq_dur * m_segment.duration
			segments.append( ( start_time, dur, s_segment.mean, s_segment.mean, s_segment.std ) )
			start_time += dur

			remaining_m_dur = sum_model_dur + m_segment.duration - dur
			remaining_s_dur = total_seq_dur - s_segment.duration

			for s_start in xrange( s_start+1, next_s ):
				s_segment = seq.segments[s_start]
				dur = s_segment.duration / remaining_s_dur * remaining_m_dur
				segments.append( ( start_time, dur, s_segment.mean, s_segment.mean, s_segment.std ) )
				start_time += dur

			m_start = m+1
			s_start = next_s
		return segments

class PairwiseAligner( object ):
	'''
	This object will take in two sequences, and be able to perform pairwise sequence alignment
	using several different algorithms. It will load in a scoring function according to whatever
	type of sequence it is given. If it is given segments, it will load the SegmentScoreMixin,
	for strings, it will assume a protein alphabet, and for other it will score simply by
	identity. 
	'''

	def __init__( self, x, y ):
		self.x = x
		self.y = y
		self.m = len(self.x)
		self.n = len(self.y)

	def _score( self, x, y ):
		if x == '-' or y == '-':
			return 0
		return 3 - abs( x - y ) ** 2

	def dotplot( self ):
		'''
		Returns a matrix giving the similarity between every i, j point in the matrix. This is a
		symmetric matrix. 
		'''

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
		return seq_score, reversed(xalign), reversed(yalign) 


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
		return seq_score, reversed(xalign), reversed(yalign)

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
				yield seq_score, reversed(xalign), reversed(yalign)


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

class RepeatFinder( object ):
    '''
    This class will take in a sequence of segments and attempt to average tandem repeats in order
    to improve the accuracy of reading a specific segment. This assumes that tandem repeats are
    due to the enzyme falling backwards and reading the same sequence multiple times, not that
    the underlying sequence being read contains tandem repeats. 

    WARNING: If the underlying sequence does contain tandem repeats, do not use this class, or
    at least try to seperate out the tandem repeats beforehand. 
    '''

    def __init__( self, sequence ):
    	pass

class PSSM( object ):
	'''
	A position specific scoring matrix. For every position in a multiple sequence alignment,
	characterize the kernel density of the distribution of means. This is done by taking in
	the means for every segment in a column, characterizing the kernel density using a 
	ball-tree method and Gaussian kernel, and then allowing for pseudocounts. Much as a
	discrete PSSM would have a lookup table for each character in each specific position,
	a continuous PSSM will return the log probability according to the kernel density,
	adjusted by a small pseudocount. 
	'''

	def __init__( self, msa ):
		'''
		Upon receiving the MSA, transform it into the PSSM by taking the kernel density of every
		column. Pseudocounts should be given in linear space, not log space, and converted into
		log space upon initialization. 
		'''

		if hasattr( msa, '__iter__' ) and not hasattr( msa[0], '__iter__' ):
			msa = [ msa ]

		for profile in msa:
			if isinstance( profile, PSSM ):
				msa = [ seq for seq in it.chain( msa, profile.msa ) ]

		self.msa = msa
		self.consensus = []
		self.pssm = []

		offset = 0
		for i, column in enumerate( zip( *msa ) ):
			column = filter( lambda x: x is not '-', column )

			if len( column ) == 0:
				for seq in self.msa:
					del seq[i-offset]
				offset += 1
				continue

			self.pssm.append( [ mean for mean in column ] )
			self.consensus.append( np.mean( [mean for mean in column] ) )

	def __getitem__( self, slice ):
		'''
		When one is slicing the PSSM, they really want the underlying data in the PSSM. 
		'''

		return self.pssm[ slice ]

	def __repr__( self ):
		'''
		A string representation of the PSSM
		'''

		return '\n'.join( "{}".format( mean ) for mean in self.consensus )

	def __len__( self ):
		'''
		The number of positions in the PSSM.
		'''

		return len( self.pssm )


class ProfileAligner( object ):
	'''
	A HMM based alignment option. Allows you to align profiles to other profiles,
	where you can use as many sequences as you want for the master or the slave
	sequences, as opposed to the matrix aligner where you can only align one
	sequence to one other sequence at a time.
	'''

	def __init__( self, master, slave, bandwidth=1 ):
		'''
		Must take in a PSSM object or a list of whatever is being aligned. Both x and y must be a
		list of at least one list, where each inner list represents a sequence. For example:

		x = [ [ 'A', 'B', 'C', 'D', 'E' ],
              [ '-', '-', 'C', 'D', 'E' ],
              [ 'A', 'B', 'D', '-', '-' ] ]
        y = [ [ 'A', 'B', 'E', 'F', 'G' ] ]

        This means that simple pairwise comparison can be done by generating PSSMs where each
        character has a ~100% probability in its respective position. All alignments are
        generalized as profile alignments in this manner.
		'''

		self.bandwidth = bandwidth
		if not isinstance( master, PSSM ):
			self.master = PSSM( master )
		else:
			self.master = master

		if not isinstance( slave, PSSM ):
			self.slave = PSSM( slave )
		else:
			self.slave = slave

	def _build_global( self, pssm, low, high ):
		'''
		Build a profile HMM for finding global alignment on ionic current sequences, using a
		uniform distribution to model the insert state, and Gaussian distributions to represent
		the observed ionic current for each state.
		'''

		model = Model( name="Global Profile Aligner" )
		insert_dist = UniformDistribution( low, high )
		last_match = model.start
		last_insert = State( insert_dist, name="I0" )
		last_delete = None

		model.add_transition( model.start, last_insert, 0.15 )
		model.add_transition( last_insert, last_insert, 0.20 )

		for i, column in enumerate( pssm ):
			match = State( GaussianKernelDensity( column, self.bandwidth ), name="M"+str(i+1) ) 
			insert = State( insert_dist, name="I"+str(i+1) )
			delete = State( None, name="D"+str(i+1) )

			model.add_transition( last_match, match, 0.60 )
			model.add_transition( last_match, delete, 0.25 )
			model.add_transition( last_insert, match, 0.65 )
			model.add_transition( last_insert, delete, 0.20 )
			model.add_transition( delete, insert, 0.15 )
			model.add_transition( insert, insert, 0.15 )
			model.add_transition( match, insert, 0.15 )

			if last_delete is not None:
				model.add_transition( last_delete, match, 0.65 )
				model.add_transition( last_delete, delete, 0.20 )

			last_match, last_insert, last_delete = match, insert, delete

		model.add_transition( last_delete, model.end, 0.85 )
		model.add_transition( last_insert, model.end, 0.85 )
		model.add_transition( last_match, model.end, 0.85 )

		model.bake()
		return model

	def _build_local( self, pssm, low, high ):
		'''
		Build a profile HMM for finding local alignment on ionic current sequences, using a
		uniform distribution to model the insert state, and Gaussian distributions to represent
		the observed ionic current for each state.
		'''		

		model = Model( name="Local Profile Aligner" )
		insert_dist = UniformDistribution( low, high )
		m = len( pssm )

		# Build the beginning repeat of the HMM, representing sequence which does not align
		# before the local alignment
		start_insert = State( insert_dist, name="Q0" )
		start_delete = State( None, name="P0" )
		model.add_transition( model.start, start_insert, 0.5 )
		model.add_transition( model.start, start_delete, 0.5 )
		model.add_transition( start_insert, start_insert, 0.75 )
		model.add_transition( start_insert, start_delete, 0.25 )

		# Build the end repeat of the HMM, representing sequence which does not align after
		# the local alignment
		end_insert = State( insert_dist, name="QE" )
		end_delete = State( None, name="PE" )

		# Build the first column of the profile-repeat. 
		last_match = State( GaussianKernelDensity( pssm[0], self.bandwidth ), name="M0" )
		last_insert = State( insert_dist, name="I0" )
		last_delete = None
		model.add_transition( last_match, last_insert, 0.15 )
		model.add_transition( last_match, end_delete, 0.05 )
		model.add_transition( last_insert, last_insert, 0.20 )
		model.add_transition( start_delete, last_match, 1. / m )

		# Iterate through the middle portion of the PSSM to build the profile repeat
		for i, column in enumerate( pssm[1:-1] ):

			# Generate new states for the three possibilities
			match = State( GaussianKernelDensity( column, self.bandwidth ), name="M"+str(i+1) )
			insert = State( insert_dist, name="I"+str(i+1) )
			delete = State( None, name="D"+str(i+1) )

			# Add in the appropriate transitions
			model.add_transition( start_delete, match, 1. / m )
			model.add_transition( last_match, match, 0.65 )
			model.add_transition( last_match, delete, 0.15 )
			model.add_transition( last_insert, delete, 0.20 )
			model.add_transition( last_insert, match, 0.65 )
			model.add_transition( insert, insert, 0.15 )
			model.add_transition( delete, insert, 0.15 ) 
			model.add_transition( match, insert, 0.15 )
			model.add_transition( match, end_delete, 0.05 )

			# Allow for there being no delete state in the first column
			if last_delete is not None:
				model.add_transition( last_delete, match, 0.65 )
				model.add_transition( last_delete, delete, 0.20 )

			# Shift over to the next column
			last_match, last_insert, last_delete = match, insert, delete

		# Add in the last match 
		match = State( GaussianKernelDensity( pssm[-1], self.bandwidth ), name="M"+str(i+2) )
		model.add_transition( start_delete, match, 1. / m )
		model.add_transition( last_match, match, 0.80 )
		model.add_transition( last_insert, match, 0.85 )
		model.add_transition( last_delete, match, 0.85 )
		model.add_transition( match, end_delete, 1.00 )

		model.add_transition( end_delete, end_insert, 0.5 )
		model.add_transition( end_delete, model.end, 0.5 )
		model.add_transition( end_insert, end_insert, 0.75 )
		model.add_transition( end_insert, model.end, 0.25 )

		model.bake()
		return model

	def _build_repeat( self, pssm, low, high ):
		'''
		Build a profile HMM for finding alignments among two sequences where the slave may have
		additional tandem repeats built into it. A uniform distribution to model the insert state, 
		and Gaussian distributions to represent the observed ionic current for each state.
		'''		

		model = Model( name="Local Profile Aligner" )
		insert_dist = UniformDistribution( low, high )
		m = len( pssm )

		# Build the beginning repeat of the HMM, representing sequence which does not align
		# before the local alignment
		intermediate_insert = State( insert_dist, name="Q" )
		start_delete = State( None, name="P0" )
		end_delete = State( None, name="PE" )

		model.add_transition( model.start, start_delete, 0.5 )
		model.add_transition( model.start, intermediate_insert, 0.5 )
		model.add_transition( intermediate_insert, intermediate_insert, 0.50 )
		model.add_transition( intermediate_insert, start_delete, 0.25 )
		model.add_transition( intermediate_insert, model.end, 0.25 )
		model.add_transition( end_delete, intermediate_insert, 0.5 )
		model.add_transition( end_delete, model.end, 0.5 )

		# Build the first column of the profile-repeat. 
		last_match = State( GaussianKernelDensity( pssm[0], self.bandwidth ), name="M0" )
		last_insert = State( insert_dist, name="I0" )
		last_delete = None
		model.add_transition( last_match, last_insert, 0.15 )
		model.add_transition( last_match, end_delete, 0.05 )
		model.add_transition( last_insert, last_insert, 0.20 )
		model.add_transition( start_delete, last_match, 1. / m )

		# Iterate through the middle portion of the PSSM to build the profile repeat
		for i, column in enumerate( pssm[1:-1] ):

			# Generate new states for the three possibilities
			match = State( GaussianKernelDensity( column, self.bandwidth ), name="M"+str(i+1) )
			insert = State( insert_dist, name="I"+str(i+1) )
			delete = State( None, name="D"+str(i+1) )

			# Add in the appropriate transitions
			model.add_transition( start_delete, match, 1. / m )
			model.add_transition( last_match, match, 0.65 )
			model.add_transition( last_match, delete, 0.15 )
			model.add_transition( last_insert, delete, 0.20 )
			model.add_transition( last_insert, match, 0.65 )
			model.add_transition( insert, insert, 0.15 )
			model.add_transition( delete, insert, 0.15 ) 
			model.add_transition( match, insert, 0.15 )
			model.add_transition( match, end_delete, 0.05 )

			# Allow for there being no delete state in the first column
			if last_delete is not None:
				model.add_transition( last_delete, match, 0.65 )
				model.add_transition( last_delete, delete, 0.20 )

			# Shift over to the next column
			last_match, last_insert, last_delete = match, insert, delete

		# Add in the last match 
		match = State( GaussianKernelDensity( pssm[-1], self.bandwidth ), name="M"+str(i+2) )
		model.add_transition( start_delete, match, 1. / m )
		model.add_transition( last_match, match, 0.80 )
		model.add_transition( last_insert, match, 0.85 )
		model.add_transition( last_delete, match, 0.85 )
		model.add_transition( match, end_delete, 1.00 )

		model.bake()
		return model

	def global_alignment( self, low=0, high=60  ):
		'''
		Perform a global alignment using a HMM. This aligns two profiles two each other,
		returning the probability of the alignment, and the two consensus alignments. 
		'''

		profile = self._build_global( self.master, low, high )
		prob, states = profile.viterbi( self.slave.consensus )
		
		master = self.master
		slave = self.slave

		# Follow the slave's path through the master, ignoring start and end state
		for i, state in enumerate( states[1:-1] ):
			sname = state[1].name

			if sname.startswith( 'D' ):
				slave.pssm.insert( i, '-' )
				for seq in slave.msa:
					seq.insert( i, '-' )
			elif sname.startswith( 'I' ):
				master.pssm.insert( i, '-' )
				for seq in master.msa:
					seq.insert( i, '-' )  

		return prob, master, slave

	def local_alignment( self, low=0, high=60 ):
		'''
		Perform a local alignment using a HMM. This aligns two profiles to each other,
		returning the probability of the alignment, and the two consensus alignments, for the
		highest scoring alignment.
		'''

		profile = self._build_local( self.master, low, high )
		prob, states = profile.viterbi( self.slave.consensus )

		master = self.master
		slave = self.slave

		first_match = True
		offset = 0

		# Follow the slave's path through master
		for i, state in enumerate( states[1:-1] ):
			sname = state[1].name

			# If found first match, delete all sequence before that from the master
			if sname.startswith( "M" ) and first_match:
				first_match = False
				offset = int( sname[1:] )
				for j in xrange( offset ):
					for seq in master.msa:
						del seq[0]
					del master.pssm[0]
					del master.consensus[0]
			
			# If there is a deletion or insertion, add appropriate gaps
			if sname.startswith( 'D' ):
				slave.pssm.insert( i, '-' )
				for seq in slave.msa:
					seq.insert( i, '-' )
			elif sname.startswith( 'I' ):
				master.pssm.insert( i-offset, '-' )
				for seq in master.msa:
					seq.insert( i-offset, '-' )

			# If reached the last sequence, cut off the remainder of the slave sequence
			if sname is 'PE':
				for i in xrange( len( states ) - i - 3 ):
					for seq in slave.msa:
						del seq[-1]
				break

		return prob, master, slave

	def repeat_alignment( self, low=0, high=60 ):
		'''
		Repeat alignment, incomplete
		'''

		profile = self._build_repeat( self.master, low, high )
		prob, states = profile.viterbi( self.slave.consensus )

		master = self.master
		slave = self.slave

		first_match = True

		# Follow the slave's path through master
		for i, state in enumerate( states[1:-1] ):
			sname = state[1].name
			print sname

class MultipleSequenceAligner( object ):
	'''
	A HMM-based multiple sequence aligner. It begins by performing a naive alignment on the sequences
	and then performs iterative refinement to produce a better alignment.
	'''
	
	def __init__( self, sequences, bandwidth=1 ):
		self.sequences = sequences
		self.bandwidth = bandwidth

	def _score( self, msa ):
		'''
		The score for a discrete MSA would be done by trying to minimize entropy. For a continuous
		variable, this is the differential entropy. This assumes that the underlying distribution
		comes from a normal distribution.
		'''

		entropy = lambda col: 0.5*math.log( 2*np.pi*np.e*np.std( col )**2 ) if len(col) > 1 and np.std(col) > 0 else 0
		score = sum( 1. / ( len(col)-col.count('-') )**2 * entropy( filter( lambda x: x is not '-', col ) ) for col in it.izip( *msa ) )
		return score

	def iterative_alignment( self, epsilon=1e-4, max_iterations=10, bandwidth=1 ):
		'''
		Perform a HMM-based iterative alignment. If an initial alignment is provided, will use that
		to begin with, otherwise will simply use the sequences provided raw. This method will peel
		the top sequence off and align it to a profile of the other sequences, continuing this 
		method until there is little change in the score for a full round of iteration. The scoring 
		mechanism is done by minimum entropy.
		'''
		import sys
		# Unpack the initial sequences
		score, msa = self.iterative_initialization( bandwidth=bandwidth )

		if score == 0:
			return 0, msa

		n = len( msa )
		# Give initial scores
		last_score = float('inf')
		best_msa, best_score = msa, score

		# Until the scores converge...
		iteration = 0
		while abs( best_score-last_score ) >= epsilon and iteration < max_iterations:
			iteration += 1
			last_score = best_score
			# Run a full round of popping from the msa queue and enqueueing at the end
			for i in xrange( n ):
				# Pull a single 'master' sequence off the top of the msa queue
				slave = filter( lambda x: x is not '-', best_msa[i] )

				# Make the rest of them slaves
				master = best_msa[:i] + best_msa[i+1:]

				# Perform the alignment using the HMM-based profile aligner
				p, x, y = ProfileAligner( master=master, slave=slave, 
					bandwidth=bandwidth ).global_alignment()

				# Reattach the peeled sequence to the chain
				msa = [ seq for seq in it.chain( x.msa, y.msa ) ]

				# Calculate the score for this run
				score = self._score( msa )
				if score < best_score:
					best_msa, best_score = msa, score

		m = max( map( len, best_msa ) )
		for seq in best_msa:
			seq.extend( ['-']*(m-len(seq) ) )

		return score, best_msa

	def iterative_initialization( self, bandwidth=1 ):
		'''
		Create an initial MSA using an iterative alignment procedure of aligning
		new sequences one at a time. 
		'''

		pssm = PSSM( self.sequences[0] )
		for seq in self.sequences[1:]:
			p, master, slave = ProfileAligner( master=pssm, 
				slave=seq, bandwidth=bandwidth ).global_alignment()
			pssm = PSSM( [ seq for seq in it.chain( master.msa, slave.msa ) ] )

		return self._score( pssm.msa), pssm.msa


def NaiveTRF( seq, penalty=-1, min_score=2 ):
	'''
	Takes in a sequence with repeats that are based on rereading certain parts of an underlying segment.
	This is made up of two steps: (1) Running local alignment repeatedly, pulling out the highest ranking
	off-diagonal alignments. (2) Stitching these back together by generating a naive MSA.
	'''  

	def NaiveSplitter( seq, penalty, min_score ):
		'''
		This naive splitter will simply look for off-diagonal alignments which correspond to finding a
		local repeat.
		'''

		_score = lambda x, y: 2 - abs( x.mean - y.mean ) ** 2 / ( x.std * y.std ) 
		n = len(seq)

		score = np.zeros( ( n+1, n+1) )
		pointer = np.zeros( ( n+1, n+1 ) )
		mask = np.identity( n+1 )

		while True:
			for i in xrange( 1, n+1 ):
				for j in xrange( i, n+1 ):

					# Score function for local alignment, allowing for a mask to prevent certain alignments
					scores = ( 0, 
							   score[i-1, j-1] + _score( seq[i-1], seq[j-1] ) if not mask[i-1, j-1] else -999,
							   score[i, j-1] + penalty if not mask[i, j-1] else -999,
							   score[i-1, j] + penalty if not mask[i-1, j] else -999
	                         )

					# Fill in both triangles of the symmetric matrix with the best score
					score[i, j] = max( scores )
					score[j, i] = score[i, j]

					# Fill in the pointer matrix appropriately
					pointer[i, j] = scores.index( score[i, j] )
					pointer[j, i] = pointer[i, j]

			# Find the maximum scoring element of the matrix
			argmax = np.argmax( score )
			i, j = argmax/(n+1), argmax%(n+1)
			alignment_score = score[i, j]

			# If the score is not above the minimum threshold, exit
			if alignment_score <= min_score:
				break

			# Begin traceback to find the length
			length = 0
			while pointer[i, j] != 0:
				length += 1

				# Mask this cell so it won't be pulled in the next iteration
				mask[i, j] = mask[j, i] = 1

				# Unpack the pointer
				p = pointer[i, j]

				if p == 1:
					i -= 1
					j -= 1
				elif p == 2:
					j -= 1
				elif p == 3:
					i -= 1

			# End on the beginning, ensure it is masked for the next iteration
			mask[i, j] = 1

			# Yield the score, the length, and the x-y coordinates of the start of the alignment
			yield alignment_score, length, i, j

	splits = NaiveSplitter( seq=seq, penalty=penalty, min_score=min_score )
	last_end, last_start, offset, sequences = 0, 0, 0, []

	for score, length, start, end in sorted( splits, key=lambda x: x[3] ):
		# Pull each off-diagonal alignment, pad the start of each segment appropriately and save them
		sequences.append( ['-']*offset + seq[ last_end:end ] )

		# Update all parameters
		offset = ( len(sequences[-1]) - ( end-start ) if last_start != start else offset )
		last_end, last_start = end, start

	# Add in the remainder of the sequence
	sequences.append( ['-']*offset + seq[ last_end: ] )
	
	# Find the longest sequence which has been appended
	n = max( map( len, sequences ) )

	# Pad the ends of the sequences
	for i, seq in enumerate( sequences ):
		sequences[i] = seq + ['-']*( n-len(seq) )

	return sequences