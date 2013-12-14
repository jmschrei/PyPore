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
	This object will take in two sequences, and be able to perform pairwise sequence alignment
	using several different algorithms. It will load in a scoring function according to whatever
	type of sequence it is given. If it is given segments, it will load the SegmentScoreMixin,
	for strings, it will assume a protein alphabet, and for other it will score simply by
	identity. 
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
        '''
        Initiate by taking in a sequence containing tandem repeats due to enzyme activity, and
        store them. 
        '''

        self.sequence = sequence
        self.n = len(sequence)
    
    def _score( self, x, y ):
        '''
        This score function will take in two segments, and return a numerical representation of
        the distance between the two segments. This is t-score difference between the two
        segments, assuming both are samples drawn from normal distributions.
        '''

        return 1 if x == y else -1

    def _splitter( self, penalty, plots, min_score ):
        '''
        The first step in finding repeats is to attempt to find split points, defined to be
        between two randem repeats. For example:

        sequence = ABCDEFGHDEFGHDEFGHIJKLMNOPQRS
                   ABCDEFGH.DEFGH.DEFGHIJKLMNOPQRS ( Split points added in )

        This identifies split points according to a self-self local alignment, repeatedly taking 
        out alignments and realignments, taking out alignments until a minimum score is reached,
        yielding them one at a time as a generator. The diagonal is masked from the beginning.
        '''

        n = self.n
        seq = self.sequence

        score = np.zeros( ( n+1, n+1 ) )
        pointer = np.zeros( ( n+1, n+1 ) )
        mask = np.identity( n+1 )
        m = 0

        while True:
            for i in xrange( 1, n+1 ):
                for j in xrange( i, n+1 ):

                    # Score function for local alignment 
                    scores = ( 0, 
                                score[i-1, j-1] + self._score( seq[i-1], seq[j-1] ) if not mask[i-1, j-1] else -999,
                                score[i, j-1] + penalty if not mask[i, j-1] else -999,
                                score[i-1, j] + penalty if not mask[i-1, j] else -999
                            )
                    
                    # Take the maximum score for each cell
                    score[i, j] = max( scores )

                    # Since this is a symmetric matrix, update the transpose cell as well
                    score[j, i] = score[i, j]

                    # Put in the correct pointer
                    pointer[i, j] = scores.index( score[i, j] )
                    pointer[j, i] = pointer[i, j]
        
            # Identify the highest scoring cell on the map
            argmax = np.argmax( score )

            i, j = argmax/(n+1), argmax%(n+1)
            seq_score = score[i, j]
            
            # If the highest scoring cell is above a certain score, continue, else stop                
            if seq_score <= min_score:
                break
                
            # Follow the pointers back to a 0
            length = 0
            while pointer[i, j] != 0:
                length += 1
                mask[i, j] = mask[j, i] = 1
                p = pointer[i, j] 
                if p == 1:
                    i -= 1
                    j -= 1
                elif p == 2:
                    j -= 1
                elif p == 3:
                    i -= 1
            mask[i,j] = 1

            # Return a plot of the raw score, pointer matrix, and mask, for every iteration
            if plots:
                plt.figure( figsize=(20,10) )
                plt.subplot(131)
                plt.imshow( score, interpolation='nearest', cmap='Reds' )
                plt.subplot(132)
                plt.imshow( pointer, interpolation='nearest', cmap='Greens' )
                plt.subplot(133)
                plt.imshow( mask, interpolation='nearest', cmap='Purples' )        
                plt.show()

            # Yield the score, length of the repeat, and i, j index of the cell before reaching 0
            yield seq_score, length, i, j

    def _naive_msa( self, penalty, plots, min_score=4 ):
        '''
        Take in a sequence and return a naive MSA based on splitting the sequences, and sliding
        them over in order to make a basic MSA to start an iterative alignment.
        '''

        splits = self._splitter( penalty, plots, min_score )

        last_end, last_start, offset, sequences = 0, 0, 0, []
        for score, length, start, end in sorted( splits, key=lambda x: x[3] ):
            padded_sequence = ['-']* offset + self.sequence[ last_end:end ]
            sequences.append( padded_sequence )

            offset = ( len(sequences[-1]) - ( end-start ) if last_start != start else offset )
            last_end, last_start = end, start

        sequences.append( ['-']*offset + self.sequence[ last_end: ] )
        n = max( map( len, sequences ) )

        for i, seq in enumerate( sequences ):
            sequences[i] = seq + ['-']*( n-len(seq) )

        return sequences

    def _msa_to_pssm( self, msa, pseudocount=1e-4 ):
        '''
        Take in a multiple sequence alignment as a list of lists, and returns a list of
        dictionaries, where the dictionaries represent the discrete frequencies of
        various characters. 
        '''
        alphabet = "QWERTYUIOPASDFGHJKLZXCVBNM"
        pssm = []

        for position in it.izip( *msa ):
            fpos = filter( lambda x: x is not '-', position )
            if len(fpos) == 0:
                continue

            pcounts = { char: fpos.count(char) for char in fpos }
            apcounts = { key: pcounts[key]+1e-2 if key in pcounts else 1e-2 for key in alphabet }
            n = sum( apcounts.values() )
            pssm.append({ key: val / n for key, val in apcounts.items()}) 

        return pssm

    def _pssm_to_hmm( self, pssm, name="Underlying Profile" ):
        '''
        Take in a MSA, using it as a PSSM, and generate a HMM from it using frequencies in the
        PSSM as the emission probabilities for the HMM. Since this is a profile HMM, it follows
        the structure outlined in Durbin, Eddy, Krogh, and Mitchinson's "Biological Sequence
        Analysis", pg. 106. 
        '''

        model = Model( name=name )
        insert_dist = { char: 1. / 26 for char in string.ascii_uppercase }

        last_match = model.start
        last_insert = State( DiscreteDistribution( insert_dist ), name="I0" )
        last_delete = None

        model.add_transition( model.start, last_insert, 0.15 )
        model.add_transition( last_insert, last_insert, 0.20 )

        for i, position in enumerate( pssm ):
            match = State( DiscreteDistribution( position ), name="M"+str(i+1) ) 
            insert = State( DiscreteDistribution( insert_dist ), name="I"+str(i+1) )
            delete = State( None, name="D"+str(i+1) )

            model.add_transition( last_match, match, 0.60 )
            model.add_transition( last_match, delete, 0.25 )
            model.add_transition( last_insert, match, 0.60 )
            model.add_transition( last_insert, delete, 0.20 )
            model.add_transition( delete, insert, 0.15 )
            model.add_transition( insert, insert, 0.20 )
            model.add_transition( match, insert, 0.15 )

            if last_delete:
                model.add_transition( last_delete, match, 0.60 )
                model.add_transition( last_delete, delete, 0.25 )

            last_match, last_insert, last_delete = match, insert, delete

        model.add_transition( last_delete, model.end, 0.80 )
        model.add_transition( last_insert, model.end, 0.80 )
        model.add_transition( last_match, model.end, 0.85 )

        model.bake()
        return model           
    
    def _iterate( self, msa, epsilon ):
        '''
        Perform iterative multiple sequence alignment, by removing a single sequence and realigning
        it to the profile. It will do this by removing a single sequence, building a profile HMM
        using the remaining sequenes, and then using the HMM to align the removed sequence back
        into the MSA.
        '''

        msa = [ [ 'A', 'B', 'C', 'D', 'E', 'F', '-', '-', '-', '-', '-' ],
                [ '-', '-', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L' ],
                [ '-', '-', 'C', 'D', 'F', 'H', '-', '-', '-', '-', '-' ],
                [ '-', '-', 'C', 'D', 'E', 'F', 'H', 'I', 'J', 'K', '-' ],
                [ 'K', 'L', 'M', 'N', 'O', '-', '-', '-', 'P', 'Q', 'R' ] ]

        difference, last_difference = 0, 10
        while abs( difference - last_difference ) >= epsilon:
            print
            for seq in msa:
                print ''.join( seq )

            seq = [ char for char in msa[ 0 ] if char != '-' ]
            msa = msa[ 1: ] 

            pssm = self._msa_to_pssm( msa )
            profile = self._pssm_to_hmm( pssm )
            prob, states = profile.viterbi( seq )

            aligned_seq = []
            inserts = []
            i, j = 0, 0
            for state in states[1:-1]:
                state = state[1].name # Unpack the name of the node
                #cons = sorted( pssm[j].items(), key=lambda x: x[1], reverse=True )[0][0]

                if state[0] == 'M':
                    #print "{0:4}{1:4}{2:4}".format( seq[i], cons, state )
                    aligned_seq.append( seq[i] )
                    i += 1
                    j += 1
                elif state[0] == 'D':
                    #print "{0:4}{1:4}{2:4}".format( '-', cons, state )
                    aligned_seq.append( '-' )
                    j += 1
                elif state[0] == 'I':
                    #print "{0:4}{1:4}{2:4}".format( seq[i], '-', state )
                    aligned_seq.append( seq[i] )    
                    i += 1

            last_difference -= 1
            msa.append( aligned_seq )

            n = max( map( len, msa ) )
            for seq in msa:
                seq.extend( ['-']*(n-len(seq)) )

        '''
        pssm = self._msa_to_pssm( msa )
        i, j = 0, 0
        for state in states[1:-1]:
            state = state[1].name
            if state[0] == 'M':
                print "{0:4}{1:4}{2:4}".format( seq[i], profile, state )
                i += 1
                j += 1
            elif state[0] == 'D':
                print "{0:4}{1:4}{2:4}".format( '-', profile, state )
                j += 1
            elif state[0] == 'I':
                print "{0:4}{1:4}{2:4}".format( seq[i], '-', state )
                i += 1
        '''
        return msa

    def compute( self, penalty=-1, plots=False, min_score=4, epsilon=1e-8 ):
        '''
        Client function that will allow you to do stuff.
        '''

        initial_MSA = self._naive_msa( penalty=penalty, plots=plots, min_score=min_score )
        consensus = self._iterate( msa=initial_MSA, epsilon=epsilon )


class MultipleSequenceAlignment( object ):
	'''
	This object will take in a series of sequences and attempt to align them all as a multiple
	sequence alignment. It uses iterative alignment, taking in an initial MSA and peeling off
	one sequence at a time and realigning them. 
	'''

	def __init__( self, sequences ):
		self.sequences

	def _msa_to_pssm( self, sequences, pseudocounts=1e-4 ):
		'''
		This takes in a given msa and generates a PSSM for it. For testing purposes, the prior
		will be a uniform distribution across the alphabet. The prior distribution will otherwise
		be the kernel density of points, using a gaussian kernel. 
		'''
		pass

    def _pssm_to_hmm( self, pssm, name="Underlying Profile" ):
        '''
        Take in a MSA, using it as a PSSM, and generate a HMM from it using frequencies in the
        PSSM as the emission probabilities for the HMM. Since this is a profile HMM, it follows
        the structure outlined in Durbin, Eddy, Krogh, and Mitchinson's "Biological Sequence
        Analysis", pg. 106. 
        '''

        model = Model( name=name )
        insert_dist = { char: 1. / 26 for char in string.ascii_uppercase }

        last_match = model.start
        last_insert = State( DiscreteDistribution( insert_dist ), name="I0" )
        last_delete = None

        model.add_transition( model.start, last_insert, 0.15 )
        model.add_transition( last_insert, last_insert, 0.20 )

        for i, position in enumerate( pssm ):
            match = State( DiscreteDistribution( position ), name="M"+str(i+1) ) 
            insert = State( DiscreteDistribution( insert_dist ), name="I"+str(i+1) )
            delete = State( None, name="D"+str(i+1) )

            model.add_transition( last_match, match, 0.60 )
            model.add_transition( last_match, delete, 0.25 )
            model.add_transition( last_insert, match, 0.60 )
            model.add_transition( last_insert, delete, 0.20 )
            model.add_transition( delete, insert, 0.15 )
            model.add_transition( insert, insert, 0.20 )
            model.add_transition( match, insert, 0.15 )

            if last_delete:
                model.add_transition( last_delete, match, 0.60 )
                model.add_transition( last_delete, delete, 0.25 )

            last_match, last_insert, last_delete = match, insert, delete

        model.add_transition( last_delete, model.end, 0.80 )
        model.add_transition( last_insert, model.end, 0.80 )
        model.add_transition( last_match, model.end, 0.85 )

        model.bake()
        return model 
