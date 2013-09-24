#!/usr/bin/env python
# Contact: Jacob Schreiber
#          jacobtribe@yahoo.com
# hmm.py

'''
hmm.py contains several hidden markov models which can be used to classify
current traces specific to nanopore data.
'''

from sklearn.hmm import GaussianHMM, _BaseHMM
from core import Segment
import string
import numpy as np

class NanoporeHMM( GaussianHMM ):
	def scale( self, salt_concentration, voltage ):
		self.means_prior *= salt_concentration * voltage
	def classify( self, event, algorithm = 'viterbi' ):
		state_means = np.array( [ state.mean for state in event.states ] )
		state_means.shape = ( state_means.shape[0], 1 )
		_, state_sequence = self.decode( state_means, algorithm)

		new_state_sequence = []
		segments = [] 
		i, j = 0, 0
		while i < len( state_sequence ) - 1:
			if state_sequence[i] != state_sequence[i+1]:
				current = event.current[ event.states[j].start: event.states[i].start + event.states[i].n ]
				segments.append( Segment( start=event.states[j].start, current = current, 
					                      event = event, second = event.file.second,
					                      hidden_state = state_sequence[j]
					                     ) )
				new_state_sequence.append( state_sequence[j] )
				j = i + 1
			i += 1
		return new_state_sequence, segments 

def AbasicFinder():
	'''
	Attempts to decode an ionic current sequence to tell the number
	of abasic residues which have passed through the pore. This assumes
	being fed superpoints representing states, and probabilities are
	scaled as such. 
	'''
	# Index 0: Random Current
	# Index 1: Abasic Residue
	n_components = 3
	startprob = np.array( [ 1, 0, 0 ] )
	transmat = np.array( [ [ 0.00, 1.00, 0.00 ],
				           [ 0.05, 0.80, 0.15 ],
				           [ 0.00, 0.90, 0.10 ] ] )
	covars = np.array( [ [ 7 ],
			             [ 5 ],
			             [ 2 ] ] )
	means = np.array( [ [ 40 ],
			            [ 20 ],
			            [ 30.75 ] ] )

	hmm = NanoporeHMM( n_components=n_components, startprob=startprob, transmat=transmat )
	hmm._means_ = means
	hmm._covars_ = covars 
	return hmm

def tRNAbasic_A8T0H7():
	n_components=7
	startprob = np.array([ 0.8, 0.15, 0.05, 0.0, 0.0, 0.0, 0.0] )
	transmat = np.array([[ 0.1, 0.2, 0.7, 0.0, 0.0, 0.0, 0.0 ],
						 [ 0.0, 0.1, 0.9, 0.0, 0.0, 0.0, 0.0 ],
						 [ 0.0, 0.0, 0.8, 0.2, 0.0, 0.0, 0.0 ],
						 [ 0.0, 0.0, 0.0, 0.1, 0.9, 0.0, 0.0 ],
						 [ 0.0, 0.0, 0.0, 0.0, 0.7, 0.2, 0.1 ],
						 [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.9 ],
						 [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0 ]])

	covars = np.array([[10],[1.42],[1.54],[1.18],[1.63],[1.32],[1.53]])
	means = np.array( [ [44],[29.1],[24.4],[28.64],[25],[29.39],[24.47] ] )

	hmm = NanoporeHMM( n_components=n_components, startprob=startprob, transmat=transmat )
	hmm._means_ = means
	hmm._covars_ = covars
	hmm.colors = [ 'r', 'g', 'b', 'g', 'b', 'g', 'b' ]
	return hmm

def tRNAbasic_A8T0H11():
	n_components=11

	startprob = np.array([ 0.20, 0.15, 0.55, 0.10, 0, 0, 0, 0, 0, 0, 0 ])

	transmat = np.array([[ 0.10, 0.30, 0.60, 0, 0, 0, 0, 0, 0, 0, 0 ],
						 [ 0, 0.30, 0.70, 0, 0, 0, 0, 0, 0, 0, 0 ],
						 [ 0, 0, 0.80, 0.20, 0, 0, 0, 0, 0, 0, 0 ],
						 [ 0, 0, 0, 0.80, 0.20, 0, 0, 0, 0, 0, 0 ],
						 [ 0, 0, 0, 0, 0.80, 0.20, 0, 0, 0, 0, 0 ],
						 [ 0, 0, 0, 0, 0, 0.25, 0.70, 0.05, 0, 0, 0 ],
						 [ 0, 0, 0, 0, 0, 0, 0.30, 0.70, 0, 0, 0 ],
						 [ 0, 0, 0, 0, 0, 0, 0, 0.80, 0.20, 0, 0 ],
						 [ 0, 0, 0, 0, 0, 0, 0, 0, 0.80, 0.20, 0 ],
						 [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.20, 0.80 ],
						 [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.00 ]])

	covars = np.array([ [7.75],[1.15], [0.155],[0.352-.15],[0.427],[0.464],[0.8678],[0.533],[0.481],[1.46],   [2.19]] )
	means = np.array( [ [40],  [29.1], [24.29],[26.04],    [23.8], [28.37],[26.8],  [25.4], [22.77],[28.06+1],[24.9] ] )

	hmm = NanoporeHMM( n_components=n_components, startprob=startprob, transmat=transmat )
	hmm._means_ = means
	hmm._covars_ = covars
	hmm.colors = [ 'r', 'y', 'b', 'b', 'b', 'g', 'b', 'b', 'b', 'm', 'b' ]
	return hmm

def Bifurcator():
	'''
	Attempts to parse an ionic current sequence that bifurcates between two
	ionic current states quickly, and many times. This assumes being fed 
	superpoints which represent segmented states. 
	'''
	n_components = 2
	startprob = np.array( [ 1, 0 ] )
	transmat = np.array( [ [ .5, .5 ],
						   [ .5, .5 ] ] )
	means = np.array( [ [ 27.0 ], 
	          			[ 33.0 ] ] )
	covars = np.array( [ [ 1.0 ],
			   		     [ 0.5 ] ] )

	hmm = NanoporeHMM( n_components = n_components )
	hmm.startprob = startprob
	hmm.transmat = transmat
	hmm._means_ = means
	hmm._covars_ = covars
	return hmm

hmm_factory = { 'Abasic Finder': AbasicFinder(),
		 		'Bifurcator': Bifurcator(),
		 		'tRNAbasic_A6T0H7': tRNAbasic_A8T0H7(),
		 		'tRNAbasic_A6T0H11': tRNAbasic_A8T0H11() }