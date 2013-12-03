#!/usr/bin/env python
# Contact: Jacob Schreiber
#          jacobtribe@yahoo.com
# hmm.py

'''
This module allows for the extension of sklearn.hmm to be more applicable to reading nanopore
traces. This means creating a wrapper for the GaussianHMM class which handles event objects,
and allowing for scaling and clustering of ionic current levels in a specific way. This also
defines several specific HMMs in a function which returns the built NanoporeHMM. 
'''

from sklearn.hmm import GaussianHMM, _BaseHMM
from core import *
import string
import numpy as np

class NanoporeHMM( GaussianHMM ):
	'''
	A wrapper for GaussianHMM, which makes it more specific for nanopore experiments. This involves
	allowing for scaling (based on drift due to evaporation, or change in salt concentration), and
	for clustering an event in a series of states defined by consecutive labels of the hidden
	state, rather than a partitioner. 
	'''

	def scale( self, mult=None, add=None ):
		'''
		Scale the distribution of the Gaussian either multiplicatively or additively. 
		'''
		if mult:
			self._means_ *= mult
		if add:
			self._means_ += add

	def classify( self, event, algorithm = 'viterbi' ):
		'''
		Take an event which has been partitioned by a partitioner, and return a list of states
		which have been reduced to the span of hidden states. For example, if you have the following:

		A B C D E F G
		X X X Y Y Z X

		where A:G are observed segments in the event, and X:Z are hidden states defined in the HMM.
		Running this method would return:

		H I J K
		X Y Z X

		where H is a distribution that is the pdf addition of A B C, and I is the pdf addition of
		D and E. 
		'''

		means = np.array([ seg.mean for seg in event.segments ])
		means.shape = ( state_means.shape[0], 1 )

		_, states = self.decode( means, algorithm )

		reduced_segments, reduced_states = [], []

		i, j = 0, 0
		while i < len( states ) - 1:
			if states[i] != states[i+1]:
				duration = np.sum([ seg.duration for seg in event.segments[j:i+1] ] )
				mean = np.sum([  ])

			if state_sequence[i] != state_sequence[i+1]:
				duration = np.sum( [ seg.duration for seg in event.segments[j:i] ] )
				mean = np.sum( [seg.mean*seg.duration for seg in event.segments[j:i]] ) / duration
				std = np.sum([seg.std*seg.duration for seg in event.segments[j:i]]) / duration

				if event.__class__.__name__ == "MetaEvent":
					segments.append( MetaSegment( start=event.segments[j].start,
												  duration=duration,
												  mean=mean,
												  std=std,
												  event=event,
												  second=event.file.second,
												  hidden_state=state_sequence[j] ) )
				else:
					current = event.current[ event.segments[j].start: event.segments[i].start + event.segments[i].n ]
					segments.append( Segment( start=event.segments[j].start, 
											  current = current, 
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

def tRNAbasic_A8T0H115():
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
	means = np.array( [ [40],  [29.1], [24.29],[26.04],    [23.8], [28.37],[26.8],  [25.4], [22.77],[28.06+1],[24.9] ] )+5

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
		 		'tRNAbasic_A8T0H11': tRNAbasic_A8T0H11(),
		 		'tRNAbasic_A8T0H115' : tRNAbasic_A8T0H115() }