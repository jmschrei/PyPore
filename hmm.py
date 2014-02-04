#!/usr/bin/env python
# Contact: Jacob Schreiber
#          jacobtribe@yahoo.com
# hmm.py

'''
This module serves as a collection for the various HMMs which are written for the nanopore lab.
HMMs must be written using Yet Another Hidden Markov Model (yahmm) package. When adding a new
HMM to the list, please follow the pattern set up, and also remember to update the factory at
the bottom. Use the format that the string is the same as the name of the function which returns
that HMM. 
'''

import pyximport
pyximport.install( setup_args={'include_dirs':np.get_include()})
from yahmm import *

def AbasicFinder():
	'''
	This HMM will attempt to find abasics, when abasics are seen as
	spikes of current up to approximately ~30 pA. Usual conditions
	this works for are 180 mV in a-haemolysin porin.
	'''

	model = Model( name="Abasic Finder" )

	s1 = State( NormalDistribution( 40.00, 7 ), name="1" )
	s2 = State( NormalDistribution( 20.00, 5 ), name="2" )
	s3 = State( NormalDistribution( 30.75, 2 ), name="3" )

	model.add_state( s1 )
	model.add_state( s2 )
	model.add_state( s3 )

	model.add_transition( model.start, s1, 1.00 )
	model.add_transition( s1, s2, 1.00 )
	model.add_transition( s2, s2, 0.80 )
	model.add_transition( s2, s3, 0.10 )
	model.add_transition( s3, s2, 0.80 )
	model.add_transition( s3, s3, 0.10 )
	model.add_transition( s2, model.end, 0.10 )
	model.add_transition( s3, model.end, 0.10 )
	
	model.bake()
	return model

def tRNAbasic_A8T0H11():
	model = Model( name="A8T0H115_Model")

	s1 = State( NormalDistribution( 40, 7.75 ), name="1", color='r' )
	s2 = State( NormalDistribution( 29.1, 1.15 ), name="2", color='y' )
	s3 = State( NormalDistribution( 24.29, 0.155), name="3", color='b' )
	s4 = State( NormalDistribution( 26.04, 0.352 ), name="4", color='b' )
	s5 = State( NormalDistribution( 23.8, 0.427 ), name="5", color='b' )
	s6 = State( NormalDistribution( 28.37, 0.464 ), name="6", color='g' )
	s7 = State( NormalDistribution( 26.8, 0.8678 ), name="7", color='b' )
	s8 = State( NormalDistribution( 25.4, 0.533 ), name="8", color='b' )
	s9 = State( NormalDistribution( 22.77, 0.481 ), name="9", color='b' )
	s10 = State( NormalDistribution( 30.06, 1.46 ), name="10", color='m' )
	s11 = State( NormalDistribution( 24.9, 2.19 ), name="11", color='b' )

	states = [ s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11 ]
	for state in states:
		model.add_state( state )

	model.add_transition( model.start, s1, 0.10 )
	model.add_transition( model.start, s2, 0.10 )
	model.add_transition( model.start, s3, 0.70 )
	model.add_transition( model.start, s4, 0.10 )
	model.add_transition( s1, s1, 0.10 )
	model.add_transition( s1, s2, 0.30 )
	model.add_transition( s1, s3, 0.60 )
	model.add_transition( s2, s2, 0.30 )
	model.add_transition( s2, s3, 0.70 )
	model.add_transition( s3, s3, 0.80 )
	model.add_transition( s3, s4, 0.20 )
	model.add_transition( s4, s4, 0.80 )
	model.add_transition( s4, s5, 0.20 )
	model.add_transition( s5, s5, 0.80 )
	model.add_transition( s5, s6, 0.20 )
	model.add_transition( s6, s6, 0.25 )
	model.add_transition( s6, s7, 0.70 )
	model.add_transition( s6, s8, 0.05 )
	model.add_transition( s7, s7, 0.30 )
	model.add_transition( s7, s8, 0.50 )
	model.add_transition( s7, model.end, 0.20 )
	model.add_transition( s8, s8, 0.70 )
	model.add_transition( s8, s9, 0.20 )
	model.add_transition( s8, model.end, 0.10 )
	model.add_transition( s9, s9, 0.70 )
	model.add_transition( s9, s10, 0.20 )
	model.add_transition( s9, model.end, 0.10 )
	model.add_transition( s10, s10, 0.10 )
	model.add_transition( s10, s11, 0.80 )
	model.add_transition( s10, model.end, 0.10 )
	model.add_transition( s11, s11, 0.5 )
	model.add_transition( s11, model.end, 0.5 )

	model.bake()
	return model


def Bifurcator():
	'''
	Attempts to parse an ionic current sequence that bifurcates between two
	ionic current states quickly, and many times.
	'''

	model = Model( name="Bifurcator" )

	s1 = State( NormalDistribution( 27.00, 1.00 ), name="1" )
	s2 = State( NormalDistribution( 33.00, 0.50 ), name="2" )

	model.add_state( s1 )
	model.add_state( s2 )

	model.add_transition( model.start, s1, 1.00 )
	model.add_transition( s1, s1, 0.45 )
	model.add_transition( s1, s2, 0.45 )
	model.add_transition( s1, model.end, 0.10 )
	model.add_transition( s2, s1, 0.45 )
	model.add_transition( s2, s2, 0.45 )
	model.add_transition( s2, model.end, 0.10 )

	model.bake()
	return model

hmm_factory = { 'Abasic Finder': AbasicFinder(),
		 		'Bifurcator': Bifurcator(),
		 		'tRNAbasic_A8T0H11': tRNAbasic_A8T0H11(),
		 	 }
