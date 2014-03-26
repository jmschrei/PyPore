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

import numpy as np
import pyximport
pyximport.install( setup_args={'include_dirs':np.get_include()})
from yahmm import *

def Phi29ProfileHMM( distributions, name="Phi29 Profile HMM", low=0, high=50 ):
	"""
	Build a profile HMM based on a list of distributions. Includes:
		* Short backslips
	"""

	model = Model( name="{}-{}".format( name, len(distributions) ) )

	insert_distribution = UniformDistribution( low, high )
	last_match = model.start
	last_insert = State( insert_distribution, name="I0" )
	last_delete, last_sb = None, None

	model.add_transition( model.start, last_insert, 0.02 )
	model.add_transition( last_insert, last_insert, 0.70 )

	for i, distribution in enumerate( distributions ):
		match = State( distribution, name="M"+str(i+1), color=['r', 'y', 'g', 'm', 'b', '0.75', '0.50', '0.25', 'r', 'y', 'g' ][i] ) 
		insert = State( insert_distribution, name="I"+str(i+1) )
		delete = State( None, name="D"+str(i+1) )
		short_backslip = State( None, name="S"+str(i+1) ) if i >= 1 else None

		model.add_transition( last_match, match, 0.60 if last_sb is None else 0.55 if not last_match.is_silent() else 0.90 )
		model.add_transition( match, match, 0.30 )
		model.add_transition( last_match, delete, 0.08 if last_sb is None else 0.03 )
		model.add_transition( match, insert, 0.02 )

		model.add_transition( insert, insert, 0.70 )
		model.add_transition( last_insert, delete, 0.10 )
		model.add_transition( last_insert, match, 0.20 )

		model.add_transition( delete, insert, 0.10 )

		if short_backslip is not None:
			model.add_transition( match, short_backslip, 0.10 )
			model.add_transition( short_backslip, last_match, 0.95 if last_sb is not None else 1.00 )

			if last_sb is not None:
				model.add_transition( short_backslip, last_sb, 0.05 )

		if last_delete is not None:
			model.add_transition( last_delete, match, 0.70 )
			model.add_transition( last_delete, delete, 0.20 )

		last_match, last_insert, last_delete = match, insert, delete
		last_sb = short_backslip

	model.add_transition( last_delete, model.end, 0.95 )
	model.add_transition( last_insert, model.end, 0.70 )
	model.add_transition( last_match, model.end, 0.50 )

	model.bake()
	return model

def Hel308ProfileHMM( distributions, name="Hel308 Profile HMM",low=0, high=90, 
	sb_length=1, lb_length=9 ):
	"""
	Generates a profile HMM for Hel308 specific data. Includes:
		* Short backslips
		* Long backslips
		* Oversegmentation handling via mixture model
		* Repeat backslip handling
	"""

	def match_model( distribution, name ):
		model = Model( name=name )

		match = State( distribution, name=name ) # Match without oversegmentation
		match_os = State( distribution, name=name ) # Match with oversegmentation

		model.add_transition( model.start, match, 0.75 )
		model.add_transition( model.start, match_os, 0.25 )

		model.add_transition( match, match, 0.10 )
		model.add_transition( match, model.end, 0.90 )

		model.add_transition( match_os, match_os, 0.80 )
		model.add_transition( match_os, model.end, 0.20 )
		return model

	model = Model("{}-{}".format( name, len(distributions) ) )

	insert_distribution = UniformDistribution( low, high )

	long_backslip_to = last_match = model.start
	last_insert = State( insert_distribution, name="I0" )
	last_delete, last_sb, last_lb = None, None, None

	matches = []
	model.add_transition( model.start, last_insert, 0.02 )
	model.add_transition( last_insert, last_insert, 0.70 )

	for i, distribution in enumerate( distributions ):
		match = match_model( distribution, name="M"+str(i+1) ) # Match state is now a short model
		insert = State( insert_distribution, name="I"+str(i+1) ) # Uniform distribution across the space
		delete = State( None, name="D"+str(i+1) ) # Silent state
		short_backslip = State( None, name="S"+str(i+1) ) if i >= sb_length else None # Silent state for moving backwards
		long_backslip = State( None, name="L"+str(i+1) ) if i >= lb_length else None # Silent state for moving backwards a lot

		model.add_model( match ) # Add that model to the main model 


		model.add_transition( last_match.end if i > 0 else last_match, match.start,
			0.90 if i == 0 else 0.80 if i <= sb_length else 0.77 if i <= lb_length else 0.75 )

		model.add_transition( last_match if i == 0 else last_match.end, delete, 0.08 if i == 0 else 0.03 )
		model.add_transition( match.end, insert, 0.02 )

		model.add_transition( insert, insert, 0.50 )
		model.add_transition( insert, match.start, 0.20 )
		model.add_transition( last_insert, delete, 0.10 )
		model.add_transition( last_insert, match.start, 0.20 )

		model.add_transition( delete, insert, 0.10 )

		if short_backslip is not None:
			model.add_transition( match.end, short_backslip, 0.03 )
			model.add_transition( short_backslip, last_match.start, 0.80 if last_sb is not None else 0.95 )

			repeat_match = State( distribution=distribution, name=match.name )
			repeat_last_match = State( distribution=distributions[i-1], name=last_match.name )

			model.add_transition( short_backslip, repeat_last_match, .05 )
			model.add_transition( repeat_last_match, repeat_match, 0.70 )
			model.add_transition( repeat_last_match, match.start, 0.10 )
			model.add_transition( repeat_last_match, repeat_last_match, 0.20 )

			model.add_transition( repeat_match, repeat_match, 0.20 )
			model.add_transition( repeat_match, repeat_last_match, 0.80 )

			if last_sb is not None:
				model.add_transition( short_backslip, last_sb, 0.15 )

		if long_backslip is not None:
			model.add_transition( match.end, long_backslip, 0.02 )
			model.add_transition( long_backslip, matches[-lb_length].start, 0.75 if last_lb is not None else 1.00 )

			if last_lb is not None:
				model.add_transition( long_backslip, last_lb, 0.25 )

		if last_delete is not None:
			model.add_transition( last_delete, match.start, 0.70 )
			model.add_transition( last_delete, delete, 0.20 )

		last_match, last_insert, last_delete = match, insert, delete
		last_sb, last_lb = short_backslip, long_backslip
		matches.append( match ) # Append after, so index -i means move i states back

	model.add_transition( last_delete, model.end, 0.95 )
	model.add_transition( last_insert, model.end, 0.70 )
	model.add_transition( last_match.end, model.end, 0.50 )
	model.bake()
	return model

def tRNAbasic():
	mean_stds = [ ( 33, 1.75 ), ( 29.1, 1.15 ), ( 24.01, 0.45 ), ( 26.04, 0.43 ), 
				  ( 24.4, 0.43 ), ( 29.17, 0.46 ), ( 26.5, 0.46 ), 
				  ( 25.7, 0.43 ), ( 22.77, 0.48 ), ( 30.06, 0.46 ), ( 24.9, 1.19 ) ]

	distributions = [ NormalDistribution( m, s ) for m, s in mean_stds ]
	return Phi29ProfileHMM( distributions )

def tRNAbasic2():
	mean_stds = [ ( 33, 1.75 ), ( 29.1, 1.15 ), ( 24.01, 0.45 ), ( 26.04, 0.43 ), ( 24.4, 0.43 ), ( 29.17, 0.46 ), ( 26.5, 0.46 ), ( 25.7, 0.43 ), ( 22.77, 0.48 ), ( 30.06, 0.46 ), ( 24.9, 1.19 ) ]

	distributions = [ NormalDistribution( m, s ) for m, s in mean_stds ]
	return Hel308ProfileHMM( distributions )

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

	s1 = State( NormalDistribution( 33, 1.75 ), name="1", color='r' )
	s2 = State( NormalDistribution( 29.1, 1.15 ), name="2", color='y' )
	s3 = State( NormalDistribution( 24.01, 0.155), name="3", color='g' )
	s4 = State( NormalDistribution( 26.04, 0.43 ), name="4", color='m' )
	s5 = State( NormalDistribution( 24.4, 0.427 ), name="5", color='b' )
	s6 = State( NormalDistribution( 29.17, 0.464 ), name="6", color='0.25' )
	s7 = State( NormalDistribution( 27.1, 0.66 ), name="7", color='0.75' )
	s8 = State( NormalDistribution( 25.7, 0.43 ), name="8", color='k' )
	s9 = State( NormalDistribution( 22.77, 0.481 ), name="9", color='c' )
	s10 = State( NormalDistribution( 30.06, 0.46 ), name="10", color='0.10' )
	s11 = State( NormalDistribution( 24.9, 1.19 ), name="11", color='r' )

	states = [ s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11 ]
	for state in states:
		model.add_state( state )

	model.add_transition( model.start, s1, 0.10 )
	model.add_transition( model.start, s2, 0.70 )
	model.add_transition( model.start, s3, 0.10 )
	model.add_transition( model.start, s4, 0.10 )
	model.add_transition( s1, s1, 0.10 )
	model.add_transition( s1, s2, 0.30 )
	model.add_transition( s1, s3, 0.60 )
	model.add_transition( s2, s2, 0.30 )
	model.add_transition( s2, s3, 0.70 )
	model.add_transition( s3, s3, 0.60 )
	model.add_transition( s3, s4, 0.30 )
	model.add_transition( s4, s3, 0.10 )
	model.add_transition( s4, s4, 0.30 )
	model.add_transition( s4, s5, 0.60 )
	model.add_transition( s4, s6, 0.05 )
	model.add_transition( s5, s5, 0.49 )
	model.add_transition( s5, s6, 0.50 )
	model.add_transition( s6, s6, 0.25 )
	model.add_transition( s6, s7, 0.70 )
	model.add_transition( s6, s8, 0.05 )
	model.add_transition( s7, s7, 0.10 )
	model.add_transition( s7, s8, 0.90 )
	model.add_transition( s8, s8, 0.70 )
	model.add_transition( s8, s9, 0.20 )
	model.add_transition( s8, model.end, 0.10 )
	model.add_transition( s9, s9, 0.70 )
	model.add_transition( s9, s10, 0.20 )
	model.add_transition( s9, model.end, 0.10 )
	model.add_transition( s10, s10, 0.10 )
	model.add_transition( s10, s11, 0.80 )
	model.add_transition( s10, model.end, 0.10 )
	model.add_transition( s11, s11, 0.25 )
	model.add_transition( s11, model.end, 0.75 )

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
