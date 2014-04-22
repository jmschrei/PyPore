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

def Phi29ProfileHMMOld( distributions, name="Phi29 Profile HMM", low=0, high=50 ):
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

def Phi29ProfileHMM( distributions, name="Phi29 Profile HMM",low=0, high=90, 
	sb_length=1 ):
	"""
	Generates a profile HMM for Phi29 specific data. Includes:
		* Short backslips
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

	last_match = model.start
	last_insert = State( insert_distribution, name="I0" )
	last_delete, last_sb, last_lb = None, None, None

	model.add_transition( model.start, last_insert, 0.02 )
	model.add_transition( last_insert, last_insert, 0.70 )

	for i, distribution in enumerate( distributions ):
		match = match_model( distribution, name="M"+str(i+1) ) # Match state is now a short model
		insert = State( insert_distribution, name="I"+str(i+1) ) # Uniform distribution across the space
		delete = State( None, name="D"+str(i+1) ) # Silent state
		short_backslip = State( None, name="S"+str(i+1) ) if i >= sb_length else None # Silent state for moving backwards

		model.add_model( match ) # Add that model to the main model 


		model.add_transition( last_match.end if i > 0 else last_match, match.start,
			0.90 if i == 0 else 0.80 if i <= sb_length else 0.77 )

		model.add_transition( last_match if i == 0 else last_match.end, delete, 0.08 if i == 0 else 0.03 )
		model.add_transition( match.end, insert, 0.02 )

		model.add_transition( insert, insert, 0.75 )
		model.add_transition( insert, match.start, 0.10 )
		model.add_transition( last_insert, delete, 0.05 )
		model.add_transition( last_insert, match.start, 0.10 )

		model.add_transition( delete, insert, 0.10 )

		if short_backslip is not None:
			model.add_transition( match.end, short_backslip, 0.03 )
			model.add_transition( short_backslip, last_match.start, 0.75 if last_sb is not None else 0.90 )

			repeat_match = State( distribution=distribution, name=match.name )
			repeat_last_match = State( distribution=distributions[i-1], name=last_match.name )

			model.add_transition( short_backslip, repeat_last_match, .10 )
			model.add_transition( repeat_last_match, repeat_match, 0.70 )
			model.add_transition( repeat_last_match, match.start, 0.10 )
			model.add_transition( repeat_last_match, repeat_last_match, 0.20 )

			model.add_transition( repeat_match, repeat_match, 0.20 )
			model.add_transition( repeat_match, repeat_last_match, 0.80 )

			if last_sb is not None:
				model.add_transition( short_backslip, last_sb, 0.15 )

		if last_delete is not None:
			model.add_transition( last_delete, match.start, 0.80 )
			model.add_transition( last_delete, delete, 0.10 )

		last_match, last_insert, last_delete = match, insert, delete
		last_sb = short_backslip

	model.add_transition( last_delete, model.end, 0.95 )
	model.add_transition( last_insert, model.end, 0.70 )
	model.add_transition( last_match.end, model.end, 0.50 )
	model.bake()
	return model

def Phi29ProfileHMMU( distributions, name="Phi29 Profile HMM",low=0, high=90, 
	sb_length=1 ):
	"""
	Generates a profile HMM for Phi29 specific data. Includes:
		* Short backslips
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

	last_match = model.start
	last_insert = State( insert_distribution, name="I0" )
	last_delete, last_sb, last_lb = None, None, None
	last_last_match = None

	model.add_transition( model.start, last_insert, 0.02 )
	model.add_transition( last_insert, last_insert, 0.70 )

	unweighted_avg = lambda idx, x: 1. * sum( d.parameters[idx] for d in x ) / len( x )
	n = len( distributions )

	for i, distribution in enumerate( distributions ):
		match = match_model( distribution, name="M"+str(i+1) ) # Match state is now a short model
		insert = State( insert_distribution, name="I"+str(i+1) ) # Uniform distribution across the space
		delete = State( None, name="D"+str(i+1) ) # Silent state
		short_backslip = State( None, name="S"+str(i+1) ) if i >= sb_length else None # Silent state for moving backwards

		model.add_model( match ) # Add that model to the main model 

		# Calculate the in-between segments in order to handle undersegmentation
		if i > 0 and i < n-1:
			dists = [ state.distribution for state in match.graph.nodes() + \
				last_match.graph.nodes() if not state.is_silent() ][::2]

			if 'KernelDensity' in dists[0].name:
				if 'KernelDensity' in dists[1].name:
					a_points, b_points = dists[0].parameters[0], dists[1].parameters[0]
					blend_points = [ (a+b)/2 for a in a_points for b in b_points ]
					blend_bandwidth = ( dists[0].parameters[1] + dists[0].parameters[1] ) / 2
					blend_match = GaussianKernelDensity( blend_points, blend_bandwidth )
				else:
					mean, std = dists[1].parameters[0], dists[1].parameters[1]
					blend_points = [ (mean+j)/2 for j in dists[0].parameters[0] ]
					blend_bandwidth = ( std+dists[0].parameters[1] ) / 2
					blend_match = GaussianKernelDensity( blend_points, blend_bandwidth )

			else:
				if 'KernelDensity' in dists[1].name:
					mean, std = dists[0].parameters[0], dists[0].parameters[1]
					blend_points = [ (mean+b)/2 for b in dists[1].parameters[0] ]
					blend_bandwidth = ( std+dists[1].parameters[1] ) / 2
					blend_match = GaussianKernelDensity( blend_points, blend_bandwidth )
				else:
					a_mean, a_std = dists[0].parameters[0], dists[0].parameters[1]
					b_mean, b_std = dists[1].parameters[0], dists[1].parameters[1]

					blend_mean = ( a_mean + b_mean ) / 2
					blend_std = ( a_std + b_std ) / 2
					blend_match = NormalDistribution( blend_mean, blend_std )

			blend_state = State( blend_match, name="U{}".format( i ) )

			model.add_transition( last_last_match.end if not isinstance( last_last_match, State ) 
				else last_last_match, blend_state, 0.05 )
			model.add_transition( blend_state, blend_state, 0.05 )
			model.add_transition( blend_state, match.end, 0.95 )

		# Return to building the model
		model.add_transition( last_match.end if i > 0 else last_match, match.start,
			0.90 if i == n-1 else 0.85 if i == 0 else 0.75 if i <= sb_length else 0.72 )

		model.add_transition( last_match if i == 0 else last_match.end, delete, 0.08 if i == 0 else 0.03 )
		model.add_transition( match.end, insert, 0.02 )

		model.add_transition( insert, insert, 0.75 )
		model.add_transition( insert, match.start, 0.10 )
		model.add_transition( last_insert, delete, 0.05 )
		model.add_transition( last_insert, match.start, 0.10 )

		model.add_transition( delete, insert, 0.10 )

		if short_backslip is not None:
			model.add_transition( match.end, short_backslip, 0.03 )
			model.add_transition( short_backslip, last_match.start, 0.75 if last_sb is not None else 0.90 )

			repeat_match = State( distribution=distribution, name=match.name )
			repeat_last_match = State( distribution=distributions[i-1], name=last_match.name )

			model.add_transition( short_backslip, repeat_last_match, .10 )
			model.add_transition( repeat_last_match, repeat_match, 0.70 )
			model.add_transition( repeat_last_match, match.start, 0.10 )
			model.add_transition( repeat_last_match, repeat_last_match, 0.20 )

			model.add_transition( repeat_match, repeat_match, 0.20 )
			model.add_transition( repeat_match, repeat_last_match, 0.80 )

			if last_sb is not None:
				model.add_transition( short_backslip, last_sb, 0.15 )

		if last_delete is not None:
			model.add_transition( last_delete, match.start, 0.80 )
			model.add_transition( last_delete, delete, 0.10 )

		last_last_match, last_match, last_insert, last_delete = last_match, match, insert, delete
		last_sb = short_backslip


	model.add_transition( last_delete, model.end, 0.95 )
	model.add_transition( last_insert, model.end, 0.70 )
	model.add_transition( last_match.end, model.end, 0.50 )
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

	last_match = model.start
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

		model.add_transition( insert, insert, 0.75 )
		model.add_transition( insert, match.start, 0.10 )
		model.add_transition( last_insert, delete, 0.05 )
		model.add_transition( last_insert, match.start, 0.10 )

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

hmm_factory = {}
