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
from yahmm import *

class HMMBoard( Model ):
	"""
	A HMM circuit board. Define the number of lanes in the circuit, with each
	lane having a start and end silent state.
	"""

	def __init__( self, n, name=None ):
		super( HMMBoard, self ).__init__( name="Board {}".format( name ) )

		self.directions = [ '>' ] * n 
		self.n = n
		for i in xrange( 1,n+1 ):
			start = State( None, name="b{}s{}".format(name, i) )
			end = State( None, name="b{}e{}".format(name, i) )

			setattr( self, "s{}".format( i ), start )
			setattr( self, "e{}".format( i ), end )

			self.add_state( start )
			self.add_state( end )

def ModularProfileModel( board_func, distributions, name, insert ):
	"""
	Create the HMM using circuit board methodologies.
	"""

	# Initialize the model, and the list of boards in the model
	model = Model( name )
	boards = []

	# For each distribution in the list, add it to the model
	for i, distribution in enumerate( distributions ):
		# If this is not the first distribution, then pull the last board to connect to
		if i > 0:
			last_board = boards[-1]

		# If the current distribution is a distribution and not a dictionary..
		if isinstance( distribution, Distribution ):
			# Build a board for that distribution and add it to the model
			board = board_func( distribution, name=i, insert=insert )
			model.add_model( board )

			# If this is the first board, there are no boards to connect to
			if i == 0:
				boards.append( board )
				continue

			# If the last board is a single distribution, simply connect to it
			if isinstance( distributions[i-1], Distribution ):
				# Add the current board to the list of boards
				boards.append( board )

				# Iterate across all the ports on the board
				for j, d in it.izip( xrange( 1,board.n+1 ), board.directions ):
					# Get the end port from the last board and the start port from this board
					end = getattr( last_board, 'e{}'.format( j ) )
					start = getattr( board, 's{}'.format( j ) )

					# Depending on the direction of that port, connect it in the appropriate
					# direction.
					if d == '>':
						model.add_transition( end, start, 1.00 )
					elif d == '<':
						model.add_transition( start, end, 1.00 )

			# If the last distribution was actually a dictionary, then we're remerging from a fork.
			elif isinstance( distributions[i-1], dict ):
				# Calculate the number of forks in there
				n = len( distributions[i-1].keys() )

				# Go through each of the previous boards
				for last_board in boards[-n:]:
					for j, d in it.izip( xrange( 1,board.n+1 ), board.directions ):
						# Get the appropriate end and start port
						end = getattr( last_board, 'e{}'.format( j ) )
						start = getattr( board, 's{}'.format( j ) )

						# Give appropriate transitions given the direction
						if d == '>':
							model.add_transition( end, start, 1.00 )
						elif d == '<':
							model.add_transition( start, end, 1.00 / n )

				# Add the board to the growing list
				boards.append( board )

		# If we're currently in a fork..
		elif isinstance( distribution, dict ):
			# Calculate the number of paths in this fork
			n = len( distribution.keys() )

			# For each path in the fork, attach the boards appropriately
			for key, dist in distribution.items():
				board = board_func( dist, "{}:{}".format( key, i+1 ), insert=insert )
				boards.append( board )
				model.add_model( board )

				# If the last position was in a fork as well..
				if isinstance( distributions[i-1], dict ):
					last_board = boards[-n-1]

					# Connect the ports appropriately
					for j, d in it.izip( xrange( 1, board.n+1 ), board.directions ):
						end = getattr( last_board, 'e{}'.format( j ) )
						start = getattr( board, 's{}'.format( j ) )

						if d == '>':
							model.add_transition( end, start, 1.00 )
						elif d == '<':
							model.add_transition( start, end, 1.00 )

				# If the last position was not in a fork, then we need to fork the
				# transitions appropriately
				else:
					# Go through each of the ports and give appropriate transition
					# probabilities. 
					for j, d in it.izip( xrange( 1, board.n+1 ), board.directions ):
						# Get the start and end states
						end = getattr( last_board, 'e{}'.format( j ) )
						start = getattr( board, 's{}'.format( j ) )

						# Give a transition in the appropriate direction.
						if d == '>':
							model.add_transition( end, start, 1.00 / n )
						elif d == '<':
							model.add_transition( start, end, 1.00 )

	board = boards[0]
	initial_insert = State( insert, name="I:0" )
	model.add_state( initial_insert )

	model.add_transition( initial_insert, initial_insert, 0.70 )
	model.add_transition( initial_insert, board.s1, 0.1 )
	model.add_transition( initial_insert, board.s2, 0.2 )

	model.add_transition( model.start, initial_insert, 0.02 )
	model.add_transition( model.start, board.s1, 0.08 )
	model.add_transition( model.start, board.s2, 0.90 )

	board = boards[-1]
	model.add_transition( board.e1, model.end, 1.00 )
	model.add_transition( board.e2, model.end, 1.00 )

	model.bake()
	return model

def NanoporeGlobalAlignmentModule( distribution, name, insert ):
	'''
	Creates a single board from a distribution. This will create a module which
	is based off the traditional global sequence alignment hmm which contains
	inserts, deletes, and matches, but adds a self loop for matches, a loop back
	to a match from an insert, and a backslip state as well.
	'''

	# Create the board object
	board = HMMBoard( 3, name )
	board.directions = [ '>', '>', '<' ]

	# Create the four states in the module
	insert = State( insert, name="I:{}".format( name ) )
	match = State( distribution, name="M:{}".format( name ) )
	delete = State( None, name="D:{}".format( name ) )
	backslip = State( None, name="B:{}".format( name ) )

	# Add transitions between these states.
	board.add_transition( board.s1, delete, 1.0 )
	board.add_transition( board.s2, match, 1.0 )
	board.add_transition( board.e3, backslip, 1.0 )

	# Add transitions from the backslip state 
	board.add_transition( backslip, match, 0.85 )
	board.add_transition( backslip, board.s3, 0.15 )

	# Add transitions from the delete state
	board.add_transition( delete, board.e1, 0.1 )
	board.add_transition( delete, board.e2, 0.8 )
	board.add_transition( delete, insert, 0.1 )

	# Add transitions from the match state
	board.add_transition( match, board.s3, 0.033 )
	board.add_transition( match, match, 0.4 )
	board.add_transition( match, board.e2, 0.5 )
	board.add_transition( match, insert, 0.033 )
	board.add_transition( match, board.e1, 0.34 )

	# Add transitions from the insert state
	board.add_transition( insert, insert, 0.50 )
	board.add_transition( insert, match, 0.20 )
	board.add_transition( insert, board.e1, 0.05 )
	board.add_transition( insert, board.e2, 0.25)

	# Return the board
	return board

def GlobalAlignmentModule( distribution, name, insert ):
    '''
    This is the repeating subunit for a typical global sequence alignment HMM.
    '''
    
    # Create the initial board with n ports on either side, in this case 2.
    board = HMMBoard( 2, name )
    
    # Define the three states in the repeating subunit, the match, the insert, and the delete
    delete = State( None, name="D:{}".format( name ) )
    insert = State( insert, name="I:{}".format( name ) )
    match = State( distribution, name="M:{}".format( name ) )
    
    # Define the transitions
    board.add_transition( board.s1, delete, 1.00 )
    board.add_transition( board.s2, match, 1.00 )
    
    board.add_transition( match, board.e2, 0.9 )
    board.add_transition( match, board.e1, 0.05 )
    board.add_transition( match, insert, 0.05 )
    
    board.add_transition( delete, insert, 0.15 )
    board.add_transition( delete, board.e1, 0.15 )
    board.add_transition( delete, board.e2, 0.7 )
    
    board.add_transition( insert, insert, 0.7 )
    board.add_transition( insert, board.e1, 0.15 )
    board.add_transition( insert, board.e2, 0.15 )
    
    # Return the model, unbaked
    return board

def Phi29GlobalAlignmentModule( distribution, name, insert=UniformDistribution( 0, 90 ) ):
    """
    Create a module which represents a full slice of the PSSM. Take in
    the distribution which should be represented at that position, and
    create a board with 7 ports on either side.
    """

    def match_model( distribution, name ):
        """
        Build a small match model, allowing for oversegmentation where the
        distribution representing number of segments is a mixture of two
        exponentials.
        """

        model = Model( name=str(name) )

        match = State( distribution, name="M:{}".format( name ) ) # Match without oversegmentation
        match_os = State( distribution, name="MO:{}".format( name ) ) # Match with oversegmentation

        model.add_states( [ match, match_os ] )

        model.add_transition( model.start, match, 0.95 )
        model.add_transition( model.start, match_os, 0.05 )

        model.add_transition( match, match, 0.10 )
        model.add_transition( match, model.end, 0.90 )

        model.add_transition( match_os, match_os, 0.80 )
        model.add_transition( match_os, model.end, 0.20 )
        return model
    
    # Create the board object
    board = HMMBoard( n=5, name=str(name) )
    board.directions = ['>', '>', '<', '>', '<']

    # Create the states in the model
    delete = State( None, name="D:{}".format( name ) )
    match = match_model( distribution, name=name )
    insert = State( insert, name="I:{}".format( name ) )
    match_s = State( distribution, name="MS:{}".format( name ) )
    match_e = State( distribution, name="ME:{}".format( name ) )

    # Add the match model
    board.add_model( match )
    
    # Add all the individual states
    board.add_states( [delete, insert, match_s, match_e] )

    # Add all transitions from the ports to the states
    board.add_transition( board.s1, delete, 1.00 )
    board.add_transition( board.s2, match.start, 1.00 )
    board.add_transition( board.e3, match_e, 1.00 )
    board.add_transition( board.s4, match_s, 1.00 )
    board.add_transition( board.e5, match.start, 0.90 )
    board.add_transition( board.e5, match_e, 0.05 )
    board.add_transition( board.e5, board.s5, 0.05 )

    board.add_transition( delete, board.e1, 0.1 )
    board.add_transition( delete, insert, 0.1 )
    board.add_transition( delete, board.e2, 0.8 )

    board.add_transition( insert, match.start, 0.10 )
    board.add_transition( insert, insert, 0.50 )
    board.add_transition( insert, board.e1, 0.05 )
    board.add_transition( insert, board.e2, 0.35 )

    board.add_transition( match.end, insert, 0.01 )
    board.add_transition( match.end, board.e1, 0.01 )
    board.add_transition( match.end, board.e2, .97 )
    board.add_transition( match.end, board.s5, .01 )

    board.add_transition( match_s, board.s3, 0.80 )
    board.add_transition( match_s, match_s, 0.20 )

    board.add_transition( match_e, board.e2, 0.10 )
    board.add_transition( match_e, match_e, 0.10 )
    board.add_transition( match_e, board.e4, 0.80 )
    return board

######################################################
# DEPRECATED MODELS                                  #
# These models use the generative technique in order #
# to build linear models. We are currently using the #
# technique of defining the repeating subunit, and   #
# then build it using a constructor.                 # 
######################################################

def Phi29ProfileHMM( distributions, name="Phi29 Profile HMM",low=0, high=90, 
	sb_length=1, verbose=True, merge='all' ):
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
	model.bake( verbose=verbose, merge=merge )
	return model

def Phi29ProfileHMMU( distributions, name="Phi29 Profile HMM",low=0, high=90, 
	sb_length=1 ):
	"""
	Generates a profile HMM for Phi29 specific data. Includes:
		* Short backslips
		* Oversegmentation handling via more complicated match state
		* Repeat backslip handling
		* Undersegmentation handling via mixture model
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

			blend_state = State( blend_match, name="U-{}".format( i ) )

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
