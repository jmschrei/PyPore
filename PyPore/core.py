# core.py
# Contact: Jacob Schreiber
#			jmschrei@soe.ucsc.edu
#
# This holds the core data types which may be abstracted in many
# different applications. 

import numpy as np
import itertools as it
import re
import json
from contextlib import contextmanager

class MetaSegment( object ):
	'''
	The metadata on an abstract segment of ionic current. All information about a segment can be 
	loaded, without the expectation of the array of floats.
	'''
	def __init__( self, **kwargs ):
		for key, value in kwargs.iteritems():
			with ignored( AttributeError ):
				setattr( self, key, value )

		# If current is passed in, get metadata directly from it, then remove
		# the reference to that array.
		if hasattr( self, "current" ):
			self.n = len( self.current )
			self.mean = np.mean( self.current )
			self.std = np.std( self.current )
			self.min = np.min( self.current )
			self.max = np.max( self.current )
			del self.current

		# Fill in start, end, and duration given that you only have two of them.
		if hasattr( self, "start" ) and hasattr( self, "end" ) and not hasattr(self, "duration" ):
			self.duration = self.end - self.start
		elif hasattr( self, "start" ) and hasattr( self, "duration" ) and not hasattr(self, "end"):
			self.end = self.start + self.duration
		elif hasattr( self, "end" ) and hasattr( self, "duration" ) and not hasattr(self, "start"):
			self.start = self.end - self.duration

	def __repr__( self ):
		'''
		The representation is a JSON.
		'''

		return self.to_json()

	def __len__( self ):
		'''
		The length of the metasegment is the length of the ionic current it
		is representing.
		'''

		return self.n

	def delete( self ):
		'''
		Delete itself. There are no arrays with which to delete references for.
		'''

		del self

	def to_meta( self ):
		'''
		Kept to allow for error handling, but since it's already a metasegment
		it won't actually do anything.
		'''

		pass

	def to_dict( self ):
		'''
		Return a dict representation of the metadata, usually used prior to
		converting the dict to a JSON.
		'''

		keys = ['mean', 'std', 'min', 'max', 'start', 'end', 'duration']
		d = { i: getattr( self, i ) for i in keys if hasattr( self, i ) }
		d['name'] = self.__class__.__name__
		return d

	def to_json( self, filename=None ):
		'''
		Return a JSON representation of this, by reporting the important
		metadata.
		'''

		_json = json.dumps( self.to_dict(), indent=4, separators=(',', ' : '))
		if filename:
			with open( filename, 'w' ) as outfile:
				outfile.write( _json )
		return _json

	@classmethod
	def from_json( self, filename=None, json=None ):
		'''
		Read in a metasegment from a JSON and return a metasegment object. 
		Either pass in a file which has a segment stored, or an actual JSON 
		object.
		'''

		assert filename or json and not (filename and json)
		import re

		if filename:
			with open( filename, 'r' ) as infile:
				json = ''.join([ line for line in infile ])

		words = re.findall( r"\[[\w'.-]+\]|[\w'.-]+", json )
		attrs = { words[i]: words[i+1] for i in xrange(0, len(words), 2) }

		return MetaSegment( **attrs )

class Segment( object ):
	'''
	A segment of ionic current, and methods relevant for collecting metadata. The ionic current is
	expected to be passed as a numpy array of floats. Metadata methods (mean, std..) are decorated 
	as properties to reduce overall computational time, making them calculated on the fly rather 
	than during analysis.
	'''
	def __init__( self, current, **kwargs ):
		'''
		The segment must have a list of ionic current, of which it stores some statistics about. 
		It may also take in as many keyword arguments as needed, such as start time or duration 
		if already known. Cannot override statistical measurements. 
		'''
		self.current = current

		for key, value in kwargs.iteritems():
			if hasattr( self, key ):
				continue
			with ignored( AttributeError ):
				setattr( self, key, value )

	def __repr__( self ):
		'''
		The string representation of this object is the JSON.
		'''

		return self.to_json()

	def __len__( self ):
		'''
		The length of a segment is the length of the underlying ionic current
		array.
		'''

		return self.n


	def to_dict( self ):
		'''
		Return a dict representation of the metadata, usually used prior to
		converting the dict to a JSON.
		'''

		keys = ['mean', 'std', 'min', 'max', 'start', 'end', 'duration']
		d = { i: getattr( self, i ) for i in keys if hasattr( self, i ) }
		d['name'] = self.__class__.__name__
		return d

	def to_json( self, filename=None ):
		'''
		Return a JSON representation of this, by reporting the important
		metadata.
		'''

		_json = json.dumps( self.to_dict(), indent=4, separators=(',', ' : '))
		if filename:
			with open( filename, 'w' ) as outfile:
				outfile.write( _json )
		return _json

	def to_meta( self ):
		'''
		Convert from a segment to a 'metasegment', which stores only metadata
		about the segment and not the full array of ionic current.
		'''

		for key in ['mean', 'std', 'min', 'max', 'end', 'start', 'duration']:
			with ignored( KeyError, AttributeError ):
				self.__dict__[ key ] = getattr( self, key )
		del self.current 

		self.__class__  = type( "MetaSegment", ( MetaSegment, ), self.__dict__ )

	def delete( self ):
		'''
		Deleting this segment requires deleting its reference to the ionic
		current array, and then deleting itself. 
		'''

		with ignored( AttributeError ):
			del self.current

		del self

	def scale( self, sampling_freq ):
		'''
		Rescale all of the values to go from samples to seconds.
		'''

		with ignored( AttributeError ):
			self.start /= sampling_freq
			self.end /= sampling_freq
			self.duration /= sampling_freq

	@property
	def mean( self ):
		return np.mean( self.current )
	@property
	def std( self ):
		return np.std( self.current )
	@property
	def min( self ):
		return np.min( self.current )
	@property
	def max( self ):
		return np.max( self.current )
	@property
	def n( self ):
		return len( self.current )


	@classmethod
	def from_json( self, filename=None, json=None ):
		'''
		Read in a segment from a JSON and return a metasegment object. Either
		pass in a file which has a segment stored, or an actual JSON object.
		'''

		assert filename or json and not (filename and json)
		import re

		if filename:
			with open( filename, 'r' ) as infile:
				json = ''.join([ line for line in infile ])
		
		if 'current' not in json:
			return MetaSegment.from_json( json=json )

		words = re.findall( r"\[[\w\s'.-]+\]|[\w'.-]+", json )
		attrs = { words[i]: words[i+1] for i in xrange(0, len(words), 2) }

		current = np.array([ float(x) for x in attrs['current'][1:-1].split() ])
		del attrs['current']
		
		return Segment( current, **attrs )

@contextmanager
def ignored( *exceptions ):
	'''
	Replace the "try, except: pass" paradigm by replacing those three lines with a single line.
	Taken from the latest 3.4 python update push by Raymond Hettinger, see:
	http://hg.python.org/cpython/rev/406b47c64480
	'''
	try:
		yield
	except exceptions:
		pass
