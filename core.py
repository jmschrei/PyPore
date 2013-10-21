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

class MetaSegment( object ):
	'''
	The metadata on an abstract segment of ionic current. All information about a segment can be loaded,
	without the expectation of the array of floats.
	'''
	def __init__( self, **kwargs ):
		for key, value in kwargs.iteritems():
			setattr( self, key, value )

		if hasattr( self, "current" ):
			print "herro"
			self.n = self.current.shape[0]
			self.mean = np.mean( self.current )
			self.std = np.std( self.current )
			self.min = np.min( self.current )
			self.max = np.max( self.current )
			del self.current

		if hasattr( self, "start" ) and hasattr( self, "end" ) and not hasattr( self, "duration" ):
			self.duration = self.end - self.start
		elif hasattr( self, "start" ) and hasattr( self, "duration" ) and not hasattr( self, "end" ):
			self.end = self.start + self.duration
		elif hasattr( self, "end" ) and hasattr( self, "duration" ) and not hasattr( self, "start" ):
			self.start = self.end - self.duration

	def __repr__( self ):
		return self.to_json()

	def __len__( self ):
		return self.n

	def delete( self ):
		del self

	def to_meta( self ):
		pass

	def to_json( self, filename=None ):
		keys = ['mean', 'std', 'min', 'max', 'start', 'end', 'duration']
		d = { i: getattr( self, i ) for i in self.__dict__.keys() + keys if hasattr( self, i ) }
		if filename:
			with open( filename, 'w' ) as outfile:
				outfile.write( dict_to_json( d ) )
		return dict_to_json( d )


	@classmethod
	def from_json( self, filename=None, json=None ):
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
		self.n = current.shape[0] 
		for key, value in kwargs.iteritems():
			if hasattr( self, key ):
				continue
			setattr( self, key, value )

		if hasattr( self, 'second' ): 
			if not hasattr( self, 'duration' ):
				self.duration = self.n / self.second
			if hasattr( self, 'start' ):
				self.start = self.start / self.second

	def __repr__( self ):
		return self.to_json()
	def __len__( self ):
		return self.n
	def __add__( self, otherSegment ):
		if isinstance( otherSegment, Segment ):
			return Segment( current=np.concatenate(( self.current, otherSegment.current )))
		raise TypeError( "Cannot add type {type} to type Segment".format( type(otherSegment) ) )

	def to_dict( self ):
		keys = ['mean', 'std', 'min', 'max', 'start', 'end', 'duration']
		d = { i: getattr( self, i ) for i in keys if hasattr( self, i ) }
		d['name'] = self.__class__.__name__
		return d

	def to_json( self, filename=None ):
		_json = json.dumps( self.to_dict(), indent=4, separators=(',', ' : '))
		if filename:
			with open( filename, 'w' ) as outfile:
				outfile.write( _json )
		return _json

	def to_meta( self ):
		for key in ['mean', 'std', 'min', 'max', 'end', 'start', 'duration']:
			try:
				self.__dict__[ key ] = getattr( self, key )
			except:
				pass

		del self.current 

		self.__class__  = type( "MetaSegment", ( MetaSegment, ), self.__dict__ )

	def delete( self ):
		try:
			del self.current
		except:
			pass

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
	def end( self ):
		return self.start + self.duration

	@classmethod
	def from_json( self, filename=None, json=None ):
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

class Container( object ):
	'''
	An abstract container for other objects. Built with the idea that it will store many types of
	other objects, which are all stratified together. Add and get method built with the intention
	of not needing to know what these object types are. 
	'''
	def __init__( self, **kwargs ):
		for key, value in kwargs.iteritems():
			if not hasattr( self, key ):
				setattr( self, key, value )
			else:
				raise Exception()

	def add( self, data ):
		'''
		Add an object type to an attribute which is the pluralized version of that type.
		For example, adding an object of type "Event" will create, or append to, an
		attribute named "events", accessible either by obj.events or by using the get
		method. 
		'''
   		attr = "{}s".format( data.__class__.__name__.lower() )
		if hasattr( self, attr ):
			if type( getattr( self, attr ) ) not in ( np.array, list ):
				setattr( self, attr, [ data ] )
			else:
				attr_data = getattr( self, attr )
				setattr( self, attr, [ x for x in it.chain( attr_data, data ) ] )
		else:
			setattr( self, attr, np.array([ data ]) )

	def get( self, attr, filter_attr=None, filter=None, indices=None ):
		'''
		Gets an object type, with a possible filter. This assumes that the attributes
		are connected in a tree-like manner, such as filter_attr has an attribute named
		attr, like filter_attr.attr leads to attr.For example, if both people and cities 
		are attributes of this container, and cities also have a people attribute, such as
		<city instance>.people, then you can use filter_attr and filter (or indices) to
		indicate which cities you want to return the people for. 
    	
		Example usage:
		import numpy as np
		pop = 5
		people = [ Person() for i in range( 100 )]
		cities = [ City( people = people[i*pop:i*pop+pop] for i in range(int(100/pop)),
						 happiness_index = np.random.rand(1) ) ] 
		population_data = Container( people = people, cities = cities )
		print population_data.get( "people", filter_attr="cities",
									filter=lambda city: city.happiness_index > 1 )

		'''
		try:
			if filter_attr:
				return np.concatenate([ getattr( datum, attr ) for datum in \
										self.get( filter_attr, filter=filter, indices=indices ) ])
			elif filter:
				return [ datum for datum in getattr( self, attr ) if filter( datum ) ]
			elif indices:
				return [ datum for i, datum in enumerate( getattr( self, attr ) ) if i in indices ]
			else:
				return getattr( self, attr )
		except:
			raise AttributeError( "Attribute {} does not exist.".format( attr ) )

def show( d, i=0 ):
	if type(d) == dict:
		for key, val in d.items():
			print "\t"*i, type(key), type(val)
			show(val, i=i+1)
	elif type(d) == list:
		for val in d:
			print "\t"*i, type(val)
			show(val, i=i+1)