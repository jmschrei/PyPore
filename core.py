# core.py
# Contact: Jacob Schreiber
#			jmschrei@soe.ucsc.edu
#
# This holds the core data types which may be abstracted in many
# different applications. 

import numpy as np
import itertools as it

class MetaSegment( object ):
	'''
	The metadata on an abstract segment of ionic current. All information about a segment can be loaded,
	without the expectation of the array of floats.
	'''
	def __init__( self, **kwargs ):
		for key, value in kwargs.iteritems():
			setattr( self, key, value )

class Segment( object ):
    '''
    An abstract segment of ionic current. Other information may be loaded in upon initialization. Magic
    methods assume reference to the ionic current stored in it. 
    '''
    def __init__( self, current, **kwargs ):
        '''
        The segment must have a list of ionic current, of which it stores some statistics about. It may
        also take in as many keyword arguments as needed, such as start time or duration if already
        known. Cannot override statistical measurements. 
        '''
        self.current = current
        self.n = current.shape[0] 

        for key, value in kwargs.iteritems():
        	setattr( self, key, value )

        if hasattr( self, 'second' ): 
            if not hasattr( self, 'duration' ):
                self.duration = self.n / self.second
            if hasattr( self, 'start' ):
                self.start = self.start / self.second

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

    def __getitem__( self, index ):
        '''Slicing this object is assumed to be slicing the current stored in it'''
        return self.current[ index ]
    def __repr__( self ):
        return "{mean}".format( mean = round(self.mean, 2) )
    def __len__( self ):
        return self.current.shape[0]
    def __add__( self, otherSegment ):
    	if isinstance( otherSegment, Segment ):
    		return Segment( current=np.concatenate( ( self.current, otherSegment.current ) ) )
    	return self

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