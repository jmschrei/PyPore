'''
'''

import numpy as np
import math

class Queue( object ):
	'''
	Implements a FIFO queue. May act as a priority queue when
	a positional argument is given for insert_at. Cannot act
	like a FIFO queue.
	'''
	def __init__( self, *args ):
		self.data = []
		self.size = 0
		self.insert( *args )
	def __getitem__( self, slice ):
		'''
		When someone tries to get a slice of the queue, what
		they really mean is they want a slice of the list
		underlying the queue.
		'''
		return self.data[slice]
	def __repr__( self ):
		'''
		If someone wants a representation of the queue, print
		all of the objects in order. Since we do not know what
		they are storing, there is no attribute we can call.
		'''
		return repr(self.data)
	def pop( self ):
		'''
		Get the first element of the queue, remove it from
		the queue and return it.. 
		'''
		if not self.empty():
			top = self.data[0]
			self.data = self.data[1:]
			self.size -= 1
			return top
		return None
	def insert( self, *args ):
		'''
		Insert the items in the order that they are received
		to the back of the queue. 
		'''
		for item in args:
			self.data.append( item )
			self.size += 1

	def insert_at( self, node, pos ):
		'''
		Inserts a piece of data at a specific position in the
		queue. Makes the queue a priority queue.
		'''
		self.data = self.data[:pos] + [node] + self.data[pos:]
		self.size += 1
	def empty( self ):
		return self.size == 0

class Stack( object ):
	'''
	Acts in a LIFO fashion. Implements push and pop as
	method names. 
	'''
	def __init__( self, *args ):
		self.data = []
		self.size = 0
		self.insert( *args )
	def __getitem__( self, slice ):
		'''
		When someone wants a slice of the stack, they want a slice
		of the array that underlies the stack.
		'''
		return self.data[slice]
	def __repr__( self ):
		'''
		When someone wants a representation of the stack, they want
		the representation of the underlying array. 
		'''
		return repr( self.data )

	def push( self, *args ):
		'''Add an item to the top of the list.'''
		for item in args:
			self.data.append( item )
			self.size += 1

	def pop( self ):
		'''Remove an item from the top of the list.'''
		if not self.empty():
			top = self.data[-1]
			self.data = self.data[:-1]
			self.size -= 1
			return top
		return None

class Node( object ):
	'''
	An abstract node object, which may be customized to explicitly
	contain the data of interest through **kwargs, or just passed
	an object to be stored if desired. Passing in a series of other
	nodes will indicate connect this node in a unidirectional manner
	to the nodes in the series. Can also explicitly use the connect
	and deconnect methods to indicate transitions.
	'''
	def __init__( self, *args, **kwargs ):
		self.nodes = []
		self.order = 0

		for item in args:
			if isinstance( item, Node ):
				self.nodes.append( item )
				self.order += 1

		for key, value in kwargs.items():
			if not hasattr( self, key ):
				setattr( self, key, value )

	def order( self ):
		'''
		The order of the node is considered to be the number of other
		nodes which it is attached to, i.e. the number of edges that
		it has coming out of it. 
		'''
		return len( self.nodes )
	
	def connect( self, node ):
		''' Explicitly connect this node to another node. '''
		self.nodes.append( node )
		self.order += 1

	def disconnect( self, index=None, node=None ):
		'''
		Explicitly disconnect this node from another node. Can either
		pass in the index of the node stored in attribute nodes, or
		pass the node object in and it will find it.
		'''
		if idx:
			del self.nodes[idx]
			self.order -= 1
		elif node:
			for i, n in enumerate( self.nodes ):
				if node is n:
					del self.nodes[i]
					self.order -= 1

class RedBlackTree( object ):
	def __init__( self, root=None ):
		self.root = root
		self.cursor = None
		self.n = 0

	def __repr__( self ):
		if self.root == None:
			return ''
		return repr( self.root.data ) + ' l' + repr( RedBlackTree(self.root.left) ) + ' r' + repr( RedBlackTree(self.root.right) )

	def insert( self, node ):
		if not isinstance( node, Node ):
			node = Node( data=node, left=None, right=None, parent=None  )

		self.cursor = self.root
		while True:
			if self.cursor == None:
				self.root = node
				node.color = 'r'
				break
			elif self.cursor.data <= node.data:
				if self.cursor.right == None:
					node.parent = self.cursor
					self.cursor.right = node
					node.color = 'r'
					break
				else:
					self.cursor = self.cursor.right 
			else:
				if self.cursor.left == None:
					node.parent = self.cursor
					self.cursor.left = node
					node.color = 'r'
					break
				else:
					self.cursor = self.cursor.left
		
		while True:
			if self.root == node:
				node.color = 'b'
				break
			elif node.parent.color == 'b':
				node.color = 'b'
				break
			elif node.parent.parent.left == 'r' and node.parent.parent.right == 'r':
				node.parent.parent.left.color = 'b'
				node.parent.parent.right.color = 'b'
				node.parent.parent = 'r'
				node.color = 'r'
			elif node == node.parent.right and node.parent == node.parent.parent.left:
				self._rotate_left( node.parent )
				node = node.left
			elif node == node.parent.left and node.parent == node.parent.parent.right:
				self._rotate_right( node.parent )
				node = node.right
			else:
				node.parent.color = 'b'
				node.parent.parent.color = 'r'
				if node == node.parent.left:
					self._rotate_right( self.parent.parent )
					break
				else:
					self._rotate_left( self.parent.parent )
					break

	def _rotate_left( self, node ):
		g, p, l = node.parent.parent, g.left, node.left
		g.left = node
		node.left = p
		p.right = l

	def _rotate_right( self, node ):
		g, p, r = node.parent.parent, g.right, node.right
		g.right = node
		node.right = p
		p.left = r

class Tree( object ):
	def __init__( self, root ):
		self.root = root
		self.nodes = self.root.nodes
	
	@property
	def n( self ):
		'''
		Does a recursive dfs to return the full size of the tree starting at
		this specific node. 
		'''
		return 1 + sum([ Tree(node).n for node in self.root.nodes ])
	def dfs( self, key, value=None, scan=False, verbose=False, func=None, filter=None ):
		'''
		Depth-first search or depth-first scan. Search will look for an item
		to see if it is present in the tree by a key-value attribute pair.
		Scan will go through the entire tree, either performing a specific
		function if func is present, or printing out all of the members in
		a post-order traversal. 
		'''
		# Code for a scan through the tree
		if scan:
			dfs_scan = []
			for node in self.root.nodes:
				subtree = Tree(node)
				if filter:
					if filter(node):
						dfs_scan.extend( subtree.dfs( key, value, scan, verbose, func, filter ) )
				else:
					dfs_scan.extend( subtree.dfs( key, value, scan, verbose, func, filter ) )
			if func:
				func(self.root)
			if verbose:
				try:
					print "{0}:{1}".format( key, getattr( self.root, key ) )
				except:
					print "{0}:{1}".format( key, "Null" )
			return dfs_scan + [ getattr( self.root, key ) ]
		
		# Code for a search through the tree 
		else:
			if getattr( self.root, key ) == value:
				if verbose:
					print "{} found".format(key)
				return True
			else:
				for node in self.root.nodes:
					if Tree(node).dfs( key, value, scan, verbose, func ):
					 	return True
			return False

	def bfs( self, key, value=None, scan=False, verbose=False, func=None ):
		'''
		Breadth-first search through the the tree. Implements a queue in order
		to go through in a breadth-first manner.
		'''
		q = Queue( self.root )
		while not q.empty():
			node = q.pop()
			if not scan:
				if getattr(node, key) == value:
					if verbose:
						print "{} found".format( value )
					return True
			else:
				if verbose:
					try:
						print "{0}:{1}".format( key, getattr( node, key ) )
					except:
						print "{0}:{1}".format( key, "Null" )
				if func:
					func(node)
			for adj_node in node.nodes:
				q.insert( adj_node )

def distance_matrix_to_dendrogram( matrix, color_threshold ):
	from scipy.cluster.hierarchy import linkage, dendrogram
	import matplotlib.pyplot as plt

	linkage = linkage( matrix, method='weighted' )
	plt.title( "Event Tree using WPGMA Linkage Matrix")
	plt.ylabel( "Distance" )
	plt.xlabel( "Event ID" )
	dend = dendrogram( linkage, color_threshold=color_threshold, leaf_font_size=10 )
	for i, d in zip( dend['icoord'], dend['dcoord'] ):
		x = 0.5 * sum(i[1:3])
		y = d[1]
		plt.plot(x,y, 'co', alpha=0.5 )
		plt.annotate( "%.3g" % y, (x,y), xytext=(0,-8), textcoords='offset points',
						va='top', ha='center' )
	plt.show()


def distance_matrix_to_tree( matrix ):
	m, n = matrix.shape
	nodes = {}
 	for i in xrange(m):
		nodes[str(i)] = Node( id = str(i), dist = {} )   
		for j in xrange( i+1, n):
			nodes[str(i)].dist[str(j)] = matrix[i, j]
		for j in xrange( 0, i ):
			nodes[str(i)].dist[str(j)] = matrix[j, i]

	while len( nodes ) > 1:
		minimum = np.inf
		for nodex in nodes.keys():
			for nodey in nodes.keys():
				try:
					if nodes[ nodex ].dist[ nodey ] < minimum:
						node_x = nodes[ nodex ]
						node_y = nodes[ nodey ]
						minimum = nodes[ nodex ].dist[ nodey ]
				except:
					pass
		distances = { node.id: ( node.dist[node_x.id] + node.dist[node_y.id] ) / 2  for node in nodes.values() if node.id != node_x.id and node.id != node_y.id }
		distances[ node_x.id ] = distances[ node_y.id ] = node_x.dist[ node_y.id ] / 2
		new_node = Node( node_x, node_y, id="("+node_x.id+','+node_y.id+')', dist=distances )
		del nodes[ node_x.id ], nodes[ node_y.id ]

		for node in nodes.values():
			del node.dist[ node_x.id ], node.dist[ node_y.id ]
			node.dist[ new_node.id ] = new_node.dist[ node.id ]
		nodes[ new_node.id ] = new_node
	return Tree( nodes.values()[0] )



