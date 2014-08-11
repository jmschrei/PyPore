#!/usr/bin/env python
# Contact: Jacob Schreiber
#          jacobtribe@soe.ucsc.com
# parsers.py 
# 
# This program will read in an abf file using read_abf.py and
# pull out the events, saving them as text files.

from __future__ import division, print_function
import sys
from itertools import tee,izip,chain
import re

import PyPore
import time
import numpy as np
try:
    from PyQt4 import QtGui as Qt
    from PyQt4 import QtCore as Qc
except:
    pass
from core import *

import pyximport
pyximport.install( setup_args={'include_dirs':np.get_include()})
from PyPore.cparsers import FastStatSplit

import json

#########################################
# EVENT PARSERS
#########################################

class parser( object ):
    def __init__( self ):
        pass

    def __repr__( self ):
        ''' Returns a representation of the parser in the form of all arguments. '''
        return self.to_json()

    def to_dict( self ):
        d = { key: val for key, val in self.__dict__.items() if key != 'param_dict'
                                                             if type(val) in (int, float)
                                                                    or ('Qt' not in repr(val) )
                                                                    and 'lambda' not in repr(val) }
        d['name'] = self.__class__.__name__
        return d

    def to_json( self, filename=False ):
        _json = json.dumps( self.to_dict(), indent=4, separators=( ',', ' : ' ) )
        if filename:
            with open( filename, 'r' ) as out:
                out.write( _json )
        return _json

    def parse( self, current ):
        ''' Takes in a current segment, and returns a list of segment objects. '''
        return [ Segment( current=current, start=0, duration=current.shape[0]/100000 ) ]

    def GUI( self ):
        '''
        A generic GUI built using PyQt4 based off the arguments presented in upon
        initialization. Requires PyQt4 to use, but the parser class does not require
        PyQt4 to run.
        '''
        grid = Qt.QGridLayout()
        param_dict = {}
        for i, (key, val) in enumerate( self.__dict__.items() ):
            param_dict[key] = Qt.QLineEdit()
            param_dict[key].setText( str(val) )
            grid.addWidget( Qt.QLabel(key), i, 0 )
            grid.addWidget( param_dict[key], i, 1 )
        self.param_dict = param_dict
        return grid

    def set_params( self ):
        '''
        Updates each paramater presented in the GUI to the value input in the lineEdit
        corresponding to that value.
        '''
        try:
            for key, lineEdit in self.param_dict.items():
                val = lineEdit.text()
                if '.' in val:
                    setattr( self, key, float( val ) )
                    continue
                for i, letter in enumerate(val):
                    if not letter.isdigit():
                        setattr( self, key, str( val ) )
                        continue
                    if i == len(val):
                        setattr( self, key, int( val ) )
        except:
            pass

    @classmethod
    def from_json( self, _json ):
        if _json.endswith(".json"):
            with open( _json, 'r' ) as infile:
                _json = ''.join(line for line in infile)

        d = json.loads( _json )
        name = d['name']
        del d['name']

        return getattr( PyPore.parsers, name )( **d )


class MemoryParse( object):
    '''
    A parser based on being fed previous split points, and splitting a raw file based
    those splits. Used predominately when loading previous split points from the 
    database cache, to reconstruct a parsed file from "memory.""
    '''
    def __init__( self, starts, ends ):
        self.starts = starts
        self.ends = ends
    def parse( self, current ):
        return [ Segment( current=np.array(current[int(s):int(e)], copy=True),
                          start=s,
                          duration=(e-s) ) for s, e in zip(self.starts, self.ends)]

class lambda_event_parser( parser ):
    '''
    A simple rule-based parser which defines events as a sequential series of points which are below a 
    certain threshold, then filtered based on other critereon such as total time or minimum current.
    Rules can be passed in at initiation, or set later, but must be a lambda function takes in a PreEvent
    object and performs some boolean operation. 
    '''
    def __init__( self, threshold=90, rules=None ):
        self.threshold = threshold
        self.rules = rules or [ lambda event: event.duration > 100000,
                                lambda event: event.min > -0.5,
                                lambda event: event.max < self.threshold ]
    def _lambda_select( self, events ):
        '''
        From all of the events, filter based on whatever set of rules has been initiated with.
        ''' 
        return [ event for event in events if np.all( [ rule( event ) for rule in self.rules ] ) ]
    
    def parse( self, current ):
        '''
        Perform a large capture of events by creating a boolean mask for when the current is below a threshold,
        then detecting the edges in those masks, and using the edges to partitition the sample. The events are
        then filtered before being returned. 
        '''
        mask = np.where( current < self.threshold, 1, 0 ) # Find where the current is below a threshold, replace with 1's
        mask = np.abs( np.diff( mask ) )                  # Find the edges, marking them with a 1, by derivative
        tics = np.concatenate( ( [0], np.where(mask ==1)[0]+1, [current.shape[0]] ) )
        del mask
        events = [ Segment(current=current, 
                            start=tics[i],
                            end=tics[i+1], 
                            duration=tics[i+1]-tics[i] ) 
                    for i, current in enumerate( np.split( current, tics[1:-1]) ) ]
        return [ event for event in self._lambda_select( events ) ]
    
    def GUI( self ):
        '''
        Override the default GUI for use in the Abada GUI, allowing for customization of the rules and threshol via
        the GUI. 
        '''
        threshDefault, timeDefault = "90", "1"
        maxCurrentDefault, minCurrentDefault = threshDefault, "-0.5" 

        grid = Qt.QGridLayout()
        
        threshLabel = Qt.QLabel( "Maximum Current" )
        threshLabel.setToolTip( "Raw ionic current threshold, which, if dropped below, indicates an event." ) 
        grid.addWidget( threshLabel, 0, 0 )

        self.threshInput = Qt.QLineEdit()
        self.threshInput.setText( threshDefault )
        grid.addWidget( self.threshInput, 0, 2, 1, 1 )

        minCurrentLabel = Qt.QLabel( "Minimum Current (pA):" )
        minCurrentLabel.setToolTip( "This sets a filter requiring all ionic current in an event be above this amount." )
        grid.addWidget( minCurrentLabel, 1, 0 )

        self.minCurrentInput = Qt.QLineEdit()
        self.minCurrentInput.setText( minCurrentDefault )
        grid.addWidget( self.minCurrentInput, 1, 2, 1, 1 )

        timeLabel = Qt.QLabel( "Time:" )
        timeLabel.setToolTip( "This sets a filter requiring all events are of a certain length." )
        grid.addWidget( timeLabel, 3, 0 ) 

        self.timeDirectionInput = Qt.QComboBox()
        self.timeDirectionInput.addItem( ">" )
        self.timeDirectionInput.addItem( "<" )
        grid.addWidget( self.timeDirectionInput, 3, 1 )

        self.timeInput = Qt.QLineEdit()
        self.timeInput.setText( timeDefault )
        grid.addWidget( self.timeInput, 3, 2, 1, 1 )
        return grid

    def set_params( self ):
        '''
        Read in the data from the GUI and use it to customize the rules or threshold of the parser. 
        '''
        self.rules = []
        self.threshold = float( self.threshInput.text() )
        self.rules.append( lambda event: event.max < self.threshold )
        if self.minCurrentInput.text() != '':
            self.rules.append( lambda event: event.min > float( self.minCurrentInput.text() ) )
        if self.timeInput.text() != '':
            if str( self.timeDirectionInput.currentText() ) == '<':
                self.rules.append( lambda event: event.duration < float( self.timeInput.text() ) * 100000. )
            elif str( self.timeDirectionInput.currentText() ) == '>':
                self.rules.append( lambda event: event.duration > float( self.timeInput.text() ) * 100000. )
        if self.rules == []:
            self.rules = None

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return izip(a, b)

class SpeedyStatSplit( parser ):
    '''
    See cparsers.pyx FastStatSplit for full documentation. This is just a
    wrapper for the cyton implementation to add a GUI.
    '''

    def __init__( self, min_width=100, max_width=1000000, window_width=10000, 
        min_gain_per_sample=None, false_positive_rate=None,
        prior_segments_per_second=None, sampling_freq=1.e5, cutoff_freq=None ):

        self.min_width = min_width
        self.max_width = max_width
        self.min_gain_per_sample = min_gain_per_sample
        self.window_width = window_width
        self.prior_segments_per_second = prior_segments_per_second
        self.false_positive_rate = false_positive_rate
        self.sampling_freq = sampling_freq
        self.cutoff_freq = cutoff_freq

    def parse( self, current ):
        parser = FastStatSplit( self.min_width, self.max_width, 
            self.window_width, self.min_gain_per_sample, self.false_positive_rate,
            self.prior_segments_per_second, self.sampling_freq, self.cutoff_freq )
        return parser.parse( current )

    def best_single_split( self, current ):
        parser = FastStatSplit( self.min_width, self.max_width, 
            self.window_width, self.min_gain_per_sample, self.false_positive_rate,
            self.prior_segments_per_second, self.sampling_freq )
        return parser.best_single_split( current )

    def GUI( self ):
        grid = Qt.QGridLayout()
        grid.addWidget( Qt.QLabel( "Minimum Width (samples): "), 0, 0, 1, 3)
        grid.addWidget( Qt.QLabel( "Maximum Width (samples): " ), 1, 0, 1, 3 )
        grid.addWidget( Qt.QLabel( "Window Width (samples): " ), 2, 0, 1, 3 )
        grid.addWidget( Qt.QLabel( "Minimum Gain / Sample: " ), 3, 0, 1, 3 )
        grid.addWidget( Qt.QLabel( "Cutoff Frequency: " ), 4, 0, 1, 3 )
        grid.addWidget( Qt.QLabel( "Sampling Frequency: " ), 5, 0, 1, 3 )
        grid.addWidget( Qt.QLabel( "Prior SPS: " ), 6, 0, 1, 3 )
        grid.addWidget( Qt.QLabel( "FPS Threshold: " ), 7, 0, 1, 3 )
        
        self.minWidth = Qt.QLineEdit()
        self.minWidth.setText('1000')
        self.maxWidth = Qt.QLineEdit()
        self.maxWidth.setText('1000000')
        self.windowWidth = Qt.QLineEdit('10000')
        self.windowWidth.setText('10000')
        self.minGain = Qt.QLineEdit()
        self.minGain.setText('')
        self.cutoff = Qt.QLineEdit()
        self.cutoff.setText('2000')
        self.sampling = Qt.QLineEdit()
        self.sampling.setText('100000')
        self.sps = Qt.QLineEdit()
        self.sps.setText('10')
        self.fps = Qt.QLineEdit()
        self.fps.setText('')

        grid.addWidget( self.minWidth, 0, 3 )
        grid.addWidget( self.maxWidth, 1, 3 )
        grid.addWidget( self.windowWidth, 2, 3 )
        grid.addWidget( self.minGain, 3, 3 )
        grid.addWidget( self.cutoff, 4, 3 )
        grid.addWidget( self.sampling, 5, 3 )
        grid.addWidget( self.sps, 6, 3 )
        grid.addWidget( self.fps, 7, 3 )
        return grid

    def set_params( self ):
        self.min_width = int(self.minWidth.text())
        self.max_width = int(self.maxWidth.text())
        self.window_width = int(self.windowWidth.text())
        self.sampling_freq = float(self.sampling.text())
        self.cutoff_freq = float(self.cutoff.text())

        self.min_gain_per_sample = None if self.minGain.text() == '' else float( self.minGain.text() )
        self.prior_segments_per_second = None if self.sps.text() == '' else float( self.sps.text() )
        self.false_positive_rate = None if self.fps.text() == '' else float( self.fps.text() )

class StatSplit( SpeedyStatSplit ):
    """
    DEPRECATED: USE SPEEDYSTATSPLIT.
    """

    def __init__( self, **kwargs ):
        SpeedyStatSplit.__init__( self, **kwargs )


#########################################
# STATE PARSERS 
#########################################

class snakebase_parser( parser ):
    '''
    A simple parser based on dividing when the peak-to-peak amplitude of a wave exceeds a certain threshold.
    '''

    def __init__( self, threshold=1.5 ):
        self.threshold = threshold

    def parse( self, current ):
        # Take the derivative of the current first
        diff = np.abs( np.diff( current ) )
        # Find the places where the derivative is low
        tics = np.concatenate( ( [0], np.where( diff < 1e-3 )[0], [ diff.shape[0] ] ) )
        # For pieces between these tics, make each point the cumulative sum of that piece and put it together piecewise
        cumsum = np.concatenate( ( [ np.cumsum( diff[ tics[i] : tics[i+1] ] ) for i in xrange( tics.shape[0]-1 ) ] ) )
        # Find the edges where the cumulative sum passes a threshold
        split_points = np.where( np.abs( np.diff( np.where( cumsum > self.threshold, 1, 0 ) ) ) == 1 )[0] + 1
        # Return segments which do pass the threshold
        return [ Segment( current=current[ tics[i]: tics[i+1] ], 
                          start=tics[i], 
                          end=tics[i+1],
                          duration=tics[i+1]-tics[i] ) 
            for i in xrange( 1, tics.shape[0] - 1, 2 ) ]

    def GUI( self ):
        threshDefault = "1.5"

        grid = Qt.QGridLayout()
        grid.setVerticalSpacing(0)
        grid.addWidget( Qt.QLabel( "Threshold" ), 0, 0 )
        self.threshInput = Qt.QLineEdit()
        self.threshInput.setToolTip("Peak to peak amplitude threshold, which if gone above, indicates a state transition.")
        self.threshInput.setText( threshDefault ) 

        grid.addWidget( self.threshInput, 0, 1 )
        grid.addWidget( self.mergerThreshInput, 1, 1 )
        return grid

    def set_params( self ):
        self.threshold = float( self.threshInput.text() )

class novakker_parser( parser ):
    '''
    This parser, implemented by Adam Novak, will attempt to do a filter-derivative based splitting.
    It sets two thresholds, a high threshold, and a low threshold for the derivative. A split occurs
    if the derivative goes above the high threshold, and has reached the low threshold since the last
    time it hit the high threshold. This ensures that the current reaches a form of stability before
    being split again.
    '''

    def __init__( self, low_thresh=1, high_thresh=2 ):
        self.low_thresh = low_thresh
        self.high_thresh = high_thresh

    def parse( self, current ):
        deriv = np.abs( np.diff( current ) )
        # Find the edges of where a series of points have a derivative greater than a threshold, notated as a 'block'
        tics = np.concatenate( ( [ 0 ], np.where( np.abs( np.diff( np.where( deriv > self.low_thresh, 1, 0 ) ) ) == 1 )[0] + 1 , [ deriv.shape[0] ] ) ) 
        # Split points will be the indices of points where the derivative passes a certain threshold and is the maximum of a 'block'
        split_points = []
        for i in xrange( 0, len(tics)-1, 2 ): # For all pairs of edges for a block..
            segment = deriv[ tics[i]:tics[i+1] ] # Save all derivatives in that block to a segment
            if np.argmax( segment ) > self.high_thresh: # If the maximum derivative in that block is above a threshold..
                split_points = np.concatenate( ( split_points, [ tics[i], tics[i+1] ] ) ) # Save the edges of the segment 
                # Now you have the edges of all transitions saved, and so the states are the current between these transitions
        tics = np.concatenate( ( [0], split_points, [ current.shape[0] ] ) )
        return [ Segment( current=current[ tics[i]: tics[i+1] ], 
                          start=tics[i],
                          end=tics[i+1],
                          duration=tics[i+1]-tics[i] ) for i in xrange( 0, tics.shape[0] - 1, 2 ) ]

    def GUI( self ):
        lowThreshDefault = "1e-2"
        highThreshDefault = "1e-1"

        grid = Qt.QGridLayout()
        grid.addWidget( Qt.QLabel( "Low-pass Threshold: " ), 0, 0 )
        grid.addWidget( Qt.QLabel( "High-pass Threshold: " ), 1, 0 )

        self.lowThreshInput = Qt.QLineEdit()
        self.lowThreshInput.setText( lowThreshDefault )
        self.lowThreshInput.setToolTip( "The lower threshold, of which one maximum is found." )
        self.highThreshInput = Qt.QLineEdit()
        self.highThreshInput.setText( highThreshDefault )
        self.highThreshInput.setToolTip( "The higher threshold, of which the maximum must be abov." )

        grid.addWidget( self.lowThreshInput, 0, 1 )
        grid.addWidget( self.highThreshInput, 1, 1 )
        return grid

    def set_params( self ):
        self.low_thresh = float( self.lowThreshInput.text() )
        self.high_thresh = float( self.highThreshInput.text() )