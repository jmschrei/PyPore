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
        events = [ Segment(current=np.array(current), copy=True, 
                            start=tics[i],
                            duration=current.shape[0] ) for i, current in enumerate( np.split( current, tics[1:-1]) ) ]
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
                self.rules.append( lambda event: event.duration < float( self.timeInput.text() ) )
            elif str( self.timeDirectionInput.currentText() ) == '>':
                self.rules.append( lambda event: event.duration > float( self.timeInput.text() ) )
        if self.rules == []:
            self.rules = None

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return izip(a, b)

class StatSplit( parser ):
    """
    DEPRECATED: USE SPEEDYSTATSPLIT.
    """

    def __init__(self, min_width=1000, max_width=1000000, 
            min_gain_per_sample=0.03, 
                window_width=10000,
                use_log=True,
            splitter="stepwise"):
        """
        create a segmenter with specified minimum and maximum segment lengths.
        (Default for max_width is 100*min_width)
        min_gain_per_sample is the minimum reduction in variance for a split to be done;
            it is multiplied by window_width to get min_gain.
        If use_log, then minimize the log of varainces, 
            otherwise minimize the variance. 
        splitter is "stepwise", "slanted", or a splitter function.
        """
        self.min_width = max( min_width, 1 ) # Avoid divide by 0
        self.max_width = max_width or 100*min_width
        self.min_gain_per_sample = min_gain_per_sample
        self.window_width = window_width or 10*min_width
        assert self.max_width >= self.min_width 
        assert self.window_width >= 2*self.min_width
        self.use_log = use_log
        self.splitter = splitter
    
    def parse(self,current, start=0, end=-1):
        """
        segments current[start:end], where current is a numpy array 
        
        returns list of segments:
            [ (start, duration0, left_end, right_end, rms residual)
                  (a1, duration1,  left_end, right_end, rms residual)
                  ...
            ]
        with   min_width <= ai - a_{i-1} = duration_{i-1} <= max_width
        
        With stepwise segmenting, left_end=right_end=mean of segment
        and rms residual = standard deviation of segment.
        """

        # normalize start and end to be normal subscripts
        n = len(current)
        if start < 0: start += n+1
        if end < 0:  end += n+1
        if start > n: start = n
        if end > n: end = n

        if self.splitter=="slanted":
            self.splitter = self._best_split_slanted
        else:
            self.splitter = self._best_split_stepwise

        self.current = current
        self.cum = np.cumsum( current )
        self.cum2 = np.cumsum( np.multiply( current,current ) )
        if self.splitter != self._best_split_stepwise:
            # For covariance computation, need cumulative sum(current*time), 
            # where time is subscript of current array.
            # Needs to be kept in double precision (or higher), since time steps of 1 can
            # be small relative to total array length.
            self.cum_ct = np.cumsum(np.multiply(current, np.linspace(0,end,num=end,endpoint=False)))

        breakpoints =  self._segment_cumulative(start, end)

        # paired is pairs of breakpoints (start,a1), (a1,a2), (a2,a3), ..., (an,end)
        paired = [p for p in pairwise(chain([start],breakpoints,[end])) ]
        assert len(paired)==len(breakpoints)+1
        
        if self.splitter == self._best_split_stepwise:
            # if stepwise splitting is done, left and right endpoints are just the mean
            # and rms residuals are just the standard deviation
            means = [self._mean_c(pair[0],pair[1]) for pair in paired]
            vars = [self._var_c(pair[0],pair[1]) for pair in paired]
            segments = [ Segment( current=current[start:end],
                              start=start,
                              duration=(end-start) ) for start,end in paired ]
            return segments

        lrs = [self._lr(pair[0],pair[1]) for pair in paired]
        lefts = [alpha+beta*s for (alpha,beta,var),(s,e) in izip(lrs,paired)]
        rights = [alpha+beta*e for (alpha,beta,var),(s,e) in izip(lrs,paired)]
        segments = [ Segment( current=current[start:end],
                              start=start,
                              duration=(end-start) ) for start,end in paired ]
        return segments 

    def _mean_c(self, start, end):
        """mean value of current for segment start:end
        (uses self.cum a numpy array that is the cumulative sum of
            a current trace (that is, self.cum[i] = sum(self.current[0:i+1]) 
            or self.cum=np.cumsum(self.current) ).
    """
        if start==end: return 0
        if start==0: return self.cum[end-1]/end
        return (self.cum[end-1]-self.cum[start-1])/(end-start)

    def _mean_c2(self, start, end):
        """mean value of current**2 for segment start:end
        (uses self.cum2, a numpy array that is the cumulative sum of
        the square of the current)
    """
        if start==end: return 0
        if start==0: return self.cum2[end-1]/end
        return (self.cum2[end-1]-self.cum2[start-1])/(end-start)

    def _var_c(self, start, end):
        """variance of current for segment start:end
        (uses self.cum2, a numpy array that is the cumulative sum of
        the square of the current)
    """
        if start==end: return 0
        if start==0: return self.cum2[end-1]/end - (self.cum[end-1]/end)**2
        return (self.cum2[end-1]-self.cum2[start-1])/(end-start) \
             - ((self.cum[end-1]-self.cum[start-1])/(end-start))**2

    def _mean_ct(self, start, end):
        """mean value of current[t]*t for segment start:end
        (uses self.cum_ct, a numpy array that is the cumulative sum of
        the current[t]*t
    """
        if start==end: return 0
        if start==0: return self.cum_ct[end-1]/end
        return (self.cum_ct[end-1]-self.cum_ct[start-1])/(end-start)
    
    def _mean_t(self, start,end):
        """mean value of start, ..., end-1"""
        return start+ (end-start-1)/2
    
    def _mean_t2(self,start,end):
        """mean value of start**2, ..., (end-1)**2 """
        return (2*end**2 + end*(2*start-3) + 2*start**2-3*start+1)/6.

    def _lr(self,start,end):
        """does a linear regression on self.current, for segment start:end.
        Returns (alpha, beta,var),
        where current[i] =approx alpha+beta*i
        and var is the mean square residual
        """
        xy_bar = self._mean_ct(start,end)
        y_bar = self._mean_c(start,end)
        x_bar = self._mean_t(start,end)
        x2_bar = self._mean_t2(start,end)
        beta = (xy_bar - x_bar*y_bar)/(x2_bar - x_bar**2)
        alpha = y_bar - beta*x_bar
#        print("DEBUG: lr({},{}) x_bar={} x2_bar={}, y_bar={}, xy_bar={}, alpha={}, beta={}".format(
#           start,end,x_bar, x2_bar, y_bar, xy_bar, alpha, beta))
        y2_bar = self._mean_c2(start,end)
        var = y2_bar - 2*alpha*y_bar- 2*beta*xy_bar +alpha**2 + 2*alpha*beta*x_bar+ beta**2*x2_bar
        return (alpha,beta,var)
    
    def _best_split_stepwise(self, start, end):
        """splits self.cum[start:end]  (0<=start<end<=len(self.current)).
        
        Needs self.cum and self.cum2:
        self.cum is a numpy array that is the cumulative sum of
            a current trace (that is, self.cum[i] = sum(self.current[0:i+1]) 
            or self.cum=np.cumsum(self.current) ).
        self.cum2 is a numpy array that is the cumulative sum of
            the square of the current trace.

        Breakpoint is chosen to maximize the probability of the two segments 
        modeled as two Gaussians.  
        Returns (x,decrease in (log)variance as a result of splitting)
        so that segments are seg1=[start:x], seg2=[x:end]
        with   min_width <= x-start and  min_width <= end-x
        (If no such x, returns None.)
        
        Note decrease in log variance is proportional to 
            log p1(seg1) + log p2(seg2) - log pall(seg1+seg2))
        so that this is a maximum-likelihood estimator of splitting point
        """
#   print("DEBUG: splitting", start,"..",end, "min=",self.min_width,file=sys.stderr)
        if end-start< 2*self.min_width:  
#           print("DEBUG: too short", start,"..",end, file=sys.stderr)
            return None
        var_summed = (end-start)*(self._var_c(start,end) if not self.use_log 
                else np.log(self._var_c(start,end)))
        max_gain=self.min_gain_per_sample*self.window_width
        x=None
        for i in xrange(start+self.min_width,end+1-self.min_width):
            low_var_summed = (i-start)*( self._var_c(start,i) if not self.use_log
                    else np.log(self._var_c(start,i)))
            high_var_summed = (end-i)*( self._var_c(i,end) if not self.use_log
                    else np.log(self._var_c(i,end)))
            gain =  var_summed - (low_var_summed+high_var_summed)
            if gain > max_gain:
                max_gain= gain
                x=i
        if x is None: 
#           print("DEBUG: nothing found", start,"..",end, file=sys.stderr)
            return None
        #print("# DEBUG: splitting at x=", x, "gain/sample=", max_gain/self.window_width, file=sys.stderr)
        
        return (x,max_gain)
    
    def _best_split_slanted(self, start, end):
        """
        splits self.cum[start:end]  (0<=start<end<=len(self.current)).
        
        Needs self.cum, self.cum2, and self.cum_ct:
        self.cum is a numpy array that is the cumulative sum of
            a current trace (that is, self.cum[i] = sum(self.current[0:i+1]) 
            or self.cum=np.cumsum(self.current) ).
        self.cum2 is a numpy array that is the cumulative sum of
            the square of the current trace.
        self.cum_ct is a numpy array that is the cumulative sum of current[i]*i
        
        Breakpoint is chosen to maximize the probability of the two segments 
        modeled as two straight-line segments plus Gaussian noise.
        
        Returns (x, (log)variance decrease as a result of splitting)
        so that segments are seg1=[start:x], seg2=[x:end]
        with   min_width <= x-start and  min_width <= end-x
        (If no such x, returns None.)
        """

#   print("DEBUG: splitting", start,"..",end, "min=",self.min_width,file=sys.stderr)
        if end-start< 2*self.min_width:  
#           print("DEBUG: too short", start,"..",end, file=sys.stderr)
            return None
        var_summed = (end-start)*( self._lr(start,end)[2] if not self.use_log
            else log(self._lr(start,end)[2]))
        max_gain=self.min_gain_per_sample*self.window_width
        x=None
        for i in xrange(start+self.min_width,end+1-self.min_width):
            low_var_summed = (i-start)*(self._lr(start,i)[2] if not self.use_log
                else log(self._lr(start,i)[2]))
            high_var_summed = (end-i)*(self._lr(i,end)[2] if not self.use_log
                else log(self._lr(i,end)[2]))
            gain =  var_summed - (low_var_summed+high_var_summed)
            if gain > max_gain:
                max_gain= gain
                x=i
        if x is None: 
#           print("DEBUG: nothing found", start,"..",end, file=sys.stderr)
            return None
        #print("# DEBUG: splitting at x=", x, "gain/sample=", max_gain/self.window_width, file=sys.stderr)
        
        return (x,max_gain)

    # PROBLEM: this recursive splitting can have O(n^2) behavior,
    # if each split only removes min_width from one end, because
    # the self.splitter routines take time proportional to the length of the segment being split.
    # Keeping window_width small helps, since behavior is 
    #  O( window_width/min_width *(end-start) 
    def _segment_cumulative(self, start, end):
        """segments cumulative sum of current and current**2 (in self.cum and self.cum2)
        returns [a1, a2, ...,  an]
        so that segments are [start:a1], [a1:a2], ... [an:end]
        with   min_width <= ai - a_{i-1} <= max_width
        (a0=start a_{n+1}=end)
        """
        
        # scan in overlapping windows to find a spliting point
        split_pair = None
        pseudostart = start
        for pseudostart in xrange(start, end-2*self.min_width, self.window_width//2 ):
            if pseudostart> start+ self.max_width:
            # scanned a long way with no splits, add a fake one at max_width
                split_at = min(start+self.max_width, end-self.min_width)
                #print("# DEBUG: adding fake split at ",split_at, "after", start, file=sys.stderr)
                return [split_at] + self._segment_cumulative(split_at,end) 

            # look for a splitting point
            pseudoend =  min(end,pseudostart+self.window_width)
            split_pair = self.splitter(pseudostart,pseudoend)
            if split_pair is not None: break

        if split_pair is None:
            if end-start <=self.max_width:
                # we've split as finely as we can, subdivide only if end-start>max_width 
                return []
            split_at = min(start+self.max_width, end-self.min_width)
            #print("# DEBUG: adding late fake split at ",split_at, "after", start, file=sys.stderr)
        else:
            split_at,gain = split_pair
        
        # splitting point found, recursively try each subpart
        return  self._segment_cumulative(start,split_at) \
            + [split_at] \
            + self._segment_cumulative(split_at,end)

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
        
        self.minWidth = Qt.QLineEdit()
        self.minWidth.setText('1000')
        self.maxWidth = Qt.QLineEdit()
        self.maxWidth.setText('1000000')
        self.windowWidth = Qt.QLineEdit('10000')
        self.windowWidth.setText('10000')
        self.minGain = Qt.QLineEdit()
        self.minGain.setText('0.05')

        grid.addWidget( self.minWidth, 0, 3 )
        grid.addWidget( self.maxWidth, 1, 3 )
        grid.addWidget( self.windowWidth, 2, 3 )
        grid.addWidget( self.minGain, 3, 3 )
        return grid

    def set_params( self ):
        try:
            self.min_width = int(self.minWidth.text())
            self.max_width = int(self.maxWidth.text())
            self.window_width = int(self.windowWidth.text())
            self.min_gain_per_sample = float(self.minGain.text())
        except:
            pass


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
        return [ Segment( current = current[ tics[i]: tics[i+1] ], start = tics[i] ) for i in xrange( 1, tics.shape[0] - 1, 2 ) ]

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

class FilterDerivativeSegmenter( parser ):
    '''
    This parser will segment an event using a filter-derivative method. It will
    first apply a bessel filter at a certain cutoff to the current, then it will
    take the derivative of that, and segment when the derivative passes a
    threshold.
    '''

    def __init__( self, low_threshold=1, high_threshold=2, cutoff_freq=1000.,
        sampling_freq=1.e5 ):
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.cutoff_freq = cutoff_freq
        self.sampling_freq = sampling_freq

    def parse( self, current ):
        '''
        Apply the filter-derivative method to filter the ionic current.
        '''

        # Filter the current using a first order Bessel filter twice, one in
        # both directions to preserve phase
        from scipy import signal
        nyquist = self.sampling_freq / 2.
        b, a = signal.bessel( 1, self.cutoff_freq / nyquist, btype='low', analog=0, output='ba' )
        filtered_current = signal.filtfilt( b, a, np.array( current ).copy() )

        # Take the derivative
        deriv = np.abs( np.diff( filtered_current ) )

        # Find the edges of the blocks which fulfill pass the lower threshold
        blocks = np.where( deriv > self.low_threshold, 1, 0 )
        block_edges = np.abs( np.diff( blocks ) )
        tics = np.where( block_edges == 1 )[0] + 1 

        # Split points are points in the each block which pass the high
        # threshold, with a maximum of one per block 
        split_points = [0] 

        for start, end in it.izip( tics[:-1:2], tics[1::2] ): # For all pairs of edges for a block..
            segment = deriv[ start:end ] # Save all derivatives in that block to a segment
            if np.argmax( segment ) > self.high_threshold: # If the maximum derivative in that block is above a threshold..
                split_points = np.concatenate( ( split_points, [ start, end ] ) ) # Save the edges of the segment 
                # Now you have the edges of all transitions saved, and so the states are the current between these transitions
        tics = np.concatenate( ( split_points, [ current.shape[0] ] ) )
        tics = map( int, tics )
        return [ Segment( current=current[ tics[i]:tics[i+1] ], start=tics[i] ) 
                    for i in xrange( 0, len(tics)-1, 2 ) ]

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
