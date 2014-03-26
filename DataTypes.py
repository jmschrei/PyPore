#!/usr/bin/env python
# Contact: Jacob Schreiber
#          jacobtribe@soe.ucsc.com
# DataTypes.py

'''
DataTypes.py contains several classes meant to hold and organize common types of nanopore 
experimental data. The most common usage involves creating a file from a .abf file, and then
parsing it to get events, and then parsing those events to get segments. See following:

from PyPore.DataTypes import *

file = File( "my_data.abf" )
file.parse( parser=lambda_event_parser() )
for event in file.events:
    event.filter( order=1, cutoff=2000 )
    event.parse( parser=SpeedyStatSplit() )


Experiment(): A container for several files run in a single experiment. Calling the parse method
              on an experiment will call the parse method of all files in it.

Sample(): A container for events which have all been classified as a single type. Only useful in 
          experiments where multiple types of events are useful to be stored, such as an experiment
          with multiple substrates.

File(): A container which holds the raw current in a file. It is given the name of a file, and will
        read it, pull the ionic current, and store it to the object.

Event(): A container for both the ionic current of a given event, and metadata, including a list of
         segments its parse method is called. 
'''

import numpy as np
from read_abf import *
from matplotlib import pyplot as plt
from hmm import *
from core import *
from database import *
from parsers import *
from alignment import *
                 
import json
import time
from itertools import chain, izip, tee, combinations
import itertools as it

class Event( Segment ):
    '''
    A container for the ionic current corresponding to an 'event', which means a portion of the 
    file containing useful data. 
    '''

    def __init__( self, current, start=0, file=None, segments=[], **kwargs ):
        # If a file is provided, divide the time into seconds appropriately
        if file is not None:
            start /= file.second
            duration = len(current) / file.second

        # If segments are provided, set an appropriate duration 
        if len(segments) > 0:
            duration = sum( seg.duration for seg in segments )

        Segment.__init__( self, current, duration=duration, filtered=False, 
            start=start, segments=segments, sample=None, file=file, **kwargs )            

         
    def filter( self, order=1, cutoff=2000. ):
        '''
        Performs a bessel filter on the selected data, normalizing the cutoff frequency by the 
        nyquist limit based on the sampling rate. 
        '''

        if type(self) != Event:
            raise TypeError( "Cannot filter a metaevent. Must have the current." )
        from scipy import signal
        nyquist = self.file.second / 2.
        (b, a) = signal.bessel( order, cutoff / nyquist, btype='low', analog=0, output = 'ba' )
        self.current = signal.filtfilt( b, a, self.current )
        self.filtered = True
        self.filter_order = order
        self.filter_cutoff = cutoff

    def parse( self, parser=SpeedyStatSplit( min_gain_per_sample=0.3 ), hmm=None ):
        '''
        Ensure that the data is filtered according to a bessel filter, and then applies a 
        plug-n-play state parser which must contain a .parse method. If a hmm is given, it will
        use a hmm to assist in the parsing. This occurs by segmenting the event using the parser,
        and then running the segments through the hmm, stringing together consecutive segments
        which yield the same state in the hmm. If no hmm is given, returns the raw parser
        segmentation.
        '''

        # Ensure that a MetaEvent has not been falsely assigned the tag Event
        if type(self) is not Event:
            raise TypeError( "Cannot parse a metaevent. Must have the current." )

        self.segments = np.array( [ Segment( current=segment.current, start=segment.start, 
            second=self.file.second, event=self ) for segment in parser.parse( self.current ) ] ) 

        # If using HMM-Guided Segmentation, run the segments through the HMM
        if hmm:

            # Currently only supports HMMs generated from yahmm
            assert type( hmm ) is Model, "TypeError: hmm must be generated from yahmm package."

            # Find the viterbi path through the events
            logp, states = self.apply_hmm( hmm )
            
            second = self.file.second
            i, j, n, segments = 0, 0, len(self.segments), []

            print "\t".join( state[1].name for state in states )
            while i < n-1:
                print i, j, n, len( self.segments )
                if states[i][1].name != states[i+1][1].name or i == n-2:
                    ledge = self.segments[j]
                    redge = ( self.segments[i] if i < n-2 else self.segments[-1] )
                    segs = self.segments[j:i+1]

                    if self.__class__.__name__ == "MetaEvent":
                        duration = sum( seg.duration for seg in segs )
                        mean = sum( seg.mean*seg.duration for seg in segs )/duration
                        std = math.sqrt( sum( seg.std**2*seg.duration for seg in segs )/duration )

                        segments.append( MetaSegment( start=ledge.start*second,
                                                      duration=duration,
                                                      mean=mean,
                                                      std=std,
                                                      event=self,
                                                      second=self.file.second,
                                                      hidden_state=states[j+1].name ) )

                    else:
                        s, e = int(ledge.start*second), int(redge.start*second+redge.n)
                        current = self.current[ s : e ]
                        segments.append( Segment( start=s,
                                                  current=current,
                                                  event=self,
                                                  second=self.file.second,
                                                  hidden_state=states[j+1][1].name ) )
                    j = i
                i += 1
            self.segments = segments
        self.state_parser = parser

    def delete( self ):
        '''
        Delete all data associated with itself, including making the call on all segments if they
        exist, ensuring that all references get removed immediately.
        '''

        with ignored( AttributeError ):
            del self.current
        with ignored( AttributeError ):
            del self.state_parser
        for segment in self.segments:
            segment.delete()
        del self

    def apply_hmm( self, hmm, algorithm='viterbi' ):
        '''
        Apply a hmm to the segments, returning the log probability and the state
        sequence. Only uses the means of the segments currently. 
        '''

        return getattr( hmm, algorithm )( np.array([ seg.mean for seg in self.segments ]) )

    def plot( self, hmm=None, cmap="Set1", **kwargs ):
        '''
        Plot the segments, colored either according to a color cycle, or according to the colors
        associated with the hidden states of a specific hmm passed in. Accepts all arguments that
        pyplot.plot accepts, and passes them along.
        '''

        if hmm:
            cm = plt.get_cmap( cmap )
            _, hmm_seq = self.apply_hmm( hmm )

            hmm_seq = filter( lambda state: not state[1].is_silent(), hmm_seq )
            n = float( hmm.name.split('-')[1] )-1
            hmm_color_cycle = [ cm( (float( state.name[1:])-1 ) / n ) for i, state in hmm_seq ]

        if 'color' in kwargs.keys(): # If the user has specified a coloring scheme..
            color_arg = kwargs['color'] # Pull out the coloring scheme
            
            if color_arg == 'cycle': # Use a 4-color rotating cycle on the segments
                color = [ 'brgc'[i%4] for i in xrange(self.n) ]

            elif color_arg == 'hmm': # coloring by HMM hidden state
                color = hmm_color_cycle

            elif color_arg == 'model': # Color by the models in the HMM 
                color, labels, i, new_model = [], [], 0, False
                cycle = [ 'b', 'r', 'c', 'k', 'y', 'm', '0.25', 'g', '0.75' ] 
                for index, state in hmm_seq:
                    if not state.is_silent():
                        color.append( cycle[i%9] )
                        if not new_model:
                            labels.append( None )
                        new_model = False
                    elif state.name.endswith( "-start" ):
                        labels.append( state.name[:-6] )
                        new_model = True
                        i += 1
            else:
                color = kwargs['color']

            del kwargs['color']
        else:
            color, color_arg = 'k', 'k'

        # Set appropriate labels 
        if 'label' in kwargs.keys():
            if isinstance( label, str ):
                labels = [ kwargs['label'] ]
            else:
                labels = kwargs['label']
        elif color_arg != 'model':
            labels = []

        if len(color) == 1:
            if self.__class__.__name__ == "MetaEvent":
                x = ( 0, self.duration )
                y_high = lambda z: self.mean + z * self.std
                y_low = lambda z: self.mean - z * self.std
                plt.plot( x, ( self.mean, self.mean ), color=color, **kwargs )
                plt.fill_between( x, y_high(1), y_low(1), color=color, alpha=0.80 )
                plt.fill_between( x, y_high(2), y_low(2), color=color, alpha=0.30 )
                plt.fill_between( x, y_high(3), y_low(3), color=color, alpha=0.15 )
            else:
                plt.plot( np.arange(0, self.duration, 1./self.file.second), 
                    self.current, color=color, **kwargs )
        else:
            for c, segment, l in it.izip_longest( color, self.segments, labels ):
                if self.__class__.__name__ == "MetaEvent":
                    x = ( segment.start, segment.duration+segment.start )
                    y_high = lambda z: segment.mean + z * segment.std
                    y_low = lambda z: segment.mean - z * segment.std
                    plt.plot( x, (segment.mean, segment.mean), color=c, label=l, **kwargs )
                    plt.fill_between( x, y_high(1), y_low(1), color=c, alpha=0.80 )
                    plt.fill_between( x, y_high(2), y_low(2), color=c, alpha=0.30 )
                    plt.fill_between( x, y_high(3), y_low(3), color=c, alpha=0.15 )
                else:
                    plt.plot( np.arange(0, segment.duration, 1./self.file.second)+segment.start, 
                        segment.current, color=c, label=l, **kwargs )

        if len(labels) > 0:
            plt.legend()
        try:
            plt.title( "Event at {filename} at {time}s".format( filename=self.file.filename, time=self.start ) )
        except:
            plt.title( "Event at {time}s".format( time=self.start ))

        plt.xlabel( "Time (s)" )
        plt.ylabel( "Current (pA)" )
        plt.ylim( self.min - 5, self.max  )
        plt.xlim( 0, self.duration )

    def to_meta( self ):
        for prop in ['mean', 'std', 'duration', 'start', 'min', 'max', 'end', 'start']:
            with ignored( AttributeError, KeyError ):
                self.__dict__[prop] = getattr( self, prop )

        with ignored( AttributeError ):
            del self.current

        for segment in self.segments:
            segment.to_meta()

        self.__class__ = type( "MetaEvent", (Event,), self.__dict__ )

    def to_dict( self ):
        keys = ['mean', 'std', 'min', 'max', 'start', 'end', 'duration', 'filtered', 
                'filter_order', 'filter_cutoff', 'n', 'state_parser', 'segments' ]
        d = { i: getattr( self, i ) for i in keys if hasattr( self, i ) }
        d['name'] = self.__class__.__name__
        return d

    def to_json( self, filename=None ):
        d = self.to_dict()

        with ignored( KeyError, AttributeError ):
            d['segments'] = [ seg.to_dict() for seg in d['segments'] ]

        with ignored( KeyError, AttributeError ):
            d['state_parser'] = d['state_parser'].to_dict()

        _json = json.dumps( d, indent=4, separators=( ',', ' : ' ) )
        if filename:
            with open( filename, 'w' ) as out:
                out.write( _json )
        return _json

    @classmethod
    def from_json( cls, _json ) :
        if _json.endswith( ".json" ):
            with open( _json, 'r' ) as infile:
                _json = ''.join(line for line in infile)

        d = json.loads( _json )

        event = MetaSegment() 
        if 'current' not in d.keys():
            event.__class__ = type("MetaEvent", (MetaSegment,), d )
        else:
            event = cls( d['current'], d['start'], )

        return event

    @classmethod
    def from_segments( cls, segments ):
        try:
            current = np.concatenate( [seg.current for seg in segments] )
            return cls( current=current, start=0, segments=segments )

        except AttributeError:
            dur = sum( seg.duration for seg in segments )
            mean = np.mean([ seg.mean*seg.duration for seg in segments ]) / dur
            std = np.sqrt( sum( seg.std ** 2 * seg.duration for seg in segments ) / dur )

            self = cls( current=np.array([ seg.mean for seg in segments ]), start=0, 
                segments=segments, mean=mean, std=std )
            self.__class__ = type( "MetaEvent", (Event,), self.__dict__ )
            return self

    @classmethod
    def from_database( cls, database, host, password, user, AnalysisID, SerialID ):
        db = MySQLDatabaseInterface(db=database, host=host, password=password, user=user)

        EventID, start, end = db.read( "SELECT ID, start, end FROM Events \
                                        WHERE AnalysisID = {0} \
                                        AND SerialID = {1}".format(AnalysisID, SerialID) )[0]

        state_query = np.array( db.read( "SELECT start, end, mean, std FROM Segments \
                                          WHERE EventID = {}".format(EventID) ) )
        
        segments = [ MetaSegment( start=start, end=end, mean=mean, 
                                  std=std, duration=end-start ) for start, end, mean, std in state_query ]

        Event.from_segments( cls, segments )

    @property
    def n( self ):
        return len( self.segments )

class File( Segment ):
    '''
    A container for the raw ionic current pulled from a .abf file, and metadata as to
    the events detected in the file. 
    '''
    def __init__( self, filename, **kwargs ):
        timestep, current = read_abf( filename )
        filename = filename.split("\\")[-1].split(".abf")[0]
        Segment.__init__( self, current=current, filename=filename, second=1000./timestep, 
                                events=[], sample=None )

    def __getitem__( self, index ):
        return self.events[ index ]

    def parse( self, parser = lambda_event_parser( threshold=90 ), delete_current=False ):
        '''
        Applies one of the plug-n-play event parsers for event detection. The parser must have a .parse method
        which returns a tuple corresponding to the 
        self.start = startg to the start of each event, and the ionic current in them. 
        '''
        self.events = [ Event( current=seg.current, start=seg.start, file=self ) 
            for seg in parser.parse( self.current ) ]
        self.event_parser = parser
        
        if delete_current:    # Deletes the current array to save significant amounts of space.
            del self.current  # Events hold a hard copy of a smaller array of current. 

    def delete( self ):
        '''
        Delete the file, and everything that is a part of it, including the ionic current stored
        to it, other properties, and all events. Calls delete on all events to remove them and all
        underlying data. 
        '''
        with ignored( AttributeError ):
            del self.current

        with ignored( AttributeError ):
            del self.event_parser

        for event in self.events:
            event.delete()
        del self

    def plot( self, color_events=True ):
        '''
        Allows you to plot a file, optionally coloring the events in a file.
        '''
        step = 1./self.second
        if color_events:
            events = [(event.start, event.end) for event in self.events]

            current = self.current[ 0:int(events[0][0]*self.second):100 ]
            plt.plot( np.arange(0, len(current) )*step*100, current, c='k', alpha=0.66 ) 

            for i, (start, end) in enumerate( events ):
                si, ei = int(start*self.second), int(end*self.second)
                current = self.current[ si:ei:5 ]
                plt.plot( np.arange(0, len(current) )*step*5+start, current, c='c', alpha=0.66 )

                si, ei = ei, None if i == len(events)-1 else int( events[i+1][0]*self.second )
                current = self.current[ si:ei:100 ]
                plt.plot( np.arange( 0, len(current) )*step*100+end, current, c='k', alpha=0.66 )

        else:
            plt.plot( np.arange( 0, len(self.current)/self.second ), self.current, c='k', alpha=0.66 )

        plt.title( "File {}".format( self.filename ) )
        plt.ylabel( "Current (pA)" )
        plt.xlabel( "Time (s)" )
        plt.show()

    def to_meta( self ):
        '''
        Remove the ionic current stored for this file, and do the same for all underlying
        structures in order to remove all references to that list. 
        '''
        with ignored( AttributeError ):
            del self.current

        for event in self.events:
            event.to_meta()

        self.__class__ = type( "MetaFile", (File,), self.__dict__ )

    def to_dict( self ):
        '''
        Return a dictionary of the important data that underlies this file. This is done with the
        intention of producing a json from it. 
        '''
        keys = [ 'filename', 'n', 'event_parser', 'mean', 'std', 'duration', 'start', 'end', 'events' ]
        d = { i: getattr( self, i ) for i in keys if hasattr( self, i ) }
        d['name'] = self.__class__.__name__
        return d

    def to_json( self, filename=None ):
        '''
        Return a json (in the form of a string) that represents the file, and allows for
        reconstruction of the instance from, using cls.from_json. 
        '''
        d = self.to_dict()

        devents = []
        for event in d['events']:
            devent = event.to_dict()
            try:
                devent['segments'] = [ state.to_dict() for state in devent['segments'] ]
                devent['state_parser'] = devent['state_parser'].to_dict()
            except:
                with ignored( KeyError, AttributeError ):
                    del devent['segments']
                    del devent['state_parser']
            devents.append( devent )

        d['events'] = devents
        d['event_parser'] = d['event_parser'].to_dict()

        _json = json.dumps( d, indent=4, separators=( ',', ' : ' ) )

        if filename:
            with open( filename, 'w' ) as outfile:
                outfile.write( _json )
        return _json

    @classmethod
    def from_json( cls, _json ):
        '''
        Read in a json (string format) and produce a file instance and all associated event
        instances. 
        '''

        if _json.endswith(".json"):
            with open( _json, 'r' ) as infile:
                _json = ''.join(line for line in infile)

        d = json.loads( _json )

        if d['name'] != "File":
            raise TypeError( "JSON does not encode a file" )

        file = File( d['filename']+".abf" )
        file.event_parser = parser.from_json( json.dumps(d['event_parser']) )
        file.events = []

        for _json in d['events']:
            s, e = int(_json['start']*file.second), int(_json['end']*file.second)

            event = Event( current=file.current[ s:e ], start=s, file=file )

            if _json['filtered']:
                event.filter( order=_json['filter_order'], cutoff=_json['filter_cutoff'] )

            event.segments = [ Segment( current=event.current[ int(s_json['start']*file.second) : int(s_json['end']*file.second) ],
                                        second=file.second, event=event, start=s_json['start']*file.second )
                                                                for s_json in _json['segments'] ]
            event.state_parser = parser.from_json( json.dumps( _json['state_parser'] ) )
            event.filtered = _json['filtered']
            file.events.append( event )

        return file

    @classmethod 
    def from_database( cls, database, host, password, user, AnalysisID=None, filename=None,
                       eventDetector=None, eventDetectorParams=None, segmenter=None,
                       segmenterParams=None, filterCutoff=None, filterOrder=None  ):
        '''
        Loads the cache for the file, if this exists. Can either provide the AnalysisID to unambiguously
        know which analysis to use, or the filename if you want the most recent analysis done on that file.
        '''
        db = MySQLDatabaseInterface(db=database, host=host, password=password, user=user)

        keys = ( "ID", "Filename", "EventDetector", "EventDetectorParams",
                 "segmenter", "segmenterParams", "FilterCutoff", "FilterOrder" )
        vals = ( AnalysisID, filename, eventDetector, eventDetectorParams, segmenter,
                 segmenterParams, filterCutoff, filterOrder )

        query_list = []
        for key, val in zip( keys, vals ):
            if val:
                if key not in ['ID', 'FilterCutoff', 'FilterOrder']:
                    query_list += ["{key} = '{val}'".format( key=key, val=val )]
                else:
                    query_list += ["{key} = {val}".format( key=key, val=val )]

        query = "SELECT * FROM AnalysisMetadata WHERE "+" AND ".join(query_list)+" ORDER BY TimeStamp DESC" 

        try:
            filename, _, AnalysisID = db.read( query )[0][0:3]
        except:
            raise DatabaseError("No analysis found with given parameters.")

        try:
            file = File(filename+".abf")
        except:
            raise IOError("File must be in local directory to parse from database.")

        query = np.array( db.read( "SELECT ID, SerialID, start, end FROM Events \
                                    WHERE AnalysisID = {0}".format(AnalysisID) ) )
        EventID, SerialID, starts, ends = query[:, 0], query[:, 1], query[:, 2], query[:,3]
        starts, ends = map( int, starts ), map( int, ends )
        
        file.parse( parser=MemoryParse( starts, ends ) )

        for i in SerialID:
            state_query = np.array( db.read( "SELECT start, end FROM Segments \
                                              WHERE EventID = {}".format(EventID[i]) ) )
            with ignored( IndexError ):
                starts, ends = state_query[:,0], state_query[:,1]
                file.events[i].parse( parser=MemoryParse( starts, ends ) )
        
        return file


    def to_database( self, database, host, password, user ):
        '''
        Caches the file to the database. This will create an entry in the AnalysisMetadata table
        for this file, and will add each event to the Event table, and each Segment to the Segment
        table. The split points are stored de facto due to the start and end parameters in the events
        and segments, and so this segmentation can be reloaded using from_database.
        '''
        db = MySQLDatabaseInterface(db=database, host=host, password=password, user=user)

        event_parser_name = self.event_parser.__class__.__name__
        event_parser_params = repr( self.event_parser )
        try:
            state_parser_name = self.events[0].state_parser.__class__.__name__
            state_parser_params = repr( self.events[0].state_parser )
        except:
            state_parser_name = "NULL"
            state_parser_params = "NULL"

        try:
            filter_order = self.events[0].filter_order
            filter_cutoff = self.events[0].filter_cutoff
        except:
            filter_order = "NULL"
            filter_cutoff = "NULL"

        metadata = "'{0}',NULL,NULL,'{1}','{2}','{3}','{4}', {5}, {6}".format( self.filename,
                                                                     event_parser_name,
                                                                     event_parser_params,
                                                                     state_parser_name,
                                                                     state_parser_params,
                                                                     filter_order,
                                                                     filter_cutoff
                                                                    )
        try:
            prevAnalysisID = db.read( "SELECT ID FROM AnalysisMetadata \
                                       WHERE Filename = '{0}' \
                                           AND EventDetector = '{1}' \
                                           AND segmenter = '{2}'".format( self.filename,
                                                                            event_parser_name,
                                                                            state_parser_name ) )[0][0]
        except IndexError:
            prevAnalysisID = None

        if prevAnalysisID is not None:
            prevAnalysisEventIDs = db.read( "SELECT ID FROM Events \
                                 WHERE AnalysisID = {0}".format( prevAnalysisID ) )
            for ID in prevAnalysisEventIDs:
                ID = ID[0]
                db.execute( "DELETE FROM Segments WHERE EventID = {0}".format( ID ) )
                db.execute( "DELETE FROM Events WHERE ID = {0}".format( ID) )
            db.execute( "DELETE FROM AnalysisMetadata WHERE ID = {0}".format( prevAnalysisID ) )

        db.execute("INSERT INTO AnalysisMetadata VALUES({0})".format(metadata))
        analysisID = db.read("SELECT ID FROM AnalysisMetadata ORDER BY Timestamp DESC")[0][0] 

        for i, event in enumerate( self.events ):
            values = "VALUES ({0},{1},{2},{3},{4},{5},NULL)".format( int(analysisID),
                                                                     i,
                                                                     event.start*100000,
                                                                     event.end*100000,
                                                                     event.mean,
                                                                     event.std 
                                                                    )
            db.execute( "INSERT INTO Events " + values )

            event_id = db.read( "SELECT ID FROM Events \
                                 WHERE AnalysisID = '{0}' \
                                 AND SerialID = {1}".format( analysisID,
                                                             i 
                                                            ) ) [-1][-1]

            for j, seg in enumerate( event.segments ):
                values = "VALUES ({0},{1},{2},{3},{4},{5})".format( int(event_id), 
                                                                    j, 
                                                                    seg.start*100000, 
                                                                    seg.end*100000,
                                                                    seg.mean,
                                                                    seg.std,
                                                                   ) 
                db.execute( "INSERT INTO Segments " + values )

    @property
    def n( self ):
        return len( self.events )


class Experiment( Container ):
    '''
    An experiment represents a series of files which all are to be analyzed together, and have
    their results bundled. This may be many files run on the same nanopore experiment, or the
    same conditions being tested over multiple days. It attempts to construct a fast and efficient
    way to store the data. Functions are as close to the file functions as possible, as you are
    simply applying these functions over multiple files.
    '''

    def __init__( self, samples=[], files=[], events=[], segments=[] ):
        Container.__init__( self, samples=samples, files=files, events=events, 
                            segments=segments, event_count=len(events) )

    def parse( self, parser=lambda_event_parser( threshold=90 ) ):
        '''
        The same interface as the file parsing method. Does not parse events; returns parsed
        file objects, the same way that file.parse returns events.  
        '''

        for file in self.files:
            file.parse( parser=parser )
            self.event_count += file.n
            self.add( file.events )
            self.add( file.segments )

    def apply_hmm( self, hmm, filter=None, indices=None ):
        segments = []
        for event in self.get( "events", filter=filter, indices=indices ):
            _, segs = hmm.viterbi( np.array( seg.mean for seg in event.segments ) )
            segments = np.concatenate( ( segments, segs ) )
        return segments

    def delete( self ):
        with ignored( AttributeError ):
            del self.events

        with ignored( AttributeError ):
            del self.segments

        for sample in self.samples:
            sample.delete()
        for file in self.files:
            file.delete()
        del self 
 
class Sample( Container ):
    '''A container for events all suggested to be from the same substrate.'''
    def __init__( self, label=None ):
        self.events = []
        self.files = [] 
        self.label = label

    def delete( self ):
        with ignored( AttributeError ):
            for file in self.files:
                file.delete()

        for event in self.events:
            event.delete()
        del self.events
        del self.files
        del self

def flatten( listOfLists ):
    return chain.from_iterable( listOfLists )

def pairwise( iterable ):
    a, b = tee( iterable )
    next(b, None)
    return izip( a, b )

def split_on_neg_dur( segments ):
    start=0
    for stop,seg in enumerate(segments):
        if seg[1]>=0: continue
        if stop>start: yield segments[start:stop]
        start=stop+1
    if segments[-1][1]>=0: yield segments[start:]

class MultipleEventAlignment( object ):
    '''
    This object takes in multiple events, which are made up of segments, and allows for
    methods that align the data in various ways. The two strategies are one-vs-all, if a model
    is selected, or all-vs-all with neighbor joining to prune the graph.
    '''
    def __init__( self, events ):
        self.events = events[:-1]
        self.aligned_events = []
        self.n = len(self.events)
        self.pairwise = np.zeros( (self.n, self.n) )
        self.score = 0

    def align( self, strategy, model_id=None, skip=0.01, backslip=0.1 ):
        if strategy == 'one-vs-all':
            assert model_id is not None
            self._one_vs_all( model_id=model_id, skip=skip, backslip=backslip )
        elif strategy == 'all-vs-all':
            self._all_vs_all()
        else:
            raise AttributeError( "alignment_type must be one-vs-all or all-vs-all." )

    def _one_vs_all( self, model_id, skip, backslip ):
        get = lambda attr: lambda event: np.array([getattr( seg, attr ) for seg in event.segments])
        model = self.events[model_id]
        aligner = SegmentAligner( get( 'mean' )( model ),
                                  get( 'std' )( model ),
                                  get( 'duration' )( model ),
                                  skip_penalty=skip, backslip_penalty=backslip )
        self.aligned_events = []
        self.model_id = model_id
        for i, event in enumerate( self.events ):
            if i == model_id:
                continue
            self.pairwise[i][model_id], order = aligner.align( event )
            aligned_event = aligner.transform( event, order )
            if aligned_event:
                self.aligned_events.append( aligned_event )
        self.score = np.sum( self.pairwise ) / i

    def _all_vs_all( self, skip, backslip ):
        get = lambda attr: lambda event: np.array([getattr( seg, attr ) for seg in event.segments])
        for i, model in enumerate( self.events ):
            aligner = SegmentAligner( get( 'mean' )( model ),
                                      get( 'std' )( model ),
                                      get( 'duration' )( model ),
                                       skip_penalty=skip, backslip_penalty=backslip )
            for j, event in enumerate( self.events ):
                if i == j:
                    continue
                self.pairwise[i][j] = self.pairwise[j][i] = aligner.align( event )
        self.score = np.sum( self.pairwise ) / np.prod( self.pairwise.shape )

    def plot( self ):
        for i, event in enumerate( self.aligned_events ):
            for segments in split_on_neg_dur( event ):
                time = [ seg[0] for seg in segments ] + [ segments[-1][0] + segments[-1][1] ]
                time_steps = [ t for t in flatten( pairwise( time ) ) ]
                seg_values = [ v for v in flatten( (seg[2], seg[3] ) for seg in segments ) ]
                plt.plot( time_steps, seg_values, color='rgbmyk'[i%6], linewidth=2, alpha=0.5 )
        
        model = self.events[self.model_id]
        time = [ seg.start for seg in model.segments ] + [ model.segments[-1].start + model.segments[-1].duration ]
        time_steps = [ t for t in flatten(pairwise(time))]
        seg_values = [ v for v in flatten( (seg.mean, seg.mean) for seg in model.segments ) ]
        plt.plot( time_steps, seg_values, color='c', linewidth=5, alpha=0.4 )
        
        plt.xlim(0,model.duration)
        plt.ylabel("Current (pA)")
        plt.xlabel("Time (s)")
        plt.title("Alignment Plot")
