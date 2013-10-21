#!/usr/bin/env python
# Contact: Jacob Schreiber
#          jacobtribe@soe.ucsc.com
# DataTypes.py

'''
DataTypes.py contains several classes meant to hold and organize common types of nanopore experimental data
which originated from .abf files, organized from the top down as follows:

Experiment(): A container for several files run in a single experiment. Calling the parse method on an
              experiment will call the parse method of all files in it.

Sample(): A container for events which have all been classified as a single type. Only useful in experiments where
          multiple types of events are useful to be stored, such as an experiment with multiple substrates.

File(): A container which holds the raw current in a file. It is given the name of a file, and will automatically
        find it and read it, assuming that it is stored on the Omar server ( omar.soe.ucsc.edu ). Its parse method
        will use a specified parser (default to lambda parser) to detect events.

Event(): A container for both the ionic current of a given event, and metadata, including a list of 'states' if
         its parse method is called. Can take in any parse methods. 

State(): A container for metadata of a specific 'state' that was parsed from an event.
'''

import numpy as np
from read_abf import read_abf
from matplotlib import pyplot as plt
from hmm import *
from core import *
from database import *
import parsers
from alignment import *
                 
import json
import time
from itertools import chain, izip, tee, combinations

class Event( Segment ):
    '''
    A container for the ionic current corresponding to an 'event', which means a portion of the 
    file containing useful data. 
    '''
    def __init__( self, current, start, file ):
        Segment.__init__( self, current, file=file, duration=current.shape[0]/file.second, 
                          filtered=False, start =start/file.second, states=[], sample=None, n=0 )
         
    def filter( self, order = 1, cutoff = 2000. ):
        '''
        Performs a bessel filter on the selected data, normalizing the cutoff frequency by the nyquist
        limit based on the sampling rate. 
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

    def parse( self, parser = parsers.snakebase_parser( threshold = 1.5 ), filter = False ):
        '''
        Ensure that the data is filtered according to a bessel filter, and then applies a plug-n-play 
        state parser which must contain a .parse method. 
        '''
        if type(self) != Event:
            raise TypeError( "Cannot parse a metaevent. Must have the current." )
        if self.filtered == False and filter:
            self.filter()
        self.states = np.array( [ Segment( current=segment.current, start=segment.start, 
                                           second=self.file.second, event=self ) for segment in parser.parse( self.current ) ] ) 
        self.n = self.states.shape[0]
        self.state_parser = parser

    def delete( self ):
        try:
            del self.current
        except:
            pass

        try:
            self.state_parser
        except:
            pass

        for segment in self.states:
            segment.delete()
        del self

    def apply_hmm( self, hmm ):
        try:
            segments = np.array([ state.mean for state in self.states ])
            segments.shape = (segments.shape[0], 1)
        except AttributeError:
            return

        return hmm.predict( segments )
                
    def plot( self, hmm=None, **kwargs ):
        '''
        Plot the states in an event in a cycle of colors, to show where state detection occured at. 
        '''
        if hmm:
            hmm_seq = self.apply_hmm( hmm )
            if hasattr( hmm, 'colors' ):
                color_cycle = [ hmm.colors[i] for i in hmm_seq ]
            else:
                color_cycle = [ 'rbgmcyk'[i%7] for i in hmm_seq ]

        if 'color' in kwargs.keys():
            if kwargs['color'] == 'cycle':
                color = [ 'brgc'[i%4] for i in xrange(self.n) ]
            elif kwargs['color'] == 'hmm':
                if hmm:
                    hmm_seq = self.apply_hmm( hmm )
                    if hasattr( hmm, 'colors' ):
                        color = [ hmm.colors[i] for i in hmm_seq ]
                    else:
                        color = 'k'
            else:
                color = kwargs['color']
            del kwargs['color']
        else:
            color = 'b'

        if len(color) == 1:
            if self.__class__.__name__ == "MetaEvent":
                x = ( 0, self.duration )
                y_high = lambda z: self.mean + z * self.std
                y_low = lambda z: self.mean - z * self.std
                plt.plot( x, ( self.mean, self.mean ), color=color, **kwargs )
                plt.fill_between( x, y_high(1), y_low(1), color=color, alpha=0.6 )
                plt.fill_between( x, y_high(2), y_low(2), color=color, alpha=0.4 )
                plt.fill_between( x, y_high(3), y_low(3), color=color, alpha=0.2 )
            else:
                plt.plot( np.arange(0, self.duration, 1./self.file.second), self.current, color=color, **kwargs )
        else:
            for c, segment in zip( color, self.states ):
                if isinstance( segment, MetaSegment ):
                    x = ( segment.start, segment.duration+segment.start )
                    y_high = lambda z: segment.mean + z * segment.std
                    y_low = lambda z: segment.mean - z * segment.std
                    plt.plot( x, (segment.mean, segment.mean), color=c, **kwargs )
                    plt.fill_between( x, y_high(1), y_low(1), color=c, alpha=0.6 )
                    plt.fill_between( x, y_high(2), y_low(2), color=c, alpha=0.4 )
                    plt.fill_between( x, y_high(3), y_low(3), color=c, alpha=0.2 )
                else:
                    plt.plot( np.arange(0, segment.duration, 1./self.file.second) + segment.start, segment.current, color=c, **kwargs )

        plt.title( "Event at {filename} at {time}s".format( filename=self.file.filename, time=self.start ) )
        plt.xlabel( "Time (s)" )
        plt.ylabel( "Current (pA)" )
        plt.grid( color='gray', linestyle=':' )
        plt.ylim( self.min - 5, self.max  )
        plt.xlim( 0, self.duration )

    def to_meta( self ):
        for prop in ['mean', 'std', 'duration', 'start', 'min', 'max', 'end', 'start']:
            try:
                self.__dict__[prop] = getattr( self, prop )
            except:
                pass

        try:
            del self.current
        except:
            pass

        for segment in self.states:
            segment.to_meta()

        self.__class__ = type( "MetaEvent", (Event,), self.__dict__ )

    def to_dict( self ):
        keys = ['mean', 'std', 'min', 'max', 'start', 'end', 'duration', 'filtered', 
                'filter_order', 'filter_cutoff', 'n', 'state_parser', 'states' ]
        d = { i: getattr( self, i ) for i in keys if hasattr( self, i ) }
        d['name'] = self.__class__.__name__
        return d

    def to_json( self, filename=None ):
        d = self.to_dict()

        try:
            d['states'] = [ seg.to_dict() for seg in d['states'] ]
        except:
            pass

        try:
            d['state_parser'] = d['state_parser'].to_dict()
        except:
            pass

        _json = json.dumps( d, indent=4, separators=( ',', ' : ' ) )
        if filename:
            with open( filename, 'w' ) as out:
                out.write( _json )
        return _json

    @classmethod
    def from_json( self, _json ) :
        if _json.endswith( ".json" ):
            with open( _json, 'r' ) as infile:
                _json = ''.join(line for line in infile)

        d = json.loads( _json )

        event = MetaSegment() 
        if 'current' not in d.keys():
            event.__class__ = type("MetaEvent", (MetaSegment,), d )
        else:
            event = Event( d['current'], d['start'], )

        

    @classmethod
    def from_segments( self, *segments ):
        try:
            current = np.concatenate( [seg.current for seg in segments] )
            Segment.__init__( self, current=current, states=segments, sample=None, n=len(segments) )
        except AttributeError:
            dur = sum( seg.duration for seg in segments )
            mean = np.mean( seg.mean*seg.duration for seg in segments ) / dur
            std = np.sqrt( sum( seg.std ** 2 * seg.duration ) / dur )  
            MetaSegment.__init__( self, states=segments, sample=None, n=len(segments),
                                  duration=dur, mean=mean, std=std )
            self.__class__ = type( "MetaEvent", (Event,), self.__dict__ )

    @classmethod
    def from_database( self, database, host, password, user, AnalysisID, SerialID ):
        db = MySQLDatabaseInterface(db=database, host=host, password=password, user=user)

        EventID, start, end = db.read( "SELECT ID, start, end FROM Events \
                                        WHERE AnalysisID = {0} \
                                        AND SerialID = {1}".format(AnalysisID, SerialID) )[0]

        state_query = np.array( db.read( "SELECT start, end, mean, std FROM Segments \
                                          WHERE EventID = {}".format(EventID) ) )
        
        states = [ MetaSegment( start=start, end=end, mean=mean, 
                                std=std, duration=end-start ) for start, end, mean, std in state_query ]

        Event.from_segments( self, states )


class File( Segment ):
    '''
    A container for the raw ionic current pulled from a .abf file, and metadata as to
    the events detected in the file. 
    '''
    def __init__( self, filename, **kwargs ):
        timestep, current = read_abf( filename )
        filename = filename.split("\\")[-1].split(".abf")[0]
        Segment.__init__( self, current=current, filename=filename, second=1000./timestep, events=[], sample=None, n=0 )

    def __getitem__( self, index ):
        return self.events[ index ]

    def parse( self, parser = parsers.lambda_event_parser( threshold=90 ) ):
        '''
        Applies one of the plug-n-play event parsers for event detection. The parser must have a .parse method
        which returns a tuple corresponding to the 
        self.start = startg to the start of each event, and the ionic current in them. 
        '''
        self.events = [ Event( current=segment.current, start=segment.start, file=self ) for segment in parser.parse( self.current ) ]
        self.n = len( self.events )
        self.event_parser = parser
        del self.current

    def delete( self ):
        try:
            del self.current
        except:
            pass

        try:
            del self.event_parser
        except:
            pass

        for event in self.events:
            event.delete()
        del self

    def to_meta( self ):
        try:
            del self.current
        except:
            pass

        for event in self.events:
            event.to_meta()

        self.__class__ = type( "MetaFile", (File,), self.__dict__ )

    def to_dict( self ):
        keys = [ 'filename', 'n', 'event_parser', 'mean', 'std', 'duration', 'start', 'end', 'events' ]
        d = { i: getattr( self, i ) for i in keys if hasattr( self, i ) }
        d['name'] = self.__class__.__name__
        return d

    def to_json( self, filename=None ):
        d = self.to_dict()

        devents = []
        for event in d['events']:
            devent = event.to_dict()
            try:
                devent['states'] = [ state.to_dict() for state in devent['states'] ]
                devent['state_parser'] = devent['state_parser'].to_dict()
            except:
                pass
            devents.append( devent )
        d['events'] = devents
        d['event_parser'] = d['event_parser'].to_dict()

        _json = json.dumps( d, indent=4, separators=( ',', ' : ' ) )

        if filename:
            with open( filename, 'w' ) as outfile:
                outfile.write( _json )
        return _json

    @classmethod
    def from_json( self, _json ):
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
            s, e = _json['start']*file.second, _json['end']*file.second

            event = Event( current=file.current[ s:e ], start=s, file=file )

            if _json['filtered']: # FIX HERE !!!!
                event.filter( order=_json['filter_order'], cutoff=_json['filter_cutoff'] )


            event.states = [ Segment( current=file.current[ s+s_json['start']*file.second : s+s_json['end']*file.second ],
                                      second=file.second, event=event, start=s_json['start']*file.second )
                                                                for s_json in _json['states'] ]
            event.state_parser = parser.from_json( json.dumps( _json['state_parser'] ) )
            event.n = _json['n']
            event.filtered = _json['filtered']
            file.events.append( event )

        file.n = d['n']

        return file

    @classmethod 
    def from_database( self, database, host, password, user, AnalysisID=None, filename=None,
                       eventDetector=None, eventDetectorParams=None, stateDetector=None,
                       stateDetectorParams=None, filterCutoff=None, filterOrder=None  ):
        '''
        Loads the cache for the file, if this exists. Can either provide the AnalysisID to unambiguously
        know which analysis to use, or the filename if you want the most recent analysis done on that file.
        '''
        db = MySQLDatabaseInterface(db=database, host=host, password=password, user=user)

        keys = ( "ID", "Filename", "EventDetector", "EventDetectorParams",
                 "StateDetector", "StateDetectorParams", "FilterCutoff", "FilterOrder" )
        vals = ( AnalysisID, filename, eventDetector, eventDetectorParams, stateDetector,
                 stateDetectorParams, filterCutoff, filterOrder )

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
        
        file.parse( parser=parsers.MemoryParse( starts, ends ) )

        for i in SerialID:
            state_query = np.array( db.read( "SELECT start, end FROM Segments \
                                              WHERE EventID = {}".format(EventID[i]) ) )
            try:
                starts, ends = state_query[:,0], state_query[:,1]
                file.events[i].parse( parser=parsers.MemoryParse( starts, ends ) )
            except IndexError:
                pass
        
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
                                           AND StateDetector = '{2}'".format( self.filename,
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

            for j, state in enumerate( event.states ):
                values = "VALUES ({0},{1},{2},{3},{4},{5})".format( int(event_id), 
                                                                    j, 
                                                                    state.start*100000, 
                                                                    state.end*100000,
                                                                    state.mean,
                                                                    state.std,
                                                                   ) 
                db.execute( "INSERT INTO Segments " + values )


class Experiment( Container ):
    def __init__( self, samples=[], files=[], events=[], states=[] ):
        Container.__init__( self, samples=samples, files=files, events=events, states=states, event_count=len(events) )
    def parse( self, parser = parsers.lambda_event_parser( threshold=90 ) ):
        for file in self.files:
            file.parse( parser=parser )
            self.event_count += file.n
            self.add( file.events )
            self.add( file.states )
    def apply_hmm( self, hmm, filter=None, indices=None ):
        segments = []
        hmm = hmm_factory[ hmm ]
        for event in self.get( "events", filter=filter, indices=indices ):
            _, segs = hmm.classify( event )
            segments = np.concatenate( ( segments, segs ) )
        return segments
    def delete( self ):
        try:
            del self.events
        except AttributeError:
            pass

        try:
            del self.states
        except AttributeError:
            pass

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
        aligner = SegmentAligner( self.events[model_id], skip_penalty=skip, backslip_penalty=backslip )
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
        for i, model in enumerate( self.events ):
            aligner = SegmentAligner( model, skip_penalty=skip, backslip_penalty=backslip )
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
        time = [ seg.start for seg in model.states ] + [ model.states[-1].start + model.states[-1].duration ]
        time_steps = [ t for t in flatten(pairwise(time))]
        seg_values = [ v for v in flatten( (seg.mean, seg.mean) for seg in model.states ) ]
        plt.plot( time_steps, seg_values, color='c', linewidth=5, alpha=0.4 )
        
        plt.xlim(0,model.duration)
        plt.ylabel("Current (pA)")
        plt.xlabel("Time (s)")
        plt.title("Alignment Plot")