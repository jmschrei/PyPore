#!/usr/bin/env python
# Contact: Jacob Schreiber
#          jacobtribe@yahoo.com
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
from align_segments import *
                 
import time

class Event( Segment ):
    '''
    A container for the ionic current corresponding to an 'event', which means a portion of the 
    file containing useful data. 
    '''
    def __init__( self, current, start, file, n ):
        Segment.__init__( self, current, file=file, duration=current.shape[0]/file.second, 
                          filtered=False, start = start / file.second, states=[], sample=None, n=0 )
         
    def filter( self, order = 1, cutoff = 2000. ):
        '''
        Performs a bessel filter on the selected data, normalizing the cutoff frequency by the nyquist
        limit based on the sampling rate. 
        '''
        from scipy import signal
        nyquist = self.file.second / 2.
        (b, a) = signal.bessel( order, cutoff / nyquist, btype='low', analog=0, output = 'ba' )
        self.current = signal.filtfilt( b, a, self.current )
        self.filtered = True

    def parse( self, parser = parsers.snakebase_parser( threshold = 1.5 ), filter = False ):
        '''
        Ensure that the data is filtered according to a bessel filter, and then applies a plug-n-play 
        state parser which must contain a .parse method. 
        '''
        if self.filtered == False and filter:
            self.filter()
        self.states = np.array( [ Segment( current = segment.current, start = segment.start, 
                                           second = self.file.second, event = self ) for segment in parser.parse( self.current ) ] ) 
        self.n = self.states.shape[0]
        self.state_parser = parser
                
    def plot( self ):
        '''
        Plot the states in an event in a cycle of colors, to show where state detection occured at. 
        '''
        for i, state in enumerate(self.states):
            plt.plot( xrange( state.start, state.start + state.n ), state.current, 'brgc'[i%4] )
        plt.show()

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
    def __init__( self, filename ):
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
        self.events = [ Event( current = segment.current, start = segment.start, file = self, n=0 ) for segment in parser.parse( self.current ) ]
        self.n = len( self.events )
        self.event_parser = parser
        del self.current

    @classmethod 
    def from_database( self, database, host, password, user, AnalysisID=None, filename=None ):
        '''
        Loads the cache for the file, if this exists. Can either provide the AnalysisID to unambiguously
        know which analysis to use, or the filename if you want the most recent analysis done on that file.
        '''
        db = MySQLDatabaseInterface(db=database, host=host, password=password, user=user)

        if AnalysisID != None:
            query = db.read("SELECT Filename FROM AnalysisMetadata \
                                WHERE ID={}".format(AnalysisID) )
            try:
                filename = query[0][0]
            except:
                raise DatabaseError("No analysis found with ID {}".format(AnalysisID))

        elif filename != None:
            AnalysisID = int( db.read( "SELECT ID FROM AnalysisMetadata \
                                        WHERE Filename = '{}' \
                                        ORDER BY Timestamp DESC".format(filename))[0][0])

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
            state_parser_name = self.event[0].state_parser.__class__.__name__
            state_parser_params = repr( self.event[0].state_parser )
        except:
            state_parser_name = "NULL"
            state_parser_params = "NULL"

        metadata = "'{0}',NULL,NULL,'{1}','{2}','{3}','{4}'".format( self.filename,
                                                                     event_parser_name,
                                                                     event_parser_params,
                                                                     state_parser_name,
                                                                     state_parser_params 
                                                                    )

        prevAnalysisID = db.read( "SELECT ID FROM AnalysisMetadata \
                                   WHERE Filename = '{0}' \
                                       AND EventDetector = '{1}' \
                                       AND StateDetector = '{2}'".format( self.filename,
                                                                        event_parser_name,
                                                                        state_parser_name ) )[0][0]
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
        
class Sample( Container ):
    '''A container for events all suggested to be from the same substrate.'''
    def __init__( self, label=None ):
        self.events = []
        self.files = [] 
        self.label = label

from itertools import chain, izip, tee, combinations

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

    def align( self, strategy, model_id=None ):
        if strategy == 'one-vs-all':
            assert model_id is not None
            self._one_vs_all( model_id=model_id )
        elif strategy == 'all-vs-all':
            self._all_vs_all()
        else:
            raise AttributeError( "alignment_type must be one-vs-all or all-vs-all." )

    def _one_vs_all( self, model_id ):
        aligner = SegmentAligner( self.events[model_id], .01, .1 )
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