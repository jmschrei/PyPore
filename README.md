# PyPore
## _Analysis of Nanopore Data_

The PyPore package is based off of a few core data analysis packages in order to provide a consistent and easy framework for handling nanopore data in the UCSC nanopore lab. The packages it requires are:

* numpy
* scipy
* matplotlib
* sklearn

Packages which are not required, but can be used, are:

* mySQLdb
* cython
* PyQt4

Let's get started!

# Files

Nanopore data files consist primarily of current levels corresponding to ions passing freely through the nanopore ("open channel"), and a blockages as something passes through the pore, such as a DNA strand ("events"). Data from nanopore experiments are stored in Axon Binary Files (extension .abf), as a sequence 32 bit floats, and supporting information about the hardware. They can be opened and loaded with the following:

```
from PyPore.DataTypes import *
file = File( "My_File.abf" ) 
```

The File class contains many methods to simplify the analysis of these files. The simplest analysis to do is to pull the events, or blockages of current, from the file, while ignoring open channel. Let's say that we are looking for any blockage of current which causes the current to dip from an open channel of ~120 pA. To be conservative, we set the threshold the current has to dip before being significant to 110 pA. This can be done simply with the file's parse method, which requires a parser class which will perform the parsing. The simplest event detector is the *lambda_event_parser, which has a keyword *threshold, indicating the raw current that serves as the threshold. 

```
from PyPore.DataTypes import *
file = File( "My_File.abf" )
file.parse( parser=lambda_event_parser( threshold=110 ) ) 
```

The events are now stored as Event objects in file.events. The only other important file methods involve loading and saving them to a cache, which we'll cover later. Files also have the properties mean, std, and n (number of events). 

# Events

Events are segments of current which correspond to something passing through the nanopore. We hope that it is something which we are interested in, such as DNA or protein. An event is usually made up of a sequence of discrete segments of current, which should correspond to reading some region of whatever is passing through. In the best case, each discrete segment in an event corresponds to a single nucleotide of DNA, or a single amino acid of a protein passing through.

Events are often noisy, and transitions between them are quick, making filtering a good option for trying to see the underlying signal. Currently only [bessel filters](http://en.wikipedia.org/wiki/Bessel_filter) are supported for filtering tasks, as they've been shown to perform very well.

Let's continue with our example, and imagine that now we want to filter each event, and look at it! The filter method has two parameters, order and cutoff, defaulting to order=1 and cutoff=2000. (Note that we now import pyplot as well.)

```
from PyPore.DataTypes import *
from matplotlib import pyplot as plt

file = File( "My_File.abf" )
file.parse( parser=lambda_event_parser( threshold=110 ) ) 

for event in file.events:
    event.filter( order=1, cutoff=2000 )
    event.plot()
    plt.show()
```

Currently, _lambda_event_parser( threshold )_ and _MemoryParse( starts, ends )_ are the only two parsers. MemoryParse takes in two lists, one of starts of events, and one of ends of events, and will cut a file into it's respective events. This is useful if you've done an analysis before and remember where the split points are. 

The plot command will draw the event on whatever canvas you have, allowing you to make subplots with the events or add them into GUIs (such as Abada!), with the downside being that you need to use plt.show() after calling the plot command. The plot command wraps the pyplot.plot command, allowing you pass in any argument that could be used by pyplot.plot, for example:

```
event.plot( alpha=0.5, marker='o' ) 
```

The next step is usually to try to segment this event into it's discrete states. There are several segmenters which have been written to do this, of which currently *StatSplit is the best, written by Dr. Kevin Karplus and based on a recursive maximum likelihood algorithm.  This algorithm was sped up by rewritting it in Cython, leading to *SpeedyStatSplit, which is a python wrapper for the cython code. Segmenting an event is the same process as detecting events in a file, by using the parse method on an event and passing in a parser.

Let's say that now we want to segment an event and view it. Using the same plot command for the event, we can specify to color by 'cycle', which colors the segments in a four-color cycle for easy viewing. SpeedyStatSplit takes in several parameters, of which *min_gain_per_sample is the most important, and 0.50 to 1.50 usually provide an adequate level to parse at, with higher numbers leading to less segments. 

```
from PyPore.DataTypes import *
from matplotlib import pyplot as plt

file = File( "My_File.abf" )
file.parse( parser=lambda_event_parser( threshold=110 ) ) 

for event in file.events:
    event.filter( order=1, cutoff=2000 )
    event.parse( parser=SpeedyStatSplit( min_gain_per_sample=0.50 ) )
    event.plot( color='cycle' )
    plt.show()
```

Currently, the parsers implemented for segmentation are _SpeedyStatSplit( min_width, max_width, min_gain_per_sample, window_width, use_log, splitter)_, _StatSplit( *same as before* )_, _snakebase\_parser( threshold, merger_thresh )_, and _novakker\_parser( low_thresh, high_thresh, merger_thresh )_. SpeedyStatSplit is the cython implementation of the StatSplit code. Snakebase parser will segment whenever the distance between two adjacent local optima of the wave is above a threshold. The novakker parser will split whenever the derivative of the current is above a high threshold, and has passed below the low threshold since the last split. 

However, both Files and Events inherit from the Segment class, described below. This means that any of the parsers will work on either files or events, as long as they have an underlying .current attribute. 

The last core functionality is the ability to apply an hidden markov model (HMM) to an event, and see which segments correspond to which hidden states. Any hmm (or more complex model!) can be used as long as it has a predict method (like sklearn hmms), but the PyPore.hmm module gives a core class, NanoporeHMM, and several examples, of how to make an hmm that will be useful. Let's say that we're dealing with an event that appears to switch between a high current state and a low current state, corresponding to a DNA strand ratcheting back and forth between two nucleotides. In order to try to group these segments, I've written an HMM named Bifurcator which will classify these segments as belonging to one group or the other. I want to visualize it's performance! 

```
from PyPore.DataTypes import *
from PyPore.hmm import Bifurcator
from matplotlib import pyplot as plt

file = File( "My_File.abf" )
file.parse( parser=lambda_event_parser( threshold=110 ) ) 

for event in file.events:
    event.filter( order=1, cutoff=2000 )
    event.parse( parser=SpeedyStatSplit( min_gain_per_sample=0.50 ) )
    event.apply_hmm( hmm=Bifurcator )
    event.plot( color='hmm' )
    plt.show()
```

You'll see that I had to import Bifurcator from the hmm module. By just using the event apply_hmm method, I can apply the hmm with minimal effort, and by coloring by 'hmm', segments will now be colored according to which hidden state in the HMM they correspond to. 

Event objects also have the properties start, end, duration, mean, std, and n (number of segments after segmentation has been performed). 

# Segments

Segments are short sequences of current samples, usually which appear to be from the same distribution. They are the core place where data are stored, as usually an event is analyzed by the metadata stored in each state. Segments have the attributes current, which stores the raw current samples, in addition to mean, std, duration, start, end, min, and max. They do not have any core methods.

If storing the raw sequence of current samples is too memory intensive, there are two ways to get rid of the current attribute. 

1) Initialize a MetaSegment object, instead of a Segment one, and feed in whatever statistics you'd like to save. This will prevent the current from ever being saved to a second object. For this example, lets assume you have a list of starts and ends of segments in an event, such as loading them from a cache.

```
event = Event( current=[...], start=..., file=... )
event.segments =  [ MetaSegment( mean=np.mean( event.current[start:end] ),
                                 std=np.std( event.current[start:end] ),
                                 duration=(end-start)/100000 ) for start, end in zip( starts, ends ) ]
```

In this example, references to the current are not stored in both the event and the segment, which may save memory if you wish to not store the raw current after analyzing a file. The duration here is divided by 100,000 because abf files store 100,000 samples per second, and we wished to convert from the integer index of the start and end to the second index of the start and end.

2) If you have the memory to store the references, but don't want to accumulate them past a single event, you can parse a file normally, and produce normal segments, then call the function to_meta() to turn them into MetaSegments. This does not require any calculation on the user part, but does require the segment have contained all current samples at one point.

```
event = Event( current=[...], start=..., file=... )
event.parse( parser=SpeedyStatSplit() )
for segment in event.segments:
    segment.to_meta() 
```

# Saving Analyses

If you perform an analysis and wish to save the results, there are multiple ways for you to do such. These operation seems common, for applications such as testing a HMM. If you write a HMM and want to make modifications to it, it would be useful to not have to redo the segmentation, but instead simply load up the same segmentation from the last time you did it. Alternatively, you may have a large batch of files you wish to analyze, and want to grab the metadata for each file to easily read after you go eat lunch, so you don't need to deal with the whole files.

## MySQL Database:
The first is to store it to a MySQL database. The tables must be properly made for this-- see database.py if you want to see how to set up your own database to store PyPore results. If you are connected to the UCSC SoE secure network, there is a MySQL database, named chenoo, which will allow you to store an analysis. This is done on the file level, in order to preserve RDBMS format. 

```
from PyPore.DataTypes import *

file = File( "My_File.abf" )
file.parse( parser=lambda_event_parser( threshold=110 ) ) 

for event in file.events:
    event.filter( order=1, cutoff=2000 )
    event.parse( parser=SpeedyStatSplit( min_gain_per_sample=0.50 ) )
    event.plot( color='cycle' )
    plt.show()
file.to_database( database="chenoo", host="...", password="...", user="..." )
```

Host, password, and user must be set for your specific database. These files can then be read back by the following code.

'''
from PyPore.DataTypes import *

file = File.from_database( database="chenoo", host="...", password="...", user="...", AnalysisID, filename, eventDetector, eventDetectorParams, segmenter, segmenterParams, filterCutoff, filterOrder )

'''

This will load the file back, with the previous segmentations. This will be anywhere from 10x to 1000x faster than performing the segmentation again. The time depends on how stable your connection with the database is, and how complex the analysis you did was. 

Now, it seems like there are a lot of parameters after user! You need to fill in as many of these as you can, to help identify which analysis you meant. AnalysisID is a primary key, but is also assigned by the database automatically when you stored it, so it is possible you do not know it. If you connect to MySQL independently and look up that ID, you can use it solely to identify which file you meant. If you do not provide enough information to uniquely identify a file, you may get an incorrect analysis.

## JSON File
A more portable and simple way to store analyses is to save the file to a json. This can be done simply with the following code.

```
from PyPore.DataTypes import *

file = File( "My_File.abf" )
file.parse( parser=lambda_event_parser( threshold=110 ) ) 

for event in file.events:
    event.filter( order=1, cutoff=2000 )
    event.parse( parser=SpeedyStatSplit( min_gain_per_sample=0.50 ) )
    event.plot( color='cycle' )
    plt.show()
file.to_json( filename="My_File.json" )
```

The representation of your analysis will then be available as a human-readable json format. It may not be particularly fun to look at, but you will be able to read the metadata from the file. A snippet from an example file looks like the following:

{
     "name" : "File",
     "n" : 16,
     "event_parser" : {
         "threshold" : 50.0,
         "name" : "lambda_event_parser"
     },
     "duration" : 750.0,
     "filename" : "13823006-s06",
     "events" : [
         {
             "std" : 1.9335278997265508,
             "end" : 31.26803,
             "state_parser" : {
                 "min_gain_per_sample" : 0.5,
                 "min_width" : 1000,
                 "window_width" : 10000,
                 "max_width" : 1000000,
                 "name" : "SpeedyStatSplit"
             },
             "name" : "Event",
             "min" : 16.508111959669066,
             "max" : 48.73997069621818,
             "segments" : [
                 {
                     "std" : 2.8403093295527646,
                     "end" : 0.01,
                     "name" : "Segment",
                     "min" : 22.330505378907066,
                     "max" : 48.73997069621818,
                     "start" : 0.0,
                     "duration" : 0.01,
                     "mean" : 27.341223956001969
                 },
                 {
                     "std" : 0.5643329015609988,
                     "end" : 2.5060499999999997,
                     "name" : "Segment",
                     "min" : 17.67660726490438,
                     "max" : 26.554361458946911,
                     "start" : 0.01,
                     "duration" : 2.49605,
                     "mean" : 24.084380592526145
                ....

The file continues to list every event, and every segment in every event. The code to reconstruct an analysis from a json file is just as long as the code to reconstruct from the database.

'''
from PyPore.DataTypes import *

file = File.from_json( "My_File.json" )
'''

This is usually faster than loading from a database, solely due to not having to connect across a network and stream data, and instead reading locally. 
