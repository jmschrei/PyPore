#! /usr/bin/env python
"""
read-abf.py: A library for reading ABF files.

Originally by Kevin Karplus
Modified by Jacob Schreiber

Functions:

read_current
        Reads an abf file of nanopore current information.
        
"""

import sys
import io
import struct   # for packing and unpacking binary
import numpy    # for making compact arrays of floats

ABF_BLOCKSIZE = 512

def read_abf(abf_file):
    """
    Reads binary from an ABF file and returns a tuple of time_step_msec and a numpy array of current.
    """
    abf_file = io.open( abf_file, 'rb' )
    block = abf_file.read(ABF_BLOCKSIZE)
    if len(block)==0:   return      #empty file
    ABF_info =     struct.unpack("<7I4hI16s5I"+ (18*"IIq")+"148x", block)
#    print >> sys.stderr, "DEBUG: ABF_info=", ABF_info
    
    uFileSignature,  uFileVersionNumber, \
        uFileInfoSize, \
    uActualEpisodes,  uFileStartDate,  uFileStartTimeMS,  uStopwatchTime, \
        nFileType, nDataFormat, nSimultaneousScan, nCRCEnable,  uFileCRC, \
        FileGUID,  uCreatorVersion,  uCreatorNameIndex,  uModifierVersion, \
        uModifierNameIndex,  uProtocolPathIndex, \
        ProtocolBlockIndex, ProtocolBytes, ProtocolNumEntries, \
        ADCBlockIndex, ADCBytes, ADCNumEntries, \
        DACBlockIndex, DACBytes, DACNumEntries, \
        EpochBlockIndex, EpochBytes, EpochNumEntries, \
        ADCPerDACBlockIndex, ADCPerDACBytes, ADCPerDACNumEntries, \
        EpochPerDACBlockIndex, EpochPerDACBytes, EpochPerDACNumEntries, \
        UserListBlockIndex, UserListBytes, UserListNumEntries, \
        StatsRegionBlockIndex, StatsRegionBytes, StatsRegionNumEntries, \
        MathBlockIndex, MathBytes, MathNumEntries, \
        StringsBlockIndex, StringsBytes, StringsNumEntries, \
        DataBlockIndex, DataBytes, DataNumEntries, \
        TagBlockIndex, TagBytes, TagNumEntries, \
        ScopeBlockIndex, ScopeBytes, ScopeNumEntries, \
        DeltaBlockIndex, DeltaBytes, DeltaNumEntries, \
        VoiceTagBlockIndex, VoiceTagBytes, VoiceTagNumEntries, \
        SynchArrayBlockIndex, SynchArrayBytes, SynchArrayNumEntries, \
        AnnotationBlockIndex, AnnotationBytes, AnnotationNumEntries, \
        StatsBlockIndex, StatsBytes, StatsNumEntries, \
        = ABF_info

    assert uFileInfoSize == len(block)
    
    # print >> sys.stderr, "DEBUG: uFileSignature=", uFileSignature
    assert uFileSignature == 0x32464241
    # At this point we've confirmed that the file is a little-endian ABF2 file
    # 0x32='2', 0x46='F', 0x42='B', 0x41='A'
    
#    print >> sys.stderr, "DEBUG: uFileVersionNumber=", uFileVersionNumber
#    print >> sys.stderr, "DEBUG: ProtocolBlockIndex=", ProtocolBlockIndex
#    print >> sys.stderr, "DEBUG: ProtocolBytes=", ProtocolBytes
#    print >> sys.stderr, "DEBUG: ProtocolNumEntries=", ProtocolNumEntries

    abf_file.seek(ProtocolBlockIndex*ABF_BLOCKSIZE)
    block = abf_file.read(ProtocolBytes*ProtocolNumEntries)
    if len(block)==0:   return      #empty file
    protocol_info =     struct.unpack("<hf?3xIff5l3hf3h3flfhfhlllhflhffll3hl2h6h2hhlhhf5h3h3f5h304x", block)
#    print >> sys.stderr, "DEBUG: protocol_info=", protocol_info
    
    nOperationMode, fADCSequenceInterval, \
        bEnableFileCompression, uFileCompressionRatio, \
        fSynchTimeUnit, \
        fSecondsPerRun, \
        lNumSamplesPerEpisode, \
        lPreTriggerSamples, \
        lEpisodesPerRun, \
        lRunsPerTrial, \
        lNumberOfTrials, \
        nAveragingMode, \
        nUndoRunCount, \
        nFirstEpisodeInRun, \
        fTriggerThreshold, \
        nTriggerSource, \
        nTriggerAction, \
        nTriggerPolarity, \
        fScopeOutputInterval, \
        fEpisodeStartToStart, \
        fRunStartToStart, \
        lAverageCount, \
        fTrialStartToStart, \
        nAutoTriggerStrategy, \
        fFirstRunDelayS, \
        \
        nChannelStatsStrategy, \
        lSamplesPerTrace, \
        lStartDisplayNum, \
        lFinishDisplayNum, \
        nShowPNRawData, \
        fStatisticsPeriod, \
        lStatisticsMeasurements, \
        nStatisticsSaveStrategy, \
        \
        fADCRange, \
        fDACRange, \
        lADCResolution, \
        lDACResolution, \
        \
        nExperimentType, \
        nManualInfoStrategy, \
        nCommentsEnable, \
        lFileCommentIndex, \
        nAutoAnalyseEnable, \
        nSignalType, \
        \
        nDigitalEnable, \
        nActiveDACChannel, \
        nDigitalHolding, \
        nDigitalInterEpisode, \
        nDigitalDACChannel, \
        nDigitalTrainActiveLogic, \
        \
        nStatsEnable, \
        nStatisticsClearStrategy, \
        \
        nLevelHysteresis, \
        lTimeHysteresis, \
        nAllowExternalTags, \
        nAverageAlgorithm, \
        fAverageWeighting, \
        nUndoPromptStrategy, \
        nTrialTriggerSource, \
        nStatisticsDisplayStrategy, \
        nExternalTagType, \
        nScopeTriggerOut, \
        \
        nLTPType, \
        nAlternateDACOutputState, \
        nAlternateDigitalOutputState, \
        \
        fCellID0,      fCellID1,      fCellID2, \
        \
        nDigitizerADCs, \
        nDigitizerDACs, \
        nDigitizerTotalDigitalOuts, \
        nDigitizerSynchDigitalOuts, \
        nDigitizerType \
     = protocol_info
     
    time_step_msec = fADCSequenceInterval *1e-3
    
    abf_file.seek(ADCBlockIndex*ABF_BLOCKSIZE)
    block = abf_file.read(ADCBytes*ADCNumEntries)
    if len(block)==0:   return      #empty file
    adc_info = []
    scale_factor = []
    offset_to_add = []
    for adc in range(ADCNumEntries):
        adc_info.append(struct.unpack("<h 2h3fhf 2h 9f 2cfc? h 2l 46x", block[adc*ADCBytes:(adc+1)*ADCBytes]))
#        print >> sys.stderr, "DEBUG: adc_info[{0}]={1}".format(adc,adc_info[adc])
    # interpret this adc info
        nADCNum, \
            \
            nTelegraphEnable, \
            nTelegraphInstrument, \
            fTelegraphAdditGain, \
            fTelegraphFilter, \
            fTelegraphMembraneCap, \
            nTelegraphMode, \
            fTelegraphAccessResistance, \
            \
            nADCPtoLChannelMap, \
            nADCSamplingSeq, \
            \
            fADCProgrammableGain, \
            fADCDisplayAmplification, \
            fADCDisplayOffset, \
            fInstrumentScaleFactor, \
            fInstrumentOffset, \
            fSignalGain, \
            fSignalOffset, \
            fSignalLowpassFilter, \
            fSignalHighpassFilter, \
            \
            nLowpassFilterType, \
            nHighpassFilterType, \
            fPostProcessLowpassFilter, \
            nPostProcessLowpassFilterType, \
            bEnabledDuringPN, \
            \
            nStatsChannelPolarity, \
            \
            lADCChannelNameIndex, \
            lADCUnitsIndex\
        = adc_info[adc]

        axonio_scale_factor =  fADCRange/fInstrumentScaleFactor/fSignalGain/fADCProgrammableGain/lADCResolution
        if nTelegraphEnable:  axonio_scale_factor /=fTelegraphAdditGain
        scale_factor.append(axonio_scale_factor)
        offset_to_add.append(fInstrumentOffset-fSignalOffset)
    

    file_array = numpy.memmap(abf_file, mode="r", dtype=numpy.dtype("<i2"), offset=DataBlockIndex*ABF_BLOCKSIZE)
    # Copy and convert to float
    current = numpy.array( file_array[:DataNumEntries:ADCNumEntries], dtype=numpy.float ) * scale_factor[0] + offset_to_add[0]
    abf_file.close()
    return time_step_msec, current
