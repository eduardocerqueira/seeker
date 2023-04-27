#date: 2023-04-27T16:54:16Z
#url: https://api.github.com/gists/54a60d570275e77c06aa1c2567df6cba
#owner: https://api.github.com/users/mmusich

import sys
import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Run3_cff import Run3
process = cms.Process("BeamMonitorLegacy", Run3)

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 1

# process.MessageLogger = cms.Service("MessageLogger",
#                                     debugModules = cms.untracked.vstring('*'),
#                                     cerr = cms.untracked.PSet(
#                                         threshold = cms.untracked.string('WARNING')
#                                     ),
#                                     destinations = cms.untracked.vstring('cerr'))


readFiles = cms.untracked.vstring('/store/express/Run2023B/ExpressPhysics/FEVT/Express-v1/000/366/451/00001/c2a2150f-9a56-4d75-a629-be4294a5de30.root',
                                  '/store/express/Run2023B/ExpressPhysics/FEVT/Express-v1/000/366/451/00000/010867a3-7a3f-4beb-8524-3ff5000a776e.root',
                                  '/store/express/Run2023B/ExpressPhysics/FEVT/Express-v1/000/366/451/00000/c6c70f87-561d-4e59-aab2-d81c3dbcd25c.root',
                                  '/store/express/Run2023B/ExpressPhysics/FEVT/Express-v1/000/366/451/00001/39764d40-af12-4fe1-b363-73f9e95e7282.root',
                                  '/store/express/Run2023B/ExpressPhysics/FEVT/Express-v1/000/366/451/00000/84e2c984-fcd3-499a-9d4b-64984cc93f23.root',
                                  '/store/express/Run2023B/ExpressPhysics/FEVT/Express-v1/000/366/451/00000/0929f055-eaac-4cb3-9db0-9f076c0706ca.root',
                                  '/store/express/Run2023B/ExpressPhysics/FEVT/Express-v1/000/366/451/00001/1f35a50e-1e16-42cd-894e-231349a394b3.root',
                                  '/store/express/Run2023B/ExpressPhysics/FEVT/Express-v1/000/366/451/00000/f9a97767-4f79-46e5-9804-7260fa1eaa14.root',
                                  '/store/express/Run2023B/ExpressPhysics/FEVT/Express-v1/000/366/451/00000/85338c01-0eb4-40ef-85f8-8a9c2c6ff255.root',
                                  '/store/express/Run2023B/ExpressPhysics/FEVT/Express-v1/000/366/451/00000/54430838-22cc-4b78-a48d-1c37e30ee8c4.root',
                                  '/store/express/Run2023B/ExpressPhysics/FEVT/Express-v1/000/366/451/00000/4d531663-bfa7-498a-8ed0-3c2b3563c8b7.root',
                                  '/store/express/Run2023B/ExpressPhysics/FEVT/Express-v1/000/366/451/00000/7bc4aa3a-b667-44dc-acb5-c4e59ad7908e.root',
                                  '/store/express/Run2023B/ExpressPhysics/FEVT/Express-v1/000/366/451/00000/9d090f31-16ad-44af-992b-4dc9a17fb6de.root',
                                  '/store/express/Run2023B/ExpressPhysics/FEVT/Express-v1/000/366/451/00001/6793a227-b7ca-4df9-a2b4-b5b0134cfedd.root',
                                  '/store/express/Run2023B/ExpressPhysics/FEVT/Express-v1/000/366/451/00001/9254b16e-4342-4f81-b74f-479ae22641e5.root',
                                  '/store/express/Run2023B/ExpressPhysics/FEVT/Express-v1/000/366/451/00000/0a8e7b89-fa2b-41f8-86b9-e2e7aa27f3f4.root',
                                  '/store/express/Run2023B/ExpressPhysics/FEVT/Express-v1/000/366/451/00001/9aa7cc50-04d6-47bb-aa90-8874dee7990d.root',
                                  '/store/express/Run2023B/ExpressPhysics/FEVT/Express-v1/000/366/451/00000/641e6934-5dfb-4c94-9bb7-793ff458c38f.root',
                                  '/store/express/Run2023B/ExpressPhysics/FEVT/Express-v1/000/366/451/00001/d14c20a0-5913-4b4d-8502-45e154066472.root',
                                  '/store/express/Run2023B/ExpressPhysics/FEVT/Express-v1/000/366/451/00000/68cd0355-3703-4fb5-9482-8c34c64fe81c.root',
                                  '/store/express/Run2023B/ExpressPhysics/FEVT/Express-v1/000/366/451/00000/65520c40-9f41-4677-a4b1-22cd5b4c3f2d.root',
                                  '/store/express/Run2023B/ExpressPhysics/FEVT/Express-v1/000/366/451/00000/eba4bf14-4a56-4c76-b832-4b19a76aad5b.root',
                                  '/store/express/Run2023B/ExpressPhysics/FEVT/Express-v1/000/366/451/00000/eab6c529-7757-4a76-bfa4-3eeeb2a0e399.root',
                                  '/store/express/Run2023B/ExpressPhysics/FEVT/Express-v1/000/366/451/00000/73a6a9f8-96cb-43af-a217-2941bc6b256e.root',
                                  '/store/express/Run2023B/ExpressPhysics/FEVT/Express-v1/000/366/451/00000/43e6721d-0c8d-4920-abd5-e566c2755104.root')

process.source = cms.Source ("PoolSource",
                             fileNames = readFiles,
                             ### As we are testing with FEVT, we don't want any unpacked collection
                             ### This makes the tests slightly more realistic (live production uses streamer files
                             inputCommands = cms.untracked.vstring(
                                 'drop *',
                                 'keep FEDRawDataCollection_rawDataCollector_*_*',
                                 'keep GlobalObjectMapRecord_hltGtStage2ObjectMap_*_*',
                                 'keep edmTriggerResults_TriggerResults_*_*'
                             ),
                             dropDescendantsOfDroppedBranches = cms.untracked.bool(True))

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(50))
process.options.numberOfThreads = 8

process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag as gtCustomise
process.GlobalTag = gtCustomise(process.GlobalTag, 'auto:run3_data', '')

#--------------------------------------------------------
# Swap offline <-> online BeamSpot as in Express and HLT
import RecoVertex.BeamSpotProducer.onlineBeamSpotESProducer_cfi as _mod
process.BeamSpotESProducer = _mod.onlineBeamSpotESProducer.clone()
import RecoVertex.BeamSpotProducer.BeamSpotOnline_cfi
process.offlineBeamSpot = RecoVertex.BeamSpotProducer.BeamSpotOnline_cfi.onlineBeamSpotProducer.clone()

#----------------
# Setup tracking
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
process.load('Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff')
process.load("Configuration.StandardSequences.RawToDigi_Data_cff")
process.load("RecoLocalTracker.Configuration.RecoLocalTracker_cff")
process.load("TrackingTools.TransientTrack.TransientTrackBuilder_cfi")
from RecoPixelVertexing.PixelLowPtUtilities.siPixelClusterShapeCache_cfi import *
process.siPixelClusterShapeCachePreSplitting = siPixelClusterShapeCache.clone(
  src = 'siPixelClustersPreSplitting'
)
process.load("RecoLocalTracker.SiPixelRecHits.PixelCPEGeneric_cfi")

process.pixelTracksCutClassifier = cms.EDProducer( "TrackCutClassifier",
    src = cms.InputTag( "pixelTracks" ),
    beamspot = cms.InputTag( "offlineBeamSpot" ),
    vertices = cms.InputTag( "" ),
    qualityCuts = cms.vdouble( -0.7, 0.1, 0.7 ),
    mva = cms.PSet(
      minPixelHits = cms.vint32( 0, 3, 3 ),
      maxDzWrtBS = cms.vdouble( 3.40282346639E38, 3.40282346639E38, 60.0 ),
      dr_par = cms.PSet(
        d0err = cms.vdouble( 0.003, 0.003, 3.40282346639E38 ),
        dr_par2 = cms.vdouble( 0.3, 0.3, 3.40282346639E38 ),
        dr_par1 = cms.vdouble( 0.4, 0.4, 3.40282346639E38 ),
        dr_exp = cms.vint32( 4, 4, 4 ),
        d0err_par = cms.vdouble( 0.001, 0.001, 3.40282346639E38 )
      ),
      maxLostLayers = cms.vint32( 99, 99, 99 ),
      min3DLayers = cms.vint32( 0, 2, 3 ),
      dz_par = cms.PSet(
        dz_par1 = cms.vdouble( 0.4, 0.4, 3.40282346639E38 ),
        dz_par2 = cms.vdouble( 0.35, 0.35, 3.40282346639E38 ),
        dz_exp = cms.vint32( 4, 4, 4 )
      ),
      minNVtxTrk = cms.int32( 3 ),
      maxDz = cms.vdouble( 3.40282346639E38, 3.40282346639E38, 3.40282346639E38 ),
      minNdof = cms.vdouble( 1.0E-5, 1.0E-5, 1.0E-5 ),
      maxChi2 = cms.vdouble( 9999., 9999., 30.0 ),
      maxDr = cms.vdouble( 99., 99., 1. ),
      minLayers = cms.vint32( 0, 2, 3 )
    ),
    ignoreVertices = cms.bool( True ),
)

#
process.pixelTracksHP = cms.EDProducer( "TrackCollectionFilterCloner",
    minQuality = cms.string( "highPurity" ),
    copyExtras = cms.untracked.bool( True ),
    copyTrajectories = cms.untracked.bool( False ),
    originalSource = cms.InputTag( "pixelTracks" ),
    originalQualVals = cms.InputTag( 'pixelTracksCutClassifier','QualityMasks' ),
    originalMVAVals = cms.InputTag( 'pixelTracksCutClassifier','MVAValues' )
)


process.tracks2monitor = cms.EDFilter('TrackSelector',
    src = cms.InputTag('pixelTracks'),
    cut = cms.string("")
)
process.tracks2monitor.src = 'pixelTracksHP'
process.tracks2monitor.cut = 'pt > 1 & abs(eta) < 2.4' 

#process.selectedPixelTracksMonitorSequence = cms.Sequence(
#    process.pixelTracksCutClassifier
#  + process.pixelTracksHP
#  + process.tracks2monitor
#  + process.selectedPixelTracksMonitor
#)

# Digitisation: produce the TCDS digis containing BST record
from EventFilter.OnlineMetaDataRawToDigi.tcdsRawToDigi_cfi import *
process.tcdsDigis = tcdsRawToDigi.clone()

rawDataInputTag = "rawDataCollector"
process.castorDigis.InputLabel           = rawDataInputTag
process.csctfDigis.producer              = rawDataInputTag 
process.dttfDigis.DTTF_FED_Source        = rawDataInputTag
process.ecalDigis.cpu.InputLabel         = rawDataInputTag
process.ecalPreshowerDigis.sourceTag     = rawDataInputTag
process.gctDigis.inputLabel              = rawDataInputTag
process.gtDigis.DaqGtInputTag            = rawDataInputTag
process.hcalDigis.InputLabel             = rawDataInputTag
process.muonCSCDigis.InputObjects        = rawDataInputTag
process.muonDTDigis.inputLabel           = rawDataInputTag
process.muonRPCDigis.InputLabel          = rawDataInputTag
process.scalersRawToDigi.scalersInputTag = rawDataInputTag
process.siPixelDigis.cpu.InputLabel      = rawDataInputTag
process.siStripDigis.ProductLabel        = rawDataInputTag
process.tcdsDigis.InputLabel             = rawDataInputTag

process.load("RecoVertex.BeamSpotProducer.BeamSpot_cfi")

#----------------------------
# Pixel tracks/vertices reco
process.load("RecoPixelVertexing.Configuration.RecoPixelVertexing_cff")
from RecoVertex.PrimaryVertexProducer.OfflinePixel3DPrimaryVertices_cfi import *
process.pixelVertices = pixelVertices.clone(
  TkFilterParameters = dict( minPt = process.pixelTracksTrackingRegions.RegionPSet.ptMin)
)
#process.pixelTracksTrackingRegions.RegionPSet.ptMin = 0.1       # used in PilotBeam 2021, but not ok for standard collisions
process.pixelTracksTrackingRegions.RegionPSet.originRadius = 0.4 # used in PilotBeam 2021, to be checked again for standard collisions
# The following parameters were used in 2018 HI:
#process.pixelTracksTrackingRegions.RegionPSet.originHalfLength = 12
#process.pixelTracksTrackingRegions.RegionPSet.originXPos =  0.08
#process.pixelTracksTrackingRegions.RegionPSet.originYPos = -0.03
#process.pixelTracksTrackingRegions.RegionPSet.originZPos = 0.

process.tracking_FirstStep = cms.Sequence(
    process.siPixelDigis 
    * process.siStripDigis
    * process.striptrackerlocalreco
    * process.offlineBeamSpot
    * process.siPixelClustersPreSplitting
    * process.siPixelRecHitsPreSplitting
    * process.siPixelClusterShapeCachePreSplitting
    * process.recopixelvertexing)

#--------
# Do no run on events with pixel or strip with HV off

process.stripTrackerHVOn = cms.EDFilter( "DetectorStateFilter",
    DCSRecordLabel = cms.untracked.InputTag( "onlineMetaDataDigis" ),
    DcsStatusLabel = cms.untracked.InputTag( "scalersRawToDigi" ),
    DebugOn = cms.untracked.bool( False ),
    DetectorType = cms.untracked.string( "sistrip" )
)

process.pixelTrackerHVOn = cms.EDFilter( "DetectorStateFilter",
    DCSRecordLabel = cms.untracked.InputTag( "onlineMetaDataDigis" ),
    DcsStatusLabel = cms.untracked.InputTag( "scalersRawToDigi" ),
    DebugOn = cms.untracked.bool( False ),
    DetectorType = cms.untracked.string( "pixel" )
)

process.p = cms.Path(process.scalersRawToDigi
                     * process.tcdsDigis
                     * process.onlineMetaDataDigis
                     * process.pixelTrackerHVOn
                     * process.stripTrackerHVOn
                     * process.tracking_FirstStep)

process.out = cms.OutputModule("PoolOutputModule",
                               fileName = cms.untracked.string('testSlimmingTest.root'),
                               outputCommands = cms.untracked.vstring(
                                   'keep *',
                                   #'keep *_trackOfThingsProducerB_*_*',
                                   #'keep *_trackOfThingsProducerI_*_*',
                                   #'keep *_thinningThingProducerB_*_*',
                                   #'keep *_thinningThingProducerCI_*_*',
                               ))

process.ep = cms.EndPath(process.out)

print("Global Tag used:", process.GlobalTag.globaltag.value())
print("Final Source settings:", process.source)
