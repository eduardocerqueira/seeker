#date: 2021-11-11T17:03:54Z
#url: https://api.github.com/gists/3e37dee87fb4d0713260046e1b34f15d
#owner: https://api.github.com/users/colizz

def concurrency_check(fragment,pi,cmssw_version):
    conc_check = 0
    conc_check_lhe = 0
    error_conc = 0
    fragment = fragment.replace(" ","").replace("\"","'")#
    if cmssw_version >= int('10_60_28'.replace('_','')):
        # first check if the code has correctly implemented concurrent features. Mark conc_check_lhe (LHE step) or conc_check (GEN step) as True if features are found
        if "ExternalLHEProducer" in fragment and "generateConcurrently=cms.untracked.bool(True)" in fragment:
            if "Herwig7GeneratorFilter" not in fragment: 
                conc_check_lhe = 1
            else:
                if "postGenerationCommand=cms.untracked.vstring('mergeLHE.py','-i','thread*/cmsgrid_final.lhe','-o','cmsgrid_final.lhe')" in fragment: 
                    conc_check_lhe = 1# 
        elif "ExternalLHEProducer" not in fragment:#
            conc_check_lhe = 1#
        if "ExternalDecays" not in fragment and "Pythia8ConcurrentHadronizerFilter" in fragment: 
            conc_check = 1
        if "Pythia8ConcurrentGeneratorFilter" in fragment and "ExternalDecays" not in fragment and "RandomizedParameters" not in fragment: 
            conc_check = 1
        if "ExternalLHEProducer" not in fragment and "_generator=cms.EDFilter" in fragment and "fromGeneratorInterface.Core.ExternalGeneratorFilterimportExternalGeneratorFilter" in fragment and "generator=ExternalGeneratorFilter(_generator" in fragment:
            if "Pythia8GeneratorFilter" in fragment and "tauola" not in fragment.lower(): 
                conc_check = 1
            if "Pythia8GeneratorFilter" in fragment and "tauola" in fragment.lower() and "_external_process_components_=cms.vstring('HepPDTESSource')" in fragment:
                conc_check = 1
            if "AMPTGeneratorFilter" in fragment or "HydjetGeneratorFilter" in fragment or "PyquenGeneratorFilter" in fragment or "Pythia6GeneratorFilter": 
                conc_check = 1
            if "ReggeGribovPartonMCGeneratorFilter" in fragment or "SherpaGeneratorFilter" in fragment: 
                conc_check = 1
            if "Herwig7GeneratorFilter" in fragment and "wmlhegen" not in pi.lower() and "plhegen" not in pi.lower(): 
                conc_check = 1 
        print("Concurrency check LHE = ",conc_check_lhe,"  Concurrency check GEN = ",conc_check)
        if conc_check_lhe and conc_check:
            print("\n The request will be generated concurrently\n")
        else:
            # then if not both the LHE and GEN step turns on concurrent features, we check if for some cases it is ok not to have concurrency
            if "Pythia8HadronizerFilter" in fragment and ("evtgen" in fragment.lower() or "tauola" in fragment.lower() or "photos" in fragment.lower()):
                print("\n Pythia8HadronizerFilter with EvtGen, Tauola, or Photos can not be made concurrently.\n")
            elif "Herwig7GeneratorFilter" in fragment and ("wmlhegen" in pi.lower() or "plhegen" in pi.lower()): 
                print("Herwig7GeneratorFilter in the wmLHEGEN or pLHEGEN campaign cannot run concurrently.")
            elif "Pythia8GeneratorFilter" in fragment and "randomizedparameters" in fragment.lower():
                print("Pythia8GeneratorFilter with RandomizedParameter scan cannot run concurrently")
            # for other cases, it is either concurrent generation parameters are missing or wrong
            else:
                print("[ERROR] Concurrent generation parameters missing or wrong. Please see https://twiki.cern.ch/twiki/bin/view/CMSPublic/WorkBookGenMultithread")
                error_conc = 1
    else:
        if "concurrent" in fragment.lower():
            print("[ERROR] Concurrent generation is not supported for versions < CMSSW_10_6_28")
            error_conc = 1
    return conc_check_lhe and conc_check, error_conc