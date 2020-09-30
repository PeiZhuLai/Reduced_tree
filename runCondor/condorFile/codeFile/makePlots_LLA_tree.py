#! /usr/bin/env python

import argparse
parser = argparse.ArgumentParser(description="A simple ttree plotter")
parser.add_argument("-i", "--inputfiles", dest="inputfiles", default=["Sync_1031_2018_ttH_v2.root"], nargs='*', help="List of input ggNtuplizer files")
parser.add_argument("-o", "--outputfile", dest="outputfile", default="plots.root", help="Output file containing plots")
parser.add_argument("-m", "--maxevents", dest="maxevents", type=int, default=-1, help="Maximum number events to loop over")
parser.add_argument("-t", "--ttree", dest="ttree", default="Ana/passedEvents", help="TTree Name")
parser.add_argument("-xs", "--cross_section", dest="cross_section", default="1.0", help="the cross section of samples")
parser.add_argument("-L", "--Lumi", dest="Lumi", default="35.9", help="the luminosities to normalized")
parser.add_argument("-N", "--NEvent", dest="NEvent", default="0", help="number of events")
args = parser.parse_args()

import numpy as np
import ROOT
import os

###########################
from deltaR import *
from array import array
#from calculateMVA import bookMVA, calMVA

from xgboost import XGBClassifier
import pickle
# Read in model saved from previous running of BDT
mass = 'M5'

if mass == 'M1':
    BDT_filename="/publicfs/cms/user/wangzebing/ALP/Analysis_code/MVA/weight/model_ALP_M1.pkl"
    mvaCut = 0.4267
elif mass == 'M5':
    BDT_filename="/publicfs/cms/user/wangzebing/ALP/Analysis_code/MVA/weight/model_ALP_M5.pkl"
    mvaCut = 0.4682
elif mass == 'M15':
    BDT_filename="/publicfs/cms/user/wangzebing/ALP/Analysis_code/MVA/weight/model_ALP_M15.pkl"
    mvaCut = 0.4893
else:
    BDT_filename="/publicfs/cms/user/wangzebing/ALP/Analysis_code/MVA/weight/model_ALP_M30.pkl"
    mvaCut = 0.4447


# load the model from disk
model = pickle.load(open(BDT_filename, 'rb'))

###########################
if os.path.isfile('~/.rootlogon.C'): ROOT.gROOT.Macro(os.path.expanduser('~/.rootlogon.C'))
ROOT.gROOT.SetBatch()
ROOT.gROOT.SetStyle("Plain")
ROOT.gStyle.SetOptStat(000000)
ROOT.gStyle.SetPalette(ROOT.kRainBow)
ROOT.gStyle.UseCurrentStyle()

sw = ROOT.TStopwatch()
sw.Start()

# Input ggNtuple
tchain = ROOT.TChain(args.ttree)
for filename in args.inputfiles: tchain.Add(filename)
print 'Total number of events: ' + str(tchain.GetEntries())

# Event weights
isMC = True
if 'Run2018' in filename:
    isMC = False
elif 'Run2017' in filename:
    isMC = False
elif 'Run2016' in filename:
    isMC = False
else:
    isMC = True

# get nEvents
nEvents = 0

for filename in args.inputfiles:

    files = ROOT.TFile(filename)
    n_his = files.Ana.Get('sumWeights')
    nEvents = nEvents + n_his.GetBinContent(1)


if isMC:
    cross_section = float(args.cross_section)
    lumi = float(args.Lumi)

    if int(args.NEvent):
        weight = cross_section * lumi * 1000.0 / float(args.NEvent)
    else:
        weight = cross_section * lumi * 1000.0 / nEvents
else:
    cross_section = 1.0
    weight = 1.0



print 'events weight: '+str(weight)
print 'events : '+str(tchain.GetEntries())


# Output file and any histograms we want
file_out = ROOT.TFile(args.outputfile, 'recreate')

nEvents_total = ROOT.TH1D('nEvents_total', 'nEvents_total', 2, 0, 2)
nEvents_total.SetBinContent(1,nEvents)
h_weight = ROOT.TH1D('Events_weight', 'Events_weight', 2, 0, 2)
h_weight.SetBinContent(1,weight)
h_cross_section = ROOT.TH1D('cross_section', 'cross_section', 2, 0, 2)
h_cross_section.SetBinContent(1,cross_section)

# pass triger
h_n = ROOT.TH1D('nEvents_ntuple', 'nEvents_ntuple', 2, 0, 2)
h_n_trig = ROOT.TH1D('nEvents_trig', 'nEvents_trig', 2, 0, 2)



npho = ROOT.TH1D('npho', 'npho', 10, 0., 10)
nlep = ROOT.TH1D('nlep', 'nlep', 10, 0., 10)

################################################################################################
Z_e_nocut = ROOT.TH1D('Z_e_nocut', 'Z_e_nocut', 100, 0, 500)
Z_mu_nocut = ROOT.TH1D('Z_mu_nocut', 'Z_mu_nocut', 100, 0, 500)

l1_id_lIso = ROOT.TH1D('l1_id_lIso', 'l1_id_lIso', 40, -20., 20.)
Z_e_lIso = ROOT.TH1D('Z_e_lIso', 'Z_e_lIso', 100, 0, 500)
Z_mu_lIso = ROOT.TH1D('Z_mu_lIso', 'Z_mu_lIso', 100, 0, 500)

Z_e_lIso_lTight = ROOT.TH1D('Z_e_lIso_lTight', 'Z_e_lIso_lTight', 100, 0, 500)
Z_mu_lIso_lTight = ROOT.TH1D('Z_mu_lIso_lTight', 'Z_mu_lIso_lTight', 100, 0, 500)

Z_50 = ROOT.TH1D('Z_50', 'Z_50', 100, 0, 500)

################################################################################################

################################################################################################




h_n.SetStats(1)
h_n_trig.SetStats(1)

Z_e_nocut.SetStats(1)
Z_mu_nocut.SetStats(1)
Z_e_lIso.SetStats(1)
Z_mu_lIso.SetStats(1)
Z_e_lIso_lTight.SetStats(1)
Z_mu_lIso_lTight.SetStats(1)

Z_50.SetStats(1)

################################################################################################

# photon cut tree
Run = array('l',[0])
LumiSect = array('l',[0])
Event = array('l',[0])
factor = array('f',[0.])

# photon var
pho1eta = array('f',[0.])
pho1Pt = array('f',[0.])
pho1R9 = array('f',[0.])
pho1IetaIeta = array('f',[0.])
pho1IetaIeta55 = array('f',[0.])
pho1HOE = array('f',[0.])
pho1CIso = array('f',[0.])
pho1NIso = array('f',[0.])
pho1PIso = array('f',[0.])
pho1PIso_noCorr = array('f',[0.])

pho2eta = array('f',[0.])
pho2Pt = array('f',[0.])
pho2R9 = array('f',[0.])
pho2IetaIeta = array('f',[0.])
pho2IetaIeta55 = array('f',[0.])
pho2HOE = array('f',[0.])
pho2CIso = array('f',[0.])
pho2NIso = array('f',[0.])
pho2PIso = array('f',[0.])
pho2PIso_noCorr = array('f',[0.])

passEleVeto = array('f',[0.])
passIeIe = array('f',[0.])
passHOverE = array('f',[0.])
passChaHadIso = array('f',[0.])
passNeuHadIso = array('f',[0.])
passPhoIso = array('f',[0.])
passdR_gl = array('f',[0.])
passdR_gg = array('f',[0.])
passH_m = array('f',[0.])


# photon cut
H_twopho = array('f',[-1.])

event_genWeight = array('f',[-1.])
event_pileupWeight = array('f',[-1.])
event_dataMCWeight = array('f',[-1.])

Z_Ceta = array('f',[-1.])
H_Ceta = array('f',[-1.])
ALP_Ceta = array('f',[-1.])

Z_pho_veto = array('f',[-1.])
H_pho_veto = array('f',[-1.])
ALP_pho_veto = array('f',[-1.])

Z_CIso = array('f',[-1.])
H_CIso = array('f',[-1.])
ALP_CIso = array('f',[-1.])

Z_NIso = array('f',[-1.])
H_NIso = array('f',[-1.])
ALP_NIso = array('f',[-1.])

Z_IeIe = array('f',[-1.])
H_IeIe = array('f',[-1.])
ALP_IeIe = array('f',[-1.])

Z_HOE = array('f',[-1.])
H_HOE = array('f',[-1.])
ALP_HOE = array('f',[-1.])

Z_PIso = array('f',[-1.])
H_PIso = array('f',[-1.])
ALP_PIso = array('f',[-1.])

dR_g1l1 = array('f',[-1.])
dR_g1l2 = array('f',[-1.])
dR_g2l1 = array('f',[-1.])
dR_g2l2 = array('f',[-1.])
var_dPhi_g1Z_beforMVA = array('f',[-1.])
var_dR_g1Z_beforMVA = array('f',[-1.])

Z_pho_veto_mva = array('f',[-1.])
H_pho_veto_mva = array('f',[-1.])
ALP_pho_veto_mva = array('f',[-1.])

dR_g1l1_mva = array('f',[-1.])
dR_g1l2_mva = array('f',[-1.])
dR_g2l1_mva = array('f',[-1.])
dR_g2l2_mva = array('f',[-1.])
var_dR_Za = array('f',[-1.])
var_dR_g1g2 = array('f',[-1.])
var_dR_g1Z = array('f',[-1.])
var_dEta_g1Z = array('f',[-1.])
var_dPhi_g1Z = array('f',[-1.])
var_PtaOverMa = array('f',[-1.])
var_PtaOverMh = array('f',[-1.])
var_Pta = array('f',[-1.])
var_MhMa = array('f',[-1.])
var_MhMZ = array('f',[-1.])

ALP_calculatedPhotonIso = array('f',[-1.])



Z_dR = array('f',[-1.])
H_dR = array('f',[-1.])
ALP_dR = array('f',[-1.])

Z_dR_pho = array('f',[-1.])
H_dR_pho = array('f',[-1.])
ALP_dR_pho = array('f',[-1.])

Z_dR_pho_Cmh = array('f',[-1.])
H_dR_pho_Cmh = array('f',[-1.])
ALP_dR_pho_Cmh = array('f',[-1.])


################################################################################################

# Tree
l1_pt = array('f',[0.])
l1_eta = array('f',[0.])
l1_phi = array('f',[0.])
l1_id = array('i',[0])



l2_pt = array('f',[0.])
l2_eta = array('f',[0.])
l2_phi = array('f',[0.])
l2_id = array('i',[0])

pho1_pt = array('f',[0.])
pho1_eta = array('f',[0.])
pho1_phi = array('f',[0.])
pho1_mva = array('f',[0.])
pho1_matche_PdgId = array('f',[0.])
pho1_matche_MomId = array('f',[0.])
pho1_matche_MomMomId = array('f',[0.])
pho1_matchedR = array('f',[0.])

pho2_pt = array('f',[0.])
pho2_eta = array('f',[0.])
pho2_phi = array('f',[0.])
pho2_mva = array('f',[0.])
pho2_matche_PdgId = array('f',[0.])
pho2_matche_MomId = array('f',[0.])
pho2_matche_MomMomId = array('f',[0.])
pho2_matchedR = array('f',[0.])

Z_m = array('f',[0.])
H_m = array('f',[0.])
ALP_m = array('f',[0.])

H_pt = array('f',[0.])
dR_pho = array('f',[0.])

event_weight = array('f',[0.])

passedEvents = ROOT.TTree("passedEvents","passedEvents")

################################################################################################
passedEvents.Branch("Run",Run,"Run/L")
passedEvents.Branch("LumiSect",LumiSect,"LumiSect/L")
passedEvents.Branch("Event",Event,"Event/L")
passedEvents.Branch("factor",factor,"factor/F")

passedEvents.Branch("pho1eta",pho1eta,"pho1eta/F")
passedEvents.Branch("pho1Pt",pho1Pt,"pho1Pt/F")
passedEvents.Branch("pho1R9",pho1R9,"pho1R9/F")
passedEvents.Branch("pho1IetaIeta",pho1IetaIeta,"pho1IetaIeta/F")
passedEvents.Branch("pho1IetaIeta55",pho1IetaIeta55,"pho1IetaIeta55/F")
passedEvents.Branch("pho1HOE",pho1HOE,"pho1HOE/F")
passedEvents.Branch("pho1CIso",pho1CIso,"pho1CIso/F")
passedEvents.Branch("pho1NIso",pho1NIso,"pho1NIso/F")
passedEvents.Branch("pho1PIso",pho1PIso,"pho1PIso/F")
passedEvents.Branch("pho1PIso_noCorr",pho1PIso_noCorr,"pho1PIso_noCorr/F")


passedEvents.Branch("pho2eta",pho2eta,"pho2eta/F")
passedEvents.Branch("pho2Pt",pho2Pt,"pho2Pt/F")
passedEvents.Branch("pho2R9",pho2R9,"pho2R9/F")
passedEvents.Branch("pho2IetaIeta",pho2IetaIeta,"pho2IetaIeta/F")
passedEvents.Branch("pho2IetaIeta55",pho2IetaIeta55,"pho2IetaIeta55/F")
passedEvents.Branch("pho2HOE",pho2HOE,"pho2HOE/F")
passedEvents.Branch("pho2CIso",pho2CIso,"pho2CIso/F")
passedEvents.Branch("pho2NIso",pho2NIso,"pho2NIso/F")
passedEvents.Branch("pho2PIso",pho2PIso,"pho2PIso/F")
passedEvents.Branch("pho2PIso_noCorr",pho2PIso_noCorr,"pho2PIso_noCorr/F")


passedEvents.Branch("passIeIe",passIeIe,"passIeIe/F")
passedEvents.Branch("passHOverE",passHOverE,"passHOverE/F")
passedEvents.Branch("passChaHadIso",passChaHadIso,"passChaHadIso/F")
passedEvents.Branch("passNeuHadIso",passNeuHadIso,"passNeuHadIso/F")
passedEvents.Branch("passPhoIso",passPhoIso,"passPhoIso/F")
passedEvents.Branch("passdR_gl",passdR_gl,"passdR_gl/F")
passedEvents.Branch("passdR_gg",passdR_gg,"passdR_gg/F")
passedEvents.Branch("passH_m",passH_m,"passH_m/F")


passedEvents.Branch("H_twopho",H_twopho,"H_twopho/F")

passedEvents.Branch("event_genWeight",event_genWeight,"event_genWeight/F")
passedEvents.Branch("event_pileupWeight",event_pileupWeight,"event_pileupWeight/F")
passedEvents.Branch("event_dataMCWeight",event_dataMCWeight,"event_dataMCWeight/F")

passedEvents.Branch("Z_Ceta",Z_Ceta,"Z_Ceta/F")
passedEvents.Branch("H_Ceta",H_Ceta,"H_Ceta/F")
passedEvents.Branch("ALP_Ceta",ALP_Ceta,"ALP_Ceta/F")

passedEvents.Branch("Z_pho_veto",Z_pho_veto,"Z_pho_veto/F")
passedEvents.Branch("H_pho_veto",H_pho_veto,"H_pho_veto/F")
passedEvents.Branch("ALP_pho_veto",ALP_pho_veto,"ALP_pho_veto/F")

passedEvents.Branch("Z_CIso",Z_CIso,"Z_CIso/F")
passedEvents.Branch("H_CIso",H_CIso,"H_CIso/F")
passedEvents.Branch("ALP_CIso",ALP_CIso,"ALP_CIso/F")

passedEvents.Branch("Z_NIso",Z_NIso,"Z_NIso/F")
passedEvents.Branch("H_NIso",H_NIso,"H_NIso/F")
passedEvents.Branch("ALP_NIso",ALP_NIso,"ALP_NIso/F")

passedEvents.Branch("Z_IeIe",Z_IeIe,"Z_IeIe/F")
passedEvents.Branch("H_IeIe",H_IeIe,"H_IeIe/F")
passedEvents.Branch("ALP_IeIe",ALP_IeIe,"ALP_IeIe/F")

passedEvents.Branch("Z_HOE",Z_HOE,"Z_HOE/F")
passedEvents.Branch("H_HOE",H_HOE,"H_HOE/F")
passedEvents.Branch("ALP_HOE",ALP_HOE,"ALP_HOE/F")

passedEvents.Branch("Z_PIso",Z_PIso,"Z_PIso/F")
passedEvents.Branch("H_PIso",H_PIso,"H_PIso/F")
passedEvents.Branch("ALP_PIso",ALP_PIso,"ALP_PIso/F")


passedEvents.Branch("dR_g1l1",dR_g1l1,"dR_g1l1/F")
passedEvents.Branch("dR_g1l2",dR_g1l2,"dR_g1l2/F")
passedEvents.Branch("dR_g2l1",dR_g2l1,"dR_g2l1/F")
passedEvents.Branch("dR_g2l2",dR_g2l2,"dR_g2l2/F")


passedEvents.Branch("Z_pho_veto_mva",Z_pho_veto_mva,"Z_pho_veto_mva/F")
passedEvents.Branch("H_pho_veto_mva",H_pho_veto_mva,"H_pho_veto_mva/F")
passedEvents.Branch("ALP_pho_veto_mva",ALP_pho_veto_mva,"ALP_pho_veto_mva/F")
passedEvents.Branch("var_dR_g1Z_beforMVA",var_dR_g1Z_beforMVA,"var_dR_g1Z_beforMVA/F")
passedEvents.Branch("var_dPhi_g1Z_beforMVA",var_dPhi_g1Z_beforMVA,"var_dPhi_g1Z_beforMVA/F")

passedEvents.Branch("dR_g1l1_mva",dR_g1l1_mva,"dR_g1l1_mva/F")
passedEvents.Branch("dR_g1l2_mva",dR_g1l2_mva,"dR_g1l2_mva/F")
passedEvents.Branch("dR_g2l1_mva",dR_g2l1_mva,"dR_g2l1_mva/F")
passedEvents.Branch("dR_g2l2_mva",dR_g2l2_mva,"dR_g2l2_mva/F")

passedEvents.Branch("var_dR_Za",var_dR_Za,"var_dR_Za/F")
passedEvents.Branch("var_dR_g1g2",var_dR_g1g2,"var_dR_g1g2/F")
passedEvents.Branch("var_dR_g1Z",var_dR_g1Z,"var_dR_g1Z/F")
passedEvents.Branch("var_dEta_g1Z",var_dEta_g1Z,"var_dEta_g1Z/F")
passedEvents.Branch("var_dPhi_g1Z",var_dPhi_g1Z,"var_dPhi_g1Z/F")
passedEvents.Branch("var_PtaOverMa",var_PtaOverMa,"var_PtaOverMa/F")
passedEvents.Branch("var_PtaOverMh",var_PtaOverMh,"var_PtaOverMh/F")
passedEvents.Branch("var_Pta",var_Pta,"var_Pta/F")
passedEvents.Branch("var_MhMa",var_MhMa,"var_MhMa/F")
passedEvents.Branch("var_MhMZ",var_MhMZ,"var_MhMZ/F")

passedEvents.Branch("ALP_calculatedPhotonIso",ALP_calculatedPhotonIso,"ALP_calculatedPhotonIso/F")



passedEvents.Branch("Z_dR",Z_dR,"Z_dR/F")
passedEvents.Branch("H_dR",H_dR,"H_dR/F")
passedEvents.Branch("ALP_dR",ALP_dR,"ALP_dR/F")

passedEvents.Branch("Z_dR_pho",Z_dR_pho,"Z_dR_pho/F")
passedEvents.Branch("H_dR_pho",H_dR_pho,"H_dR_pho/F")
passedEvents.Branch("ALP_dR_pho",ALP_dR_pho,"ALP_dR_pho/F")

passedEvents.Branch("Z_dR_pho_Cmh",Z_dR_pho_Cmh,"Z_dR_pho_Cmh/F")
passedEvents.Branch("H_dR_pho_Cmh",H_dR_pho_Cmh,"H_dR_pho_Cmh/F")
passedEvents.Branch("ALP_dR_pho_Cmh",ALP_dR_pho_Cmh,"ALP_dR_pho_Cmh/F")


################################################################################################


passedEvents.Branch("l1_pt",l1_pt,"l1_pt/F")
passedEvents.Branch("l1_eta",l1_eta,"l1_eta/F")
passedEvents.Branch("l1_phi",l1_phi,"l1_phi/F")
passedEvents.Branch("l1_id",l1_id,"l1_id/I")

passedEvents.Branch("l2_pt",l2_pt,"l2_pt/F")
passedEvents.Branch("l2_eta",l2_eta,"l2_eta/F")
passedEvents.Branch("l2_phi",l2_phi,"l2_phi/F")
passedEvents.Branch("l2_id",l2_id,"l2_id/I")

passedEvents.Branch("pho1_pt",pho1_pt,"pho1_pt/F")
passedEvents.Branch("pho1_eta",pho1_eta,"pho1_eta/F")
passedEvents.Branch("pho1_phi",pho1_phi,"pho1_phi/F")
passedEvents.Branch("pho1_mva",pho1_mva,"pho1_mva/F")
passedEvents.Branch("pho1_matche_PdgId",pho1_matche_PdgId,"pho1_matche_PdgId/F")
passedEvents.Branch("pho1_matche_MomId",pho1_matche_MomId,"pho1_matche_MomId/F")
passedEvents.Branch("pho1_matche_MomMomId",pho1_matche_MomMomId,"pho1_matche_MomMomId/F")
passedEvents.Branch("pho1_matchedR",pho1_matchedR,"pho1_matchedR/F")

passedEvents.Branch("pho2_pt",pho2_pt,"pho2_pt/F")
passedEvents.Branch("pho2_eta",pho2_eta,"pho2_eta/F")
passedEvents.Branch("pho2_phi",pho2_phi,"pho2_phi/F")
passedEvents.Branch("pho2_mva",pho2_mva,"pho2_mva/F")
passedEvents.Branch("pho2_matche_PdgId",pho2_matche_PdgId,"pho2_matche_PdgId/F")
passedEvents.Branch("pho2_matche_MomId",pho2_matche_MomId,"pho2_matche_MomId/F")
passedEvents.Branch("pho2_matche_MomMomId",pho2_matche_MomMomId,"pho2_matche_MomMomId/F")
passedEvents.Branch("pho2_matchedR",pho2_matchedR,"pho2_matchedR/F")

passedEvents.Branch("Z_m",Z_m,"Z_m/F")
passedEvents.Branch("H_m",H_m,"H_m/F")
passedEvents.Branch("ALP_m",ALP_m,"ALP_m/F")
passedEvents.Branch("H_pt",H_pt,"H_pt/F")
passedEvents.Branch("dR_pho",dR_pho,"dR_pho/F")


passedEvents.Branch("event_weight",event_weight,"event_weight/F")


################################################################ mathod ################################################################
cutBased = True
mvaBased = False

#reader = bookMVA()
########################################################################################################################################
#Loop over all the events in the input ntuple
for ievent,event in enumerate(tchain):#, start=650000):
    if ievent > args.maxevents and args.maxevents != -1: break
    #if ievent == 5000000: break
    if ievent % 10000 == 0: print 'Processing entry ' + str(ievent)


    # Loop over all the electrons in an event

    # pho parameters
    foundpho1 = False
    foundpho2 = False
    pho1_maxPt = 0.0
    pho2_maxPt = 0.0
    pho1_index = 0
    pho2_index = 0
    pho_passEleVeto = True
    pho_passIeIe = True
    pho_passHOverE = True
    pho_passChaHadIso = True
    pho_passNeuHadIso = True
    passedPhoIso = True

    mva_value = -999.0

    # initialise Z parameters
    Nlep = 0
    lep=0

    Zmass = 91.1876
    dZmass = 9999.0
    n_Zs = 0
    Z_pt = []
    Z_eta = []
    Z_phi = []
    Z_mass = []
    Z_index = 0
    Z_lepindex1 = []
    Z_lepindex2 = []
    foundZ = False
    lep_leadindex = [] # lepindex[0] for leading, lepindex[1] for subleading

    # pass trigger
################################################################################################
    h_n.Fill(event.passedTrig)
    if (not event.passedTrig): continue
    h_n_trig.Fill(event.passedTrig)

    # find all Z candidates
################################################################################################
    Nlep = event.lep_pt.size()
    nlep.Fill(Nlep)

    if Nlep!=2: continue

    for i in range(Nlep):

        for j in range(i+1,Nlep):

            if ((event.lep_id[i] + event.lep_id[j]) != 0): continue

            lifsr = ROOT.TLorentzVector()
            ljfsr = ROOT.TLorentzVector()
            lifsr.SetPtEtaPhiM(event.lepFSR_pt[i], event.lepFSR_eta[i], event.lepFSR_phi[i], event.lepFSR_mass[i])
            ljfsr.SetPtEtaPhiM(event.lepFSR_pt[j], event.lepFSR_eta[j], event.lepFSR_phi[j], event.lepFSR_mass[j])

            Z = ROOT.TLorentzVector()
            Z = (lifsr + ljfsr)

            if (Z.M()>0.0):
                n_Zs = n_Zs + 1
                Z_pt.append(Z.Pt())
                Z_eta.append(Z.Eta())
                Z_phi.append(Z.Phi())
                Z_mass.append(Z.M())
                Z_lepindex1.append(i)
                Z_lepindex2.append(j)

                foundZ = True
        # lep j
    # lep i

    if (not foundZ): continue

    # find Z
    for i in range(n_Zs):
        if (abs(Z_mass[i] - Zmass) <= dZmass):
            dZmass = abs(Z_mass[i] - Zmass)
            Z_index = i

    # find Z end

    if (event.lepFSR_pt[Z_lepindex1[Z_index]] >= event.lepFSR_pt[Z_lepindex2[Z_index]]):
        lep_leadindex.append(Z_lepindex1[Z_index])
        lep_leadindex.append(Z_lepindex2[Z_index])
    else:
        lep_leadindex.append(Z_lepindex2[Z_index])
        lep_leadindex.append(Z_lepindex1[Z_index])
################################################################################################



    l1_find = ROOT.TLorentzVector()
    l2_find = ROOT.TLorentzVector()
    Z_find = ROOT.TLorentzVector()

    l1_find.SetPtEtaPhiM(event.lepFSR_pt[lep_leadindex[0]], event.lepFSR_eta[lep_leadindex[0]], event.lepFSR_phi[lep_leadindex[0]], event.lepFSR_mass[lep_leadindex[0]])
    l2_find.SetPtEtaPhiM(event.lepFSR_pt[lep_leadindex[1]], event.lepFSR_eta[lep_leadindex[1]], event.lepFSR_phi[lep_leadindex[1]], event.lepFSR_mass[lep_leadindex[1]])

    Z_find = (l1_find + l2_find)

    if (abs(event.lep_id[lep_leadindex[0]]) == 11):
        Z_e_nocut.Fill(Z_find.M())
    if (abs(event.lep_id[lep_leadindex[0]]) == 13):
        Z_mu_nocut.Fill(Z_find.M())

        # Leptons Cuts
################################################################################################
    # pass lep isolation

    if (abs(event.lep_id[lep_leadindex[0]]) == 11):
        if (event.lep_RelIsoNoFSR[lep_leadindex[0]] > 0.35): continue
        if (event.lep_RelIsoNoFSR[lep_leadindex[1]] > 0.35): continue

        # pt Cut
        if (event.lepFSR_pt[lep_leadindex[0]] <= 25): continue
        if (event.lepFSR_pt[lep_leadindex[1]] <= 15): continue

    if (abs(event.lep_id[lep_leadindex[0]]) == 13):
        if (event.lep_RelIsoNoFSR[lep_leadindex[0]] > 0.35): continue
        if (event.lep_RelIsoNoFSR[lep_leadindex[1]] > 0.35): continue

        # pt Cut
        if (event.lepFSR_pt[lep_leadindex[0]] <= 20): continue
        if (event.lepFSR_pt[lep_leadindex[1]] <= 10): continue


    if (abs(event.lep_id[lep_leadindex[0]]) == 11):
        Z_e_lIso.Fill(Z_find.M())
    if (abs(event.lep_id[lep_leadindex[0]]) == 13):
        Z_mu_lIso.Fill(Z_find.M())

    # lep Tight ID Cut
    if (not (event.lep_tightId[lep_leadindex[0]])): continue
    if (not (event.lep_tightId[lep_leadindex[1]])): continue

    if (abs(event.lep_id[lep_leadindex[0]]) == 11):
        Z_e_lIso_lTight.Fill(Z_find.M())
    if (abs(event.lep_id[lep_leadindex[0]]) == 13):
        Z_mu_lIso_lTight.Fill(Z_find.M())
################################################################################################


    # m_Z > 50 GeV
################################################################################################
    if (Z_find.M() < 50): continue
    Z_50.Fill(Z_find.M())

################################################################################################


    # Find photon
############################################################

    N_pho = event.pho_pt.size()
    npho.Fill(N_pho)
    if (N_pho < 2): continue


    for i in range(N_pho):
        if (event.pho_hasPixelSeed[i] == 1): continue
        if (event.pho_pt[i] > pho1_maxPt):
            pho1_maxPt = event.pho_pt[i]
            pho1_index = i
            foundpho1 = True

    for j in range(N_pho):
        if (event.pho_hasPixelSeed[j] == 1): continue
        if j == pho1_index: continue
        if (event.pho_pt[j] > pho2_maxPt):
            pho2_maxPt = event.pho_pt[j]
            pho2_index = j
            foundpho2 = True

    if (foundpho1 and foundpho2):

        Run[0] = event.Run
        LumiSect[0] = event.LumiSect
        Event[0] = event.Event
        if isMC:
            lep_dataMC = event.lep_dataMC[lep_leadindex[0]] * event.lep_dataMC[lep_leadindex[1]]
        else:
            lep_dataMC = 1.0
        factor[0] = event.genWeight * event.pileupWeight * lep_dataMC * weight
        event_weight[0] = weight
        event_genWeight[0] = event.genWeight
        event_pileupWeight[0] = event.pileupWeight
        event_dataMCWeight[0] = lep_dataMC

        l1_id[0] = event.lep_id[lep_leadindex[0]]
        l2_id[0] = event.lep_id[lep_leadindex[1]]
    ################################################################################################
        Z_Ceta[0] = -99.0
        H_Ceta[0] = -99.0
        ALP_Ceta[0] = -99.0

        Z_pho_veto[0] = -99.0
        H_pho_veto[0] = -99.0
        ALP_pho_veto[0] = -99.0

        pho1eta[0] = -99.0
        pho1Pt[0] = -99.0
        pho1R9[0] = -99.0
        pho1IetaIeta[0] = -99.0
        pho1IetaIeta55[0] = -99.0
        pho1HOE[0] = -99.0
        pho1CIso[0] = -99.0
        pho1NIso[0] = -99.0
        pho1PIso[0] = -99.0
        pho1PIso_noCorr[0] = -99.0


        pho2eta[0] = -99.0
        pho2Pt[0] = -99.0
        pho2R9[0] = -99.0
        pho2IetaIeta[0] = -99.0
        pho2IetaIeta55[0] = -99.0
        pho2HOE[0] = -99.0
        pho2CIso[0] = -99.0
        pho2NIso[0] = -99.0
        pho2PIso[0] = -99.0
        pho2PIso_noCorr[0] = -99.0

        passIeIe[0] = -99.0
        passHOverE[0] = -99.0
        passChaHadIso[0] = -99.0
        passNeuHadIso[0] = -99.0
        passPhoIso[0] = -99.0
        passdR_gl[0] = -99.0
        passdR_gg[0] = -99.0
        passH_m[0] = -99.0

        Z_HOE[0] = -99.0
        H_HOE[0] = -99.0
        ALP_HOE[0] = -99.0

        Z_CIso[0] = -99.0
        H_CIso[0] = -99.0
        ALP_CIso[0] = -99.0

        Z_NIso[0] = -99.0
        H_NIso[0] = -99.0
        ALP_NIso[0] = -99.0

        Z_IeIe[0] = -99.0
        H_IeIe[0] = -99.0
        ALP_IeIe[0] = -99.0

        Z_PIso[0] = -99.0
        H_PIso[0] = -99.0
        ALP_PIso[0] = -99.0

        dR_g1l1[0] = -99.0
        dR_g1l2[0] = -99.0
        dR_g2l1[0] = -99.0
        dR_g2l2[0] = -99.0
        var_dR_g1Z_beforMVA[0] = -99.0
        var_dPhi_g1Z_beforMVA[0] = -99.0

        Z_pho_veto_mva[0] = -99.0
        H_pho_veto_mva[0] = -99.0
        ALP_pho_veto_mva[0] = -99.0

        dR_g1l1_mva[0] = -99.0
        dR_g1l2_mva[0] = -99.0
        dR_g2l1_mva[0] = -99.0
        dR_g2l2_mva[0] = -99.0
        var_dR_Za[0] = -99.0
        var_dR_g1g2[0] = -99.0
        var_dR_g1Z[0] = -99.0
        var_dEta_g1Z[0] = -99.0
        var_dPhi_g1Z[0] = -99.0
        var_PtaOverMa[0] = -99.0
        var_PtaOverMh[0] = -99.0
        var_Pta[0] = -99.0
        var_MhMa[0] = -99.0
        var_MhMZ[0] = -99.0

        ALP_calculatedPhotonIso[0] = -99.0



        Z_dR[0] = -99.0
        H_dR[0] = -99.0
        ALP_dR[0] = -99.0

        Z_dR_pho[0] = -99.0
        H_dR_pho[0] = -99.0
        ALP_dR_pho[0] = -99.0

        Z_dR_pho_Cmh[0] = -99.0
        H_dR_pho_Cmh[0] = -99.0
        ALP_dR_pho_Cmh[0] = -99.0

        l1_pt[0] = -99.0
        l2_pt[0] = -99.0
        l1_eta[0] = -99.0
        l2_eta[0] = -99.0
        l1_phi[0] = -99.0
        l2_phi[0] = -99.0

        pho1_pt[0] = -99.0
        pho1_eta[0] = -99.0
        pho1_phi[0] = -99.0
        pho1_mva[0] = -99.0
        pho1_matche_PdgId[0] = -99.0
        pho1_matche_MomId[0] = -99.0
        pho1_matche_MomMomId[0] = -99.0
        pho1_matchedR[0] = -99.0

        pho2_pt[0] = -99.0
        pho2_eta[0] = -99.0
        pho2_phi[0] = -99.0
        pho2_mva[0] = -99.0
        pho2_matche_PdgId[0] = -99.0
        pho2_matche_MomId[0] = -99.0
        pho2_matche_MomMomId[0] = -99.0
        pho2_matchedR[0] = -99.0


        Z_m[0] = -99.0
        H_m[0] = -99.0
        ALP_m[0] = -99.0
        dR_pho[0] = -99.0
        H_pt[0] = -99.0

    ################################################################################################

        pho1_find = ROOT.TLorentzVector()
        pho2_find = ROOT.TLorentzVector()

        #pho1_find.SetPtEtaPhiE(event.pho_pt[pho1_index], event.pho_eta[pho1_index], event.pho_phi[pho1_index], event.pho_pt[pho1_index] * np.cosh(event.pho_eta[pho1_index]))
        #pho2_find.SetPtEtaPhiE(event.pho_pt[pho2_index], event.pho_eta[pho2_index], event.pho_phi[pho2_index], event.pho_pt[pho2_index] * np.cosh(event.pho_eta[pho2_index]))

        pho1_find.SetPtEtaPhiM(event.pho_pt[pho1_index], event.pho_eta[pho1_index], event.pho_phi[pho1_index], 0.0)
        pho2_find.SetPtEtaPhiM(event.pho_pt[pho2_index], event.pho_eta[pho2_index], event.pho_phi[pho2_index], 0.0)

        ALP_find = ROOT.TLorentzVector()
        ALP_find = (pho1_find + pho2_find)
    #######################################################################################################

        # Higgs Candidates
    #######################################################################################################
        H_find = ROOT.TLorentzVector()
        H_find = (Z_find + ALP_find)
    #######################################################################################################
        H_twopho[0] = H_find.M()
        # Photon Cuts
    #######################################################################################################

        if (((abs(event.pho_eta[pho1_index]) > 1.566) and (abs(event.pho_eta[pho1_index]) < 2.5)) or (abs(event.pho_eta[pho1_index]) < 1.4442)) and (((abs(event.pho_eta[pho2_index]) > 1.566) and (abs(event.pho_eta[pho2_index]) < 2.5)) or (abs(event.pho_eta[pho2_index]) < 1.4442)):
            #if (((abs(event.pho_eta[pho2_index]) >1.4442) and (abs(event.pho_eta[pho2_index]) < 1.566)) or (abs(event.pho_eta[pho2_index]) >2.5)): continue

            if (event.pho_EleVote[pho1_index] == 0 ): pho_passEleVeto = False
            if (event.pho_EleVote[pho2_index] == 0 ): pho_passEleVeto = False

            if deltaR(pho1_find.Eta(),pho1_find.Phi(),pho2_find.Eta(),pho2_find.Phi()) < 0.3:
                pho1_phoIso = event.pho_photonIso[pho1_index] - pho2_find.Pt()
                pho2_phoIso = event.pho_photonIso[pho2_index] - pho1_find.Pt()
            else:
                pho1_phoIso = event.pho_photonIso[pho1_index]
                pho2_phoIso = event.pho_photonIso[pho2_index]


            # photon 1
            # barrel
            if (abs(event.pho_eta[pho1_index]) < 1.4442):
                if (event.pho_full5x5_sigmaIetaIeta[pho1_index] > 0.00996): pho_passIeIe = False
                if (event.pho_hadronicOverEm[pho1_index] > 0.02148): pho_passHOverE = False
                if (event.pho_chargedHadronIso[pho1_index] > 0.65 ): pho_passChaHadIso = False
                if (event.pho_neutralHadronIso[pho1_index] > (0.317 + event.pho_pt[pho1_index]*0.01512 + event.pho_pt[pho1_index]*event.pho_pt[pho1_index]*0.00002259)): pho_passNeuHadIso = False
                if (event.pho_photonIso[pho1_index] > (2.044 + event.pho_pt[pho1_index]*0.004017)): passedPhoIso = False

            # endcap
            else:
                if (event.pho_full5x5_sigmaIetaIeta[pho1_index] > 0.0271): pho_passIeIe = False
                if (event.pho_hadronicOverEm[pho1_index] > 0.0321): pho_passHOverE = False
                if (event.pho_chargedHadronIso[pho1_index] > 0.517 ): pho_passChaHadIso = False
                if (event.pho_neutralHadronIso[pho1_index] > (2.716 + event.pho_pt[pho1_index]*0.0117 + event.pho_pt[pho1_index]*event.pho_pt[pho1_index]*0.000023)): pho_passNeuHadIso = False
                if (event.pho_photonIso[pho1_index] > (3.032 + event.pho_pt[pho1_index]*0.0037)): passedPhoIso = False
            # photon 2
            # barrel
            if (abs(event.pho_eta[pho2_index]) < 1.4442):
                if (event.pho_full5x5_sigmaIetaIeta[pho2_index] > 0.00996): pho_passIeIe = False
                if (event.pho_hadronicOverEm[pho2_index] > 0.02148): pho_passHOverE = False
                if (event.pho_chargedHadronIso[pho2_index] > 0.65 ): pho_passChaHadIso = False
                if (event.pho_neutralHadronIso[pho2_index] > (0.317 + event.pho_pt[pho1_index]*0.01512 + event.pho_pt[pho1_index]*event.pho_pt[pho1_index]*0.00002259)): pho_passNeuHadIso = False
                if (event.pho_photonIso[pho2_index] > (2.044 + event.pho_pt[pho1_index]*0.004017)): passedPhoIso = False

            # endcap
            else:
                if (event.pho_full5x5_sigmaIetaIeta[pho2_index] > 0.0271): pho_passIeIe = False
                if (event.pho_hadronicOverEm[pho2_index] > 0.0321): pho_passHOverE = False
                if (event.pho_chargedHadronIso[pho2_index] > 0.517 ): pho_passChaHadIso = False
                if (event.pho_neutralHadronIso[pho2_index] > (2.716 + event.pho_pt[pho1_index]*0.0117 + event.pho_pt[pho1_index]*event.pho_pt[pho1_index]*0.000023)): pho_passNeuHadIso = False
                if (event.pho_photonIso[pho2_index] > (3.032 + event.pho_pt[pho1_index]*0.0037)): passedPhoIso = False


            # no Cuts
            Z_Ceta[0] = Z_find.M()
            H_Ceta[0] = H_find.M()
            ALP_Ceta[0] = ALP_find.M()

            #H_Ceta.Fill(H_find.M())

            if pho_passEleVeto:

                Z_pho_veto[0] = Z_find.M()
                H_pho_veto[0] = H_find.M()
                ALP_pho_veto[0] = ALP_find.M()

                dR_l1g1 = deltaR(l1_find.Eta(), l1_find.Phi(), pho1_find.Eta(), pho1_find.Phi())
                dR_l1g2 = deltaR(l1_find.Eta(), l1_find.Phi(), pho2_find.Eta(), pho2_find.Phi())
                dR_l2g1 = deltaR(l2_find.Eta(), l2_find.Phi(), pho1_find.Eta(), pho1_find.Phi())
                dR_l2g2 = deltaR(l2_find.Eta(), l2_find.Phi(), pho2_find.Eta(), pho2_find.Phi())


                var_dR_g1Z_beforMVA[0] = deltaR(pho1_find.Eta(), pho1_find.Phi(), Z_find.Eta(), Z_find.Phi())
                var_dPhi_g1Z_beforMVA[0] =  pho1_find.Phi()-Z_find.Phi()

                dR_g1g2 = deltaR(pho1_find.Eta(), pho1_find.Phi(), pho2_find.Eta(), pho2_find.Phi())

                ################################################################ MVA ################################################################

                #var_value_pho1 = {}
                #var_value_pho1['pho1Pt'] = event.pho_pt[pho1_index]
                #var_value_pho1['pho1R9'] = event.pho_R9[pho1_index]
                #var_value_pho1['pho1IetaIeta'] = event.pho_sigmaIetaIeta[pho1_index]
                #var_value_pho1['pho1HOE'] = event.pho_hadronicOverEm[pho1_index]
                #var_value_pho1['pho1CIso'] = event.pho_chargedHadronIso[pho1_index]
                #var_value_pho1['pho1NIso'] = event.pho_neutralHadronIso[pho1_index]
                #var_value_pho1['pho1PIso'] = event.pho_photonIso[pho1_index]

                #var_value_pho2 = {}
                #var_value_pho2['pho1Pt'] = event.pho_pt[pho2_index]
                #var_value_pho2['pho1R9'] = event.pho_R9[pho2_index]
                #var_value_pho2['pho1IetaIeta'] = event.pho_sigmaIetaIeta[pho2_index]
                #var_value_pho2['pho1HOE'] = event.pho_hadronicOverEm[pho2_index]
                #var_value_pho2['pho1CIso'] = event.pho_chargedHadronIso[pho2_index]
                #var_value_pho2['pho1NIso'] = event.pho_neutralHadronIso[pho2_index]
                #var_value_pho2['pho1PIso'] = event.pho_photonIso[pho2_index]

                #mva_value_pho1 = calMVA(reader,var_value_pho1)
                #mva_value_pho2 = calMVA(reader,var_value_pho2)

                #print "pho1 mva value: ",mva_value_pho1
                #print "pho2 mva value: ",mva_value_pho2

                ################################################################ End MVA ################################################################


                dR_g1l1[0] = dR_l1g1
                dR_g1l2[0] = dR_l1g2
                dR_g2l1[0] = dR_l2g1
                dR_g2l2[0] = dR_l2g2

                dR_pho[0] = dR_g1g2

                pho1eta[0] = event.pho_eta[pho1_index]
                pho1Pt[0] = event.pho_pt[pho1_index]
                pho1R9[0] = event.pho_R9[pho1_index]
                pho1IetaIeta[0] = event.pho_sigmaIetaIeta[pho1_index]
                pho1IetaIeta55[0] = event.pho_full5x5_sigmaIetaIeta[pho1_index]
                pho1HOE[0] = event.pho_hadronicOverEm[pho1_index]
                pho1CIso[0] = event.pho_chargedHadronIso[pho1_index]
                pho1NIso[0] = event.pho_neutralHadronIso[pho1_index]
                pho1PIso[0] = pho1_phoIso
                pho1PIso_noCorr[0] = event.pho_photonIso[pho1_index]


                pho2eta[0] = event.pho_eta[pho2_index]
                pho2Pt[0] = event.pho_pt[pho2_index]
                pho2R9[0] = event.pho_R9[pho2_index]
                pho2IetaIeta[0] = event.pho_sigmaIetaIeta[pho2_index]
                pho2IetaIeta55[0] = event.pho_full5x5_sigmaIetaIeta[pho2_index]
                pho2HOE[0] = event.pho_hadronicOverEm[pho2_index]
                pho2CIso[0] = event.pho_chargedHadronIso[pho2_index]
                pho2NIso[0] = event.pho_neutralHadronIso[pho2_index]
                pho2PIso[0] = pho2_phoIso
                pho2PIso_noCorr[0] = event.pho_photonIso[pho2_index]

                var_dR_Za[0] = deltaR(Z_find.Eta(), Z_find.Phi(), ALP_find.Eta(), ALP_find.Phi())
                var_dR_g1g2[0] = dR_g1g2
                var_dR_g1Z[0] = deltaR(pho1_find.Eta(), pho1_find.Phi(), Z_find.Eta(), Z_find.Phi())
                var_dEta_g1Z[0] = pho1_find.Eta() - Z_find.Eta()
                var_dPhi_g1Z[0] = pho1_find.Phi() - Z_find.Phi()
                var_PtaOverMa[0] = ALP_find.Pt()/ALP_find.M()
                var_PtaOverMh[0] = ALP_find.Pt()/H_find.M()
                var_Pta[0] = ALP_find.Pt()
                var_MhMa[0] = ALP_find.M()+H_find.M()
                var_MhMZ[0] = Z_find.M()+H_find.M()

                ################# calculate ALP's photon isolation #################
                checkDuplicate = ''
                ALP_calculatedPhotonIso_tmp = 0.0
                isReco = False
                for i in range(event.pho_PF_Id.size()):
                    if str(event.pho_PF_Id[i]) in checkDuplicate: continue
                    checkDuplicate = checkDuplicate + ' ' + str(event.pho_PF_Id[i])
                    dR_AP = deltaR(ALP_find.Eta(), ALP_find.Phi(), event.pho_PF_eta[i], event.pho_PF_phi[i])

                    # check if pfPhoton == recoPhoton
                    for pho in range(N_pho):
                        dR_Pg = deltaR(event.pho_eta[pho], event.pho_phi[pho], event.pho_PF_eta[i], event.pho_PF_phi[i])
                        if dR_Pg < 0.08: isReco = True

                    if isReco: continue

                    if dR_AP <= 0.3:
                        ALP_calculatedPhotonIso_tmp += event.pho_PF_pt[i]

                #print checkDuplicate
                ALP_calculatedPhotonIso[0] = ALP_calculatedPhotonIso_tmp
                ################# END calculate ALP's photon isolation #################

                l1_pt[0] = event.lepFSR_pt[lep_leadindex[0]]
                l2_pt[0] = event.lepFSR_pt[lep_leadindex[1]]
                l1_eta[0] = event.lepFSR_eta[lep_leadindex[0]]
                l2_eta[0] = event.lepFSR_eta[lep_leadindex[1]]
                l1_phi[0] = event.lepFSR_phi[lep_leadindex[0]]
                l2_phi[0] = event.lepFSR_phi[lep_leadindex[1]]

                ################# define cut flow #################
                cutFlow = {}

                cutdR_gl = (dR_l1g1 > 0.4) and (dR_l1g2 > 0.4) and (dR_l2g1 > 0.4) and (dR_l2g2 > 0.4)
                if mass == 'M1':
                    cutdR_gg = dR_g1g2 < 0.15 ## 1GeV
                elif mass == 'M5':
                    cutdR_gg = 0.1 < dR_g1g2 < 0.5 ## 5GeV
                elif mass == 'M15':
                    cutdR_gg = 0.2 < dR_g1g2 < 2 ## 15 GeV
                else:
                    cutdR_gg = 0.6 < dR_g1g2 < 6  ## 30 GeV

                cutH_m = (H_find.M()>100) and (H_find.M()<180)

                cutFlow['cut1'] = pho_passChaHadIso
                cutFlow['cut2'] = pho_passNeuHadIso
                cutFlow['cut3'] = cutdR_gl
                cutFlow['cut4'] = cutdR_gg
                cutFlow['cut5'] = pho_passHOverE
                cutFlow['cut6'] = pho_passIeIe
                cutFlow['cut7'] = passedPhoIso
                cutFlow['cut8'] = cutH_m

                passIeIe[0] = pho_passIeIe
                passHOverE[0] = pho_passHOverE
                passChaHadIso[0] = pho_passChaHadIso
                passNeuHadIso[0] = pho_passNeuHadIso
                passPhoIso[0] = passedPhoIso
                passdR_gl[0] = cutdR_gl
                passdR_gg[0] = cutdR_gg
                passH_m[0] = cutH_m

                ###################################################

                if cutBased:

                    if cutFlow['cut1']:
                        Z_CIso[0] = Z_find.M()
                        H_CIso[0] = H_find.M()
                        ALP_CIso[0] = ALP_find.M()

                        if cutFlow['cut2']:
                            Z_NIso[0] = Z_find.M()
                            H_NIso[0] = H_find.M()
                            ALP_NIso[0] = ALP_find.M()

                            if cutFlow['cut3']:

                                Z_dR[0] = Z_find.M()
                                H_dR[0] = H_find.M()
                                ALP_dR[0] = ALP_find.M()

                                if cutFlow['cut4']:
                                    Z_dR_pho[0] = Z_find.M()
                                    H_dR_pho[0] = H_find.M()
                                    ALP_dR_pho[0] = ALP_find.M()

                                    if cutFlow['cut5']:
                                        Z_HOE[0] = Z_find.M()
                                        H_HOE[0] = H_find.M()
                                        ALP_HOE[0] = ALP_find.M()

                                        MVA_list = [event.pho_full5x5_sigmaIetaIeta[pho1_index], event.pho_photonIso[pho1_index], event.pho_full5x5_sigmaIetaIeta[pho2_index], event.pho_photonIso[pho2_index], ALP_calculatedPhotonIso_tmp, ALP_find.Pt()/ALP_find.M(), ALP_find.Pt()/H_find.M(), Z_find.M()+H_find.M()]
                                        MVA_value = model.predict_proba(MVA_list)[:, 1]

                                        if MVA_value>mvaCut:
                                            Z_m[0] = Z_find.M()
                                            H_m[0] = H_find.M()
                                            ALP_m[0] = ALP_find.M()


                                        if cutFlow['cut6']:
                                            Z_IeIe[0] = Z_find.M()
                                            H_IeIe[0] = H_find.M()
                                            ALP_IeIe[0] = ALP_find.M()

                                            if cutFlow['cut7']:
                                                Z_PIso[0] = Z_find.M()
                                                H_PIso[0] = H_find.M()
                                                ALP_PIso[0] = ALP_find.M()

                                                if cutFlow['cut8'] :
                                                    Z_dR_pho_Cmh[0] = Z_find.M()
                                                    H_dR_pho_Cmh[0] = H_find.M()
                                                    ALP_dR_pho_Cmh[0] = ALP_find.M()

                                                    # Fill Tree



                                                    pho1_pt[0] = event.pho_pt[pho1_index]
                                                    pho1_eta[0] = event.pho_eta[pho1_index]
                                                    pho1_phi[0] = event.pho_phi[pho1_index]
                                                    pho1_mva[0] = event.pho_mva[pho1_index]
                                                    #pho1_matche_PdgId[0] = event.pho_matchedR03_PdgId[pho1_index]
                                                    #pho1_matche_MomId[0] = event.pho_matchedR03_MomId[pho1_index]
                                                    #pho1_matche_MomMomId[0] = event.pho_matchedR03_MomMomId[pho1_index]
                                                    #pho1_matchedR[0] = event.pho_matchedR[pho1_index]

                                                    pho2_pt[0] = event.pho_pt[pho2_index]
                                                    pho2_eta[0] = event.pho_eta[pho2_index]
                                                    pho2_phi[0] = event.pho_phi[pho2_index]
                                                    pho2_mva[0] = event.pho_mva[pho2_index]
                                                    #pho2_matche_PdgId[0] = event.pho_matchedR03_PdgId[pho2_index]
                                                    #pho2_matche_MomId[0] = event.pho_matchedR03_MomId[pho2_index]
                                                    #pho2_matche_MomMomId[0] = event.pho_matchedR03_MomMomId[pho2_index]
                                                    #pho2_matchedR[0] = event.pho_matchedR[pho2_index]


                                                    H_pt[0] = H_find.Pt()
                                                # Higgs mass cut
                                            # dR(pho1 , pho2)
                                        # dR(lepton, photon)
                                    # photon isolation
                                # photon sigmaIetaIeta
                            # photon neutralHadronIso
                        # photon chargedHadronIso
                    # photon HOverE



                if mvaBased:


                    if (mva_value_pho1 >= 0.211842108794):
                        if (mva_value_pho2 >= 0.211842108794):

                            Z_pho_veto_mva[0] = Z_find.M()
                            H_pho_veto_mva[0] = H_find.M()
                            ALP_pho_veto_mva[0] = ALP_find.M()

                            #dR_l1g1 = deltaR(l1_find.Eta(), l1_find.Phi(), pho1_find.Eta(), pho1_find.Phi())
                            #dR_l1g2 = deltaR(l1_find.Eta(), l1_find.Phi(), pho2_find.Eta(), pho2_find.Phi())
                            #dR_l2g1 = deltaR(l2_find.Eta(), l2_find.Phi(), pho1_find.Eta(), pho1_find.Phi())
                            #dR_l2g2 = deltaR(l2_find.Eta(), l2_find.Phi(), pho2_find.Eta(), pho2_find.Phi())
                            dR_g1l1_mva[0] = dR_l1g1
                            dR_g1l2_mva[0] = dR_l1g2
                            dR_g2l1_mva[0] = dR_l2g1
                            dR_g2l2_mva[0] = dR_l2g2
                            var_dR_Za[0] = deltaR(Z_find.Eta(), Z_find.Phi(), ALP_find.Eta(), ALP_find.Phi())
                            var_dR_g1g2[0] = dR_g1g2
                            var_dR_g1Z[0] = deltaR(pho1_find.Eta(), pho1_find.Phi(), Z_find.Eta(), Z_find.Phi())
                            var_dEta_g1Z[0] = pho1_find.Eta() - Z_find.Eta()
                            var_dPhi_g1Z[0] = pho1_find.Phi() - Z_find.Phi()
                            var_PtaOverMa[0] = ALP_find.Pt()/ALP_find.M()
                            var_PtaOverMh[0] = ALP_find.Pt()/H_find.M()
                            var_Pta[0] = ALP_find.Pt()
                            var_MhMa[0] = ALP_find.M()+H_find.M()
                            var_MhMZ[0] = Z_find.M()+H_find.M()



                            if (dR_l1g1 > 0.4) and (dR_l1g2 > 0.4) and (dR_l2g1 > 0.4) and (dR_l2g2 > 0.4):

                                Z_dR[0] = Z_find.M()
                                H_dR[0] = H_find.M()
                                ALP_dR[0] = ALP_find.M()

                                if dR_g1g2 < 1:
                                    Z_dR_pho[0] = Z_find.M()
                                    H_dR_pho[0] = H_find.M()
                                    ALP_dR_pho[0] = ALP_find.M()

                                    if (H_find.M()>100) and (H_find.M()<180) :
                                        Z_dR_pho_Cmh[0] = Z_find.M()
                                        H_dR_pho_Cmh[0] = H_find.M()
                                        ALP_dR_pho_Cmh[0] = ALP_find.M()

                                        # Fill Tree

                                        pho1_pt[0] = event.pho_pt[pho1_index]
                                        pho1_eta[0] = event.pho_eta[pho1_index]
                                        pho1_phi[0] = event.pho_phi[pho1_index]
                                        pho1_mva[0] = event.pho_mva[pho1_index]
                                        #pho1_matche_PdgId[0] = event.pho_matchedR03_PdgId[pho1_index]
                                        #pho1_matche_MomId[0] = event.pho_matchedR03_MomId[pho1_index]
                                        #pho1_matche_MomMomId[0] = event.pho_matchedR03_MomMomId[pho1_index]
                                        #pho1_matchedR[0] = event.pho_matchedR[pho1_index]

                                        pho2_pt[0] = event.pho_pt[pho2_index]
                                        pho2_eta[0] = event.pho_eta[pho2_index]
                                        pho2_phi[0] = event.pho_phi[pho2_index]
                                        pho2_mva[0] = event.pho_mva[pho2_index]
                                        #pho2_matche_PdgId[0] = event.pho_matchedR03_PdgId[pho2_index]
                                        #pho2_matche_MomId[0] = event.pho_matchedR03_MomId[pho2_index]
                                        #pho2_matche_MomMomId[0] = event.pho_matchedR03_MomMomId[pho2_index]
                                        #pho2_matchedR[0] = event.pho_matchedR[pho2_index]


                                        Z_m[0] = Z_find.M()
                                        H_m[0] = H_find.M()
                                        ALP_m[0] = ALP_find.M()
                                        dR_pho[0] = dR_g1g2
                                        H_pt[0] = H_find.Pt()

            # photon electron veto
        # photon eta cuts
    # find two photons

#######################################################################################################

        passedEvents.Fill()











file_out.Write()
file_out.Close()

sw.Stop()
print 'Real time: ' + str(round(sw.RealTime() / 60.0,2)) + ' minutes'
print 'CPU time:  ' + str(round(sw.CpuTime() / 60.0,2)) + ' minutes'
