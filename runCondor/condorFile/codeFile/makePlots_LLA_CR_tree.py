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
parser.add_argument("-y", "--Year", dest="year", default="2017", help="which year's datasetes")
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

import SystematicUnc_ALP as Sys

########## photon SFs
if args.year == '2016':
    f_SFs = '/publicfs/cms/user/wangzebing/ALP/Plot/SFs/egammaEffi_2016.txt'
    BDT_filename="/publicfs/cms/user/wangzebing/ALP/Analysis_code/MVA/weight/nodR/model_ALP_massindependent_2016.pkl"
    mvaCut = 0.9381

    chCut_EB = 0.202
    chCut_EE = 0.034
    neuCut_EB = [0.264, 0.0148, 0.000017]
    neuCut_EE = [0.586, 0.0163, 0.000014]
    hoeCut_EB = 0.0269
    hoeCut_EE = 0.0213
elif args.year == '2017':
    f_SFs = '/publicfs/cms/user/wangzebing/ALP/Plot/SFs/egammaEffi_2017.txt'
    BDT_filename="/publicfs/cms/user/wangzebing/ALP/Analysis_code/MVA/weight/nodR/model_ALP_massindependent_2017.pkl"
    mvaCut = 0.8571

    chCut_EB = 0.65
    chCut_EE = 0.517
    neuCut_EB = [0.317, 0.01512, 0.00002259]
    neuCut_EE = [2.716, 0.0117, 0.000023]
    hoeCut_EB = 0.02148
    hoeCut_EE = 0.0321
elif args.year == '2018':
    f_SFs = '/publicfs/cms/user/wangzebing/ALP/Plot/SFs/egammaEffi_2018.txt'
    BDT_filename="/publicfs/cms/user/wangzebing/ALP/Analysis_code/MVA/weight/nodR/model_ALP_massindependent_2018.pkl"
    mvaCut = 0.9204

    chCut_EB = 0.65
    chCut_EE = 0.517
    neuCut_EB = [0.317, 0.01512, 0.00002259]
    neuCut_EE = [2.716, 0.0117, 0.000023]
    hoeCut_EB = 0.02148
    hoeCut_EE = 0.0321
else:
    print "do not include at 2016/2017/2018"
    exit(0)

# load the model from disk
model = pickle.load(open(BDT_filename, 'rb'))

f = open(f_SFs)
lines = f.readlines()
pt_l = []
pt_r = []
eta_l = []
eta_r = []
SFs = []
for line in lines:
    if line[0] == '#': continue
    eta_l.append(float(line.split()[0]))
    eta_r.append(float(line.split()[1]))
    pt_l.append(float(line.split()[2]))
    pt_r.append(float(line.split()[3]))
    SFs.append(float(line.split()[4])/float(line.split()[6]))




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
event_weight = array('f',[0.])
event_genWeight = array('f',[-1.])
event_pileupWeight = array('f',[-1.])
event_dataMCWeight = array('f',[-1.])

Z_m = array('f',[0.])
H_m = array('f',[0.])
ALP_m = array('f',[0.])
H_pt = array('f',[0.])


l1_pt = array('f',[0.])
l1_eta = array('f',[0.])
l1_phi = array('f',[0.])
l1_id = array('i',[0])
l1_scaleup = array('f',[0.])
l1_scaledn = array('f',[0.])
l1_smearup = array('f',[0.])
l1_smeardn = array('f',[0.])

l2_pt = array('f',[0.])
l2_eta = array('f',[0.])
l2_phi = array('f',[0.])
l2_id = array('i',[0])
l2_scaleup = array('f',[0.])
l2_scaledn = array('f',[0.])
l2_smearup = array('f',[0.])
l2_smeardn = array('f',[0.])

# photon var
pho1eta = array('f',[0.])
pho1Pt = array('f',[0.])
pho1phi = array('f',[0.])
pho1R9 = array('f',[0.])
pho1IetaIeta = array('f',[0.])
pho1IetaIeta55 = array('f',[0.])
pho1HOE = array('f',[0.])
pho1CIso = array('f',[0.])
pho1NIso = array('f',[0.])
pho1PIso = array('f',[0.])
pho1PIso_noCorr = array('f',[0.])
pho1SFs = array('f',[0.])
pho1scaleup = array('f',[0.])
pho1scaledn = array('f',[0.])
pho1smearup = array('f',[0.])
pho1smeardn = array('f',[0.])
pho1ShowerShapeSys = array('f',[0.])

pho2eta = array('f',[0.])
pho2Pt = array('f',[0.])
pho2phi = array('f',[0.])
pho2R9 = array('f',[0.])
pho2IetaIeta = array('f',[0.])
pho2IetaIeta55 = array('f',[0.])
pho2HOE = array('f',[0.])
pho2CIso = array('f',[0.])
pho2NIso = array('f',[0.])
pho2PIso = array('f',[0.])
pho2PIso_noCorr = array('f',[0.])
pho2SFs = array('f',[0.])
pho2scaleup = array('f',[0.])
pho2scaledn = array('f',[0.])
pho2smearup = array('f',[0.])
pho2smeardn = array('f',[0.])
pho2ShowerShapeSys = array('f',[0.])

dR_g1l1 = array('f',[-1.])
dR_g1l2 = array('f',[-1.])
dR_g2l1 = array('f',[-1.])
dR_g2l2 = array('f',[-1.])

dR_pho = array('f',[0.])

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

# photon cut

passEleVeto = array('f',[0.])
passIeIe = array('f',[0.])
passHOverE = array('f',[0.])
passChaHadIso = array('f',[0.])
passNeuHadIso = array('f',[0.])
passPhoIso = array('f',[0.])
passdR_gl = array('f',[0.])
passdR_gg = array('f',[0.])
passH_m = array('f',[0.])
passBDT = array('f',[0.])
Val_BDT = array('f',[0.])


################################################################################################

passedEvents = ROOT.TTree("passedEvents","passedEvents")

################################################################################################
passedEvents.Branch("Run",Run,"Run/L")
passedEvents.Branch("LumiSect",LumiSect,"LumiSect/L")
passedEvents.Branch("Event",Event,"Event/L")
passedEvents.Branch("factor",factor,"factor/F")
passedEvents.Branch("event_weight",event_weight,"event_weight/F")
passedEvents.Branch("event_genWeight",event_genWeight,"event_genWeight/F")
passedEvents.Branch("event_pileupWeight",event_pileupWeight,"event_pileupWeight/F")
passedEvents.Branch("event_dataMCWeight",event_dataMCWeight,"event_dataMCWeight/F")

passedEvents.Branch("Z_m",Z_m,"Z_m/F")
passedEvents.Branch("H_m",H_m,"H_m/F")
passedEvents.Branch("ALP_m",ALP_m,"ALP_m/F")
passedEvents.Branch("H_pt",H_pt,"H_pt/F")


passedEvents.Branch("l1_pt",l1_pt,"l1_pt/F")
passedEvents.Branch("l1_eta",l1_eta,"l1_eta/F")
passedEvents.Branch("l1_phi",l1_phi,"l1_phi/F")
passedEvents.Branch("l1_id",l1_id,"l1_id/I")
passedEvents.Branch("l1_scaleup",l1_scaleup,"l1_scaleup/F")
passedEvents.Branch("l1_scaledn",l1_scaledn,"l1_scaledn/F")
passedEvents.Branch("l1_smearup",l1_smearup,"l1_smearup/F")
passedEvents.Branch("l1_smeardn",l1_smeardn,"l1_smeardn/F")

passedEvents.Branch("l2_pt",l2_pt,"l2_pt/F")
passedEvents.Branch("l2_eta",l2_eta,"l2_eta/F")
passedEvents.Branch("l2_phi",l2_phi,"l2_phi/F")
passedEvents.Branch("l2_id",l2_id,"l2_id/I")
passedEvents.Branch("l2_scaleup",l2_scaleup,"l2_scaleup/F")
passedEvents.Branch("l2_scaledn",l2_scaledn,"l2_scaledn/F")
passedEvents.Branch("l2_smearup",l2_smearup,"l2_smearup/F")
passedEvents.Branch("l2_smeardn",l2_smeardn,"l2_smeardn/F")


passedEvents.Branch("pho1eta",pho1eta,"pho1eta/F")
passedEvents.Branch("pho1Pt",pho1Pt,"pho1Pt/F")
passedEvents.Branch("pho1phi",pho1phi,"pho1phi/F")
passedEvents.Branch("pho1R9",pho1R9,"pho1R9/F")
passedEvents.Branch("pho1IetaIeta",pho1IetaIeta,"pho1IetaIeta/F")
passedEvents.Branch("pho1IetaIeta55",pho1IetaIeta55,"pho1IetaIeta55/F")
passedEvents.Branch("pho1HOE",pho1HOE,"pho1HOE/F")
passedEvents.Branch("pho1CIso",pho1CIso,"pho1CIso/F")
passedEvents.Branch("pho1NIso",pho1NIso,"pho1NIso/F")
passedEvents.Branch("pho1PIso",pho1PIso,"pho1PIso/F")
passedEvents.Branch("pho1PIso_noCorr",pho1PIso_noCorr,"pho1PIso_noCorr/F")
passedEvents.Branch("pho1SFs",pho1SFs,"pho1SFs/F")
passedEvents.Branch("pho1scaleup",pho1scaleup,"pho1scaleup/F")
passedEvents.Branch("pho1scaledn",pho1scaledn,"pho1scaledn/F")
passedEvents.Branch("pho1smearup",pho1smearup,"pho1smearup/F")
passedEvents.Branch("pho1smeardn",pho1smeardn,"pho1smeardn/F")
passedEvents.Branch("pho1ShowerShapeSys",pho1ShowerShapeSys,"pho1ShowerShapeSys/F")

passedEvents.Branch("pho2eta",pho2eta,"pho2eta/F")
passedEvents.Branch("pho2Pt",pho2Pt,"pho2Pt/F")
passedEvents.Branch("pho2phi",pho2phi,"pho2phi/F")
passedEvents.Branch("pho2R9",pho2R9,"pho2R9/F")
passedEvents.Branch("pho2IetaIeta",pho2IetaIeta,"pho2IetaIeta/F")
passedEvents.Branch("pho2IetaIeta55",pho2IetaIeta55,"pho2IetaIeta55/F")
passedEvents.Branch("pho2HOE",pho2HOE,"pho2HOE/F")
passedEvents.Branch("pho2CIso",pho2CIso,"pho2CIso/F")
passedEvents.Branch("pho2NIso",pho2NIso,"pho2NIso/F")
passedEvents.Branch("pho2PIso",pho2PIso,"pho2PIso/F")
passedEvents.Branch("pho2PIso_noCorr",pho2PIso_noCorr,"pho2PIso_noCorr/F")
passedEvents.Branch("pho2SFs",pho2SFs,"pho2SFs/F")
passedEvents.Branch("pho2scaleup",pho2scaleup,"pho2scaleup/F")
passedEvents.Branch("pho2scaledn",pho2scaledn,"pho2scaledn/F")
passedEvents.Branch("pho2smearup",pho2smearup,"pho2smearup/F")
passedEvents.Branch("pho2smeardn",pho2smeardn,"pho2smeardn/F")
passedEvents.Branch("pho2ShowerShapeSys",pho2ShowerShapeSys,"pho2ShowerShapeSys/F")

passedEvents.Branch("dR_g1l1",dR_g1l1,"dR_g1l1/F")
passedEvents.Branch("dR_g1l2",dR_g1l2,"dR_g1l2/F")
passedEvents.Branch("dR_g2l1",dR_g2l1,"dR_g2l1/F")
passedEvents.Branch("dR_g2l2",dR_g2l2,"dR_g2l2/F")

passedEvents.Branch("dR_pho",dR_pho,"dR_pho/F")

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

passedEvents.Branch("passEleVeto",passEleVeto,"passEleVeto/F")
passedEvents.Branch("passIeIe",passIeIe,"passIeIe/F")
passedEvents.Branch("passHOverE",passHOverE,"passHOverE/F")
passedEvents.Branch("passChaHadIso",passChaHadIso,"passChaHadIso/F")
passedEvents.Branch("passNeuHadIso",passNeuHadIso,"passNeuHadIso/F")
passedEvents.Branch("passPhoIso",passPhoIso,"passPhoIso/F")
passedEvents.Branch("passdR_gl",passdR_gl,"passdR_gl/F")
passedEvents.Branch("passdR_gg",passdR_gg,"passdR_gg/F")
passedEvents.Branch("passH_m",passH_m,"passH_m/F")
passedEvents.Branch("passBDT",passBDT,"passBDT/F")
passedEvents.Branch("Val_BDT",Val_BDT,"Val_BDT/F")

################################################################################################

#reader = bookMVA()
########################################################################################################################################
#Loop over all the events in the input ntuple
for ievent,event in enumerate(tchain):#, start=650000):
    if ievent > args.maxevents and args.maxevents != -1: break
    if ievent == 500000: break
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
        if (event.pho_hasPixelSeed[i] == 0): continue
        if (event.pho_pt[i] > pho1_maxPt):
            pho1_maxPt = event.pho_pt[i]
            pho1_index = i
            foundpho1 = True

    for j in range(N_pho):
        if (event.pho_hasPixelSeed[j] == 0): continue
        if j == pho1_index: continue
        if (event.pho_pt[j] > pho2_maxPt):
            pho2_maxPt = event.pho_pt[j]
            pho2_index = j
            foundpho2 = True

    if (foundpho1 and foundpho2):


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
                if (event.pho_hadronicOverEm[pho1_index] > hoeCut_EB): pho_passHOverE = False
                if (event.pho_chargedHadronIso[pho1_index] > chCut_EB ): pho_passChaHadIso = False
                if (event.pho_neutralHadronIso[pho1_index] > (neuCut_EB[0] + event.pho_pt[pho1_index]*neuCut_EB[1] + event.pho_pt[pho1_index]*event.pho_pt[pho1_index]*neuCut_EB[2])): pho_passNeuHadIso = False
                if (event.pho_photonIso[pho1_index] > (2.044 + event.pho_pt[pho1_index]*0.004017)): passedPhoIso = False

            # endcap
            else:
                if (event.pho_full5x5_sigmaIetaIeta[pho1_index] > 0.0271): pho_passIeIe = False
                if (event.pho_hadronicOverEm[pho1_index] > hoeCut_EE): pho_passHOverE = False
                if (event.pho_chargedHadronIso[pho1_index] > chCut_EE ): pho_passChaHadIso = False
                if (event.pho_neutralHadronIso[pho1_index] > (neuCut_EE[0] + event.pho_pt[pho1_index]*neuCut_EE[1] + event.pho_pt[pho1_index]*event.pho_pt[pho1_index]*neuCut_EE[2])): pho_passNeuHadIso = False
                if (event.pho_photonIso[pho1_index] > (3.032 + event.pho_pt[pho1_index]*0.0037)): passedPhoIso = False
            # photon 2
            # barrel
            if (abs(event.pho_eta[pho2_index]) < 1.4442):
                if (event.pho_full5x5_sigmaIetaIeta[pho2_index] > 0.00996): pho_passIeIe = False
                if (event.pho_hadronicOverEm[pho2_index] > hoeCut_EB): pho_passHOverE = False
                if (event.pho_chargedHadronIso[pho2_index] > chCut_EB ): pho_passChaHadIso = False
                if (event.pho_neutralHadronIso[pho2_index] > (neuCut_EB[0] + event.pho_pt[pho1_index]*neuCut_EB[1] + event.pho_pt[pho1_index]*event.pho_pt[pho1_index]*neuCut_EB[2])): pho_passNeuHadIso = False
                if (event.pho_photonIso[pho2_index] > (2.044 + event.pho_pt[pho1_index]*0.004017)): passedPhoIso = False

            # endcap
            else:
                if (event.pho_full5x5_sigmaIetaIeta[pho2_index] > 0.0271): pho_passIeIe = False
                if (event.pho_hadronicOverEm[pho2_index] > hoeCut_EE): pho_passHOverE = False
                if (event.pho_chargedHadronIso[pho2_index] > chCut_EE ): pho_passChaHadIso = False
                if (event.pho_neutralHadronIso[pho2_index] > (neuCut_EE[0] + event.pho_pt[pho1_index]*neuCut_EE[1] + event.pho_pt[pho1_index]*event.pho_pt[pho1_index]*neuCut_EE[2])): pho_passNeuHadIso = False
                if (event.pho_photonIso[pho2_index] > (3.032 + event.pho_pt[pho1_index]*0.0037)): passedPhoIso = False

            #H_Ceta.Fill(H_find.M())

            dR_l1g1 = deltaR(l1_find.Eta(), l1_find.Phi(), pho1_find.Eta(), pho1_find.Phi())
            dR_l1g2 = deltaR(l1_find.Eta(), l1_find.Phi(), pho2_find.Eta(), pho2_find.Phi())
            dR_l2g1 = deltaR(l2_find.Eta(), l2_find.Phi(), pho1_find.Eta(), pho1_find.Phi())
            dR_l2g2 = deltaR(l2_find.Eta(), l2_find.Phi(), pho2_find.Eta(), pho2_find.Phi())

            dR_g1g2 = deltaR(pho1_find.Eta(), pho1_find.Phi(), pho2_find.Eta(), pho2_find.Phi())
            dR_g1Z = deltaR(pho1_find.Eta(), pho1_find.Phi(), Z_find.Eta(), Z_find.Phi())

            cutdR_gl = (dR_l1g1 > 0.4) and (dR_l1g2 > 0.4) and (dR_l2g1 > 0.4) and (dR_l2g2 > 0.4)
            cutdR_gg = dR_g1g2 > 0.02
            cutH_m = (H_find.M()>118) and (H_find.M()<130)

            if  pho_passHOverE: continue
            if  pho_passChaHadIso: continue
            if  pho_passNeuHadIso: continue
            if not cutdR_gl: continue
            if not cutH_m:continue
            if pho1_find.Pt() > 45: continue
            if pho2_find.Pt() > 45: continue

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

            Z_m[0] = Z_find.M()
            H_m[0] = H_find.M()
            ALP_m[0] = ALP_find.M()
            H_pt[0] = H_find.Pt()

            l1_id[0] = event.lep_id[lep_leadindex[0]]
            l2_id[0] = event.lep_id[lep_leadindex[1]]
            l1_pt[0] = event.lepFSR_pt[lep_leadindex[0]]
            l2_pt[0] = event.lepFSR_pt[lep_leadindex[1]]
            l1_eta[0] = event.lepFSR_eta[lep_leadindex[0]]
            l2_eta[0] = event.lepFSR_eta[lep_leadindex[1]]
            l1_phi[0] = event.lepFSR_phi[lep_leadindex[0]]
            l2_phi[0] = event.lepFSR_phi[lep_leadindex[1]]

            l1_scaleup[0] = event.lep_scaleup[lep_leadindex[0]]
            l1_scaledn[0] = event.lep_scaledn[lep_leadindex[0]]
            l1_smearup[0] = event.lep_smearup[lep_leadindex[0]]
            l1_smeardn[0] = event.lep_smeardn[lep_leadindex[0]]
            l2_scaleup[0] = event.lep_scaleup[lep_leadindex[1]]
            l2_scaledn[0] = event.lep_scaledn[lep_leadindex[1]]
            l2_smearup[0] = event.lep_smearup[lep_leadindex[1]]
            l2_smeardn[0] = event.lep_smeardn[lep_leadindex[1]]

            ################################## Photon variables ##############################################################

            dR_g1l1[0] = dR_l1g1
            dR_g1l2[0] = dR_l1g2
            dR_g2l1[0] = dR_l2g1
            dR_g2l2[0] = dR_l2g2

            dR_pho[0] = dR_g1g2

            pho1Pt[0] = event.pho_pt[pho1_index]
            pho1eta[0] = event.pho_eta[pho1_index]
            pho1phi[0] = event.pho_phi[pho1_index]
            pho1R9[0] = event.pho_R9[pho1_index]
            pho1IetaIeta[0] = event.pho_sigmaIetaIeta[pho1_index]
            pho1IetaIeta55[0] = event.pho_full5x5_sigmaIetaIeta[pho1_index]
            pho1HOE[0] = event.pho_hadronicOverEm[pho1_index]
            pho1CIso[0] = event.pho_chargedHadronIso[pho1_index]
            pho1NIso[0] = event.pho_neutralHadronIso[pho1_index]
            pho1PIso[0] = pho1_phoIso
            pho1PIso_noCorr[0] = event.pho_photonIso[pho1_index]
            pho1scaleup[0] = event.pho_scaleup[pho1_index]
            pho1scaledn[0] = event.pho_scaledn[pho1_index]
            pho1smearup[0] = event.pho_smearup[pho1_index]
            pho1smeardn[0] = event.pho_smeardn[pho1_index]
            pho1ShowerShapeSys[0] = Sys.showerShapeUncVal(event.pho_eta[pho1_index], event.pho_R9[pho1_index])[1]

            pho2Pt[0] = event.pho_pt[pho2_index]
            pho2eta[0] = event.pho_eta[pho2_index]
            pho2phi[0] = event.pho_phi[pho2_index]
            pho2R9[0] = event.pho_R9[pho2_index]
            pho2IetaIeta[0] = event.pho_sigmaIetaIeta[pho2_index]
            pho2IetaIeta55[0] = event.pho_full5x5_sigmaIetaIeta[pho2_index]
            pho2HOE[0] = event.pho_hadronicOverEm[pho2_index]
            pho2CIso[0] = event.pho_chargedHadronIso[pho2_index]
            pho2NIso[0] = event.pho_neutralHadronIso[pho2_index]
            pho2PIso[0] = pho2_phoIso
            pho2PIso_noCorr[0] = event.pho_photonIso[pho2_index]
            pho2scaleup[0] = event.pho_scaleup[pho2_index]
            pho2scaledn[0] = event.pho_scaledn[pho2_index]
            pho2smearup[0] = event.pho_smearup[pho2_index]
            pho2smeardn[0] = event.pho_smeardn[pho2_index]
            pho2ShowerShapeSys[0] = Sys.showerShapeUncVal(event.pho_eta[pho2_index], event.pho_R9[pho2_index])[1]

            var_dR_Za[0] = deltaR(Z_find.Eta(), Z_find.Phi(), ALP_find.Eta(), ALP_find.Phi())
            var_dR_g1g2[0] = dR_g1g2
            var_dR_g1Z[0] = dR_g1Z
            var_dEta_g1Z[0] = pho1_find.Eta() - Z_find.Eta()
            var_dPhi_g1Z[0] = pho1_find.Phi() - Z_find.Phi()
            var_PtaOverMa[0] = ALP_find.Pt()/ALP_find.M()
            var_PtaOverMh[0] = ALP_find.Pt()/H_find.M()
            var_Pta[0] = ALP_find.Pt()
            var_MhMa[0] = ALP_find.M()+H_find.M()
            var_MhMZ[0] = Z_find.M()+H_find.M()

            ################# photon SFs

            pho1_SFs = 1.0
            pho2_SFs = 1.0
            if isMC:
                for i in range(len(SFs)):
                    if (event.pho_pt[pho1_index] > pt_l[i] and event.pho_pt[pho1_index] < pt_r[i]) and (event.pho_eta[pho1_index] > eta_l[i] and event.pho_eta[pho1_index] < eta_r[i]):
                        pho1_SFs = SFs[i]
                        break

                for i in range(len(SFs)):
                    if (event.pho_pt[pho2_index] > pt_l[i] and event.pho_pt[pho2_index] < pt_r[i]) and (event.pho_eta[pho2_index] > eta_l[i] and event.pho_eta[pho2_index] < eta_r[i]):
                        pho2_SFs = SFs[i]
                        break

            pho1SFs[0] = pho1_SFs
            pho2SFs[0] = pho2_SFs

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



            ################# define cut flow #################
            cutFlow = {}

            cutdR_gl = (dR_l1g1 > 0.4) and (dR_l1g2 > 0.4) and (dR_l2g1 > 0.4) and (dR_l2g2 > 0.4)
            cutdR_gg = dR_g1g2 > 0.02
            cutH_m = (H_find.M()>118) and (H_find.M()<130)

            cutFlow['cut1'] = pho_passChaHadIso
            cutFlow['cut2'] = pho_passNeuHadIso
            cutFlow['cut3'] = cutdR_gl
            cutFlow['cut4'] = cutdR_gg
            cutFlow['cut5'] = pho_passHOverE
            cutFlow['cut6'] = pho_passIeIe
            cutFlow['cut7'] = passedPhoIso
            cutFlow['cut8'] = cutH_m

            passEleVeto[0] = pho_passEleVeto
            passIeIe[0] = pho_passIeIe
            passHOverE[0] = pho_passHOverE
            passChaHadIso[0] = pho_passChaHadIso
            passNeuHadIso[0] = pho_passNeuHadIso
            passPhoIso[0] = passedPhoIso
            passdR_gl[0] = cutdR_gl
            passdR_gg[0] = cutdR_gg
            passH_m[0] = cutH_m

            MVA_list = [event.pho_pt[pho1_index], event.pho_eta[pho1_index], event.pho_phi[pho1_index], event.pho_R9[pho1_index], event.pho_full5x5_sigmaIetaIeta[pho1_index] ,event.pho_pt[pho2_index], event.pho_eta[pho2_index], event.pho_phi[pho2_index], event.pho_R9[pho2_index], event.pho_full5x5_sigmaIetaIeta[pho2_index],ALP_calculatedPhotonIso_tmp, dR_g1Z, ALP_find.Pt(), Z_find.M()+H_find.M(), H_find.Pt() ]
            MVA_value = model.predict_proba(MVA_list)[:, 1]
            passBDT[0] = MVA_value>mvaCut
            Val_BDT[0] = MVA_value
            #######################################################################################################

            passedEvents.Fill()











file_out.Write()
file_out.Close()

sw.Stop()
print 'Real time: ' + str(round(sw.RealTime() / 60.0,2)) + ' minutes'
print 'CPU time:  ' + str(round(sw.CpuTime() / 60.0,2)) + ' minutes'
