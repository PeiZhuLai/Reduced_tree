#! /usr/bin/env python

import argparse
parser = argparse.ArgumentParser(description="A simple ttree plotter")
parser.add_argument("-i", "--inputfiles", dest="inputfiles", default=["Sync_1031_2018_ttH_v2.root"], nargs='*', help="List of input ggNtuplizer files")
parser.add_argument("-o", "--outputfile", dest="outputfile", default="plots.root", help="Output file containing plots")
parser.add_argument("-m", "--maxevents", dest="maxevents", type=int, default=-1, help="Maximum number events to loop over")
parser.add_argument("-t", "--ttree", dest="ttree", default="Ana/passedEvents", help="TTree Name")
parser.add_argument("-xs", "--cross_section", dest="cross_section", default="1.0", help="the cross section of samples")
parser.add_argument("-L", "--Lumi", dest="Lumi", default="35.9", help="the luminosities to normalized")
args = parser.parse_args()

import numpy as np
import ROOT
import os

###########################
from deltaR import *
from array import array


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

pho2eta = array('f',[0.])
pho2Pt = array('f',[0.])
pho2R9 = array('f',[0.])
pho2IetaIeta = array('f',[0.])
pho2IetaIeta55 = array('f',[0.])
pho2HOE = array('f',[0.])
pho2CIso = array('f',[0.])
pho2NIso = array('f',[0.])
pho2PIso = array('f',[0.])


# photon cut
H_twopho = array('f',[-1.])

Z_Ceta = array('f',[-1.])
H_Ceta = array('f',[-1.])
ALP_Ceta = array('f',[-1.])

Z_pho_veto = array('f',[-1.])
H_pho_veto = array('f',[-1.])
ALP_pho_veto = array('f',[-1.])

Z_pho_veto_mva = array('f',[-1.])
H_pho_veto_mva = array('f',[-1.])
ALP_pho_veto_mva = array('f',[-1.])

Z_pho_veto_IeIe_HOE = array('f',[-1.])
H_pho_veto_IeIe_HOE = array('f',[-1.])
ALP_pho_veto_IeIe_HOE = array('f',[-1.])

Z_pho_veto_IeIe_HOE_CIso = array('f',[-1.])
H_pho_veto_IeIe_HOE_CIso = array('f',[-1.])
ALP_pho_veto_IeIe_HOE_CIso = array('f',[-1.])

Z_pho_veto_IeIe_HOE_CIso_NIso = array('f',[-1.])
H_pho_veto_IeIe_HOE_CIso_NIso = array('f',[-1.])
ALP_pho_veto_IeIe_HOE_CIso_NIso = array('f',[-1.])

Z_pho_veto_IeIe_HOE_CIso_NIso_PIso = array('f',[-1.])
H_pho_veto_IeIe_HOE_CIso_NIso_PIso = array('f',[-1.])
ALP_pho_veto_IeIe_HOE_CIso_NIso_PIso = array('f',[-1.])

Z_dR = array('f',[-1.])
H_dR = array('f',[-1.])
ALP_dR = array('f',[-1.])

Z_dR_cut1 = array('f',[-1.])
H_dR_cut1 = array('f',[-1.])
ALP_dR_cut1 = array('f',[-1.])

Z_dR_cut1_Cmh = array('f',[-1.])
H_dR_cut1_Cmh = array('f',[-1.])
ALP_dR_cut1_Cmh = array('f',[-1.])


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

pho2_pt = array('f',[0.])
pho2_eta = array('f',[0.])
pho2_phi = array('f',[0.])
pho2_mva = array('f',[0.])

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


passedEvents.Branch("pho1eta",pho1eta,"pho1eta/F")
passedEvents.Branch("pho1Pt",pho1Pt,"pho1Pt/F")
passedEvents.Branch("pho1R9",pho1R9,"pho1R9/F")
passedEvents.Branch("pho1IetaIeta",pho1IetaIeta,"pho1IetaIeta/F")
passedEvents.Branch("pho1IetaIeta55",pho1IetaIeta55,"pho1IetaIeta55/F")
passedEvents.Branch("pho1HOE",pho1HOE,"pho1HOE/F")
passedEvents.Branch("pho1CIso",pho1CIso,"pho1CIso/F")
passedEvents.Branch("pho1NIso",pho1NIso,"pho1NIso/F")
passedEvents.Branch("pho1PIso",pho1PIso,"pho1PIso/F")


passedEvents.Branch("pho2eta",pho2eta,"pho2eta/F")
passedEvents.Branch("pho2Pt",pho2Pt,"pho2Pt/F")
passedEvents.Branch("pho2R9",pho2R9,"pho2R9/F")
passedEvents.Branch("pho2IetaIeta",pho2IetaIeta,"pho2IetaIeta/F")
passedEvents.Branch("pho2IetaIeta55",pho2IetaIeta55,"pho2IetaIeta55/F")
passedEvents.Branch("pho2HOE",pho2HOE,"pho2HOE/F")
passedEvents.Branch("pho2CIso",pho2CIso,"pho2CIso/F")
passedEvents.Branch("pho2NIso",pho2NIso,"pho2NIso/F")
passedEvents.Branch("pho2PIso",pho2PIso,"pho2PIso/F")




passedEvents.Branch("H_twopho",H_twopho,"H_twopho/F")

passedEvents.Branch("Z_Ceta",Z_Ceta,"Z_Ceta/F")
passedEvents.Branch("H_Ceta",H_Ceta,"H_Ceta/F")
passedEvents.Branch("ALP_Ceta",ALP_Ceta,"ALP_Ceta/F")

passedEvents.Branch("Z_pho_veto",Z_pho_veto,"Z_pho_veto/F")
passedEvents.Branch("H_pho_veto",H_pho_veto,"H_pho_veto/F")
passedEvents.Branch("ALP_pho_veto",ALP_pho_veto,"ALP_pho_veto/F")

passedEvents.Branch("Z_pho_veto_mva",Z_pho_veto_mva,"Z_pho_veto_mva/F")
passedEvents.Branch("H_pho_veto_mva",H_pho_veto_mva,"H_pho_veto_mva/F")
passedEvents.Branch("ALP_pho_veto_mva",ALP_pho_veto_mva,"ALP_pho_veto_mva/F")

passedEvents.Branch("Z_pho_veto_IeIe_HOE",Z_pho_veto_IeIe_HOE,"Z_pho_veto_IeIe_HOE/F")
passedEvents.Branch("H_pho_veto_IeIe_HOE",H_pho_veto_IeIe_HOE,"H_pho_veto_IeIe_HOE/F")
passedEvents.Branch("ALP_pho_veto_IeIe_HOE",ALP_pho_veto_IeIe_HOE,"ALP_pho_veto_IeIe_HOE/F")

passedEvents.Branch("Z_pho_veto_IeIe_HOE_CIso",Z_pho_veto_IeIe_HOE_CIso,"Z_pho_veto_IeIe_HOE_CIso/F")
passedEvents.Branch("H_pho_veto_IeIe_HOE_CIso",H_pho_veto_IeIe_HOE_CIso,"H_pho_veto_IeIe_HOE_CIso/F")
passedEvents.Branch("ALP_pho_veto_IeIe_HOE_CIso",ALP_pho_veto_IeIe_HOE_CIso,"ALP_pho_veto_IeIe_HOE_CIso/F")

passedEvents.Branch("Z_pho_veto_IeIe_HOE_CIso_NIso",Z_pho_veto_IeIe_HOE_CIso_NIso,"Z_pho_veto_IeIe_HOE_CIso_NIso/F")
passedEvents.Branch("H_pho_veto_IeIe_HOE_CIso_NIso",H_pho_veto_IeIe_HOE_CIso_NIso,"H_pho_veto_IeIe_HOE_CIso_NIso/F")
passedEvents.Branch("ALP_pho_veto_IeIe_HOE_CIso_NIso",ALP_pho_veto_IeIe_HOE_CIso_NIso,"ALP_pho_veto_IeIe_HOE_CIso_NIso/F")

passedEvents.Branch("Z_pho_veto_IeIe_HOE_CIso_NIso_PIso",Z_pho_veto_IeIe_HOE_CIso_NIso_PIso,"Z_pho_veto_IeIe_HOE_CIso_NIso_PIso/F")
passedEvents.Branch("H_pho_veto_IeIe_HOE_CIso_NIso_PIso",H_pho_veto_IeIe_HOE_CIso_NIso_PIso,"H_pho_veto_IeIe_HOE_CIso_NIso_PIso/F")
passedEvents.Branch("ALP_pho_veto_IeIe_HOE_CIso_NIso_PIso",ALP_pho_veto_IeIe_HOE_CIso_NIso_PIso,"ALP_pho_veto_IeIe_HOE_CIso_NIso_PIso/F")

passedEvents.Branch("Z_dR",Z_dR,"Z_dR/F")
passedEvents.Branch("H_dR",H_dR,"H_dR/F")
passedEvents.Branch("ALP_dR",ALP_dR,"ALP_dR/F")

passedEvents.Branch("Z_dR_cut1",Z_dR_cut1,"Z_dR_cut1/F")
passedEvents.Branch("H_dR_cut1",H_dR_cut1,"H_dR_cut1/F")
passedEvents.Branch("ALP_dR_cut1",ALP_dR_cut1,"ALP_dR_cut1/F")

passedEvents.Branch("Z_dR_cut1_Cmh",Z_dR_cut1_Cmh,"Z_dR_cut1_Cmh/F")
passedEvents.Branch("H_dR_cut1_Cmh",H_dR_cut1_Cmh,"H_dR_cut1_Cmh/F")
passedEvents.Branch("ALP_dR_cut1_Cmh",ALP_dR_cut1_Cmh,"ALP_dR_cut1_Cmh/F")


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

passedEvents.Branch("pho2_pt",pho2_pt,"pho2_pt/F")
passedEvents.Branch("pho2_eta",pho2_eta,"pho2_eta/F")
passedEvents.Branch("pho2_phi",pho2_phi,"pho2_phi/F")
passedEvents.Branch("pho2_mva",pho2_mva,"pho2_mva/F")

passedEvents.Branch("Z_m",Z_m,"Z_m/F")
passedEvents.Branch("H_m",H_m,"H_m/F")
passedEvents.Branch("ALP_m",ALP_m,"ALP_m/F")
passedEvents.Branch("H_pt",H_pt,"H_pt/F")
passedEvents.Branch("dR_pho",dR_pho,"dR_pho/F")


passedEvents.Branch("event_weight",event_weight,"event_weight/F")



#Loop over all the events in the input ntuple
for ievent,event in enumerate(tchain):#, start=650000):
    if ievent > args.maxevents and args.maxevents != -1: break
    #if ievent == 100000: break
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
    pho_passMVA = True


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


    npho.Fill(event.pho_pt.size())
    if (event.pho_pt.size() < 1): continue


    for i in range(event.pho_pt.size()):
        if (event.pho_pt[i] < 15.): continue
        if (event.pho_hasPixelSeed[i] == 1): continue
        if (event.pho_pt[i] > pho1_maxPt):
            pho1_maxPt = event.pho_pt[i]
            pho1_index = i
            foundpho1 = True

    '''
    for j in range(event.pho_pt.size()):
        if (event.pho_hasPixelSeed[j] == 1): continue
        if j == pho1_index: continue
        if (event.pho_pt[j] > pho2_maxPt):
            pho2_maxPt = event.pho_pt[j]
            pho2_index = j
            foundpho2 = True
    '''

    if (foundpho1):

        Run[0] = event.Run
        LumiSect[0] = event.LumiSect
        Event[0] = event.Event
        event_weight[0] = weight
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


        pho2eta[0] = -99.0
        pho2Pt[0] = -99.0
        pho2R9[0] = -99.0
        pho2IetaIeta[0] = -99.0
        pho2IetaIeta55[0] = -99.0
        pho2HOE[0] = -99.0
        pho2CIso[0] = -99.0
        pho2NIso[0] = -99.0
        pho2PIso[0] = -99.0

        Z_pho_veto_mva[0] = -99.0
        H_pho_veto_mva[0] = -99.0
        ALP_pho_veto_mva[0] = -99.0

        Z_pho_veto_IeIe_HOE[0] = -99.0
        H_pho_veto_IeIe_HOE[0] = -99.0
        ALP_pho_veto_IeIe_HOE[0] = -99.0

        Z_pho_veto_IeIe_HOE_CIso[0] = -99.0
        H_pho_veto_IeIe_HOE_CIso[0] = -99.0
        ALP_pho_veto_IeIe_HOE_CIso[0] = -99.0

        Z_pho_veto_IeIe_HOE_CIso_NIso[0] = -99.0
        H_pho_veto_IeIe_HOE_CIso_NIso[0] = -99.0
        ALP_pho_veto_IeIe_HOE_CIso_NIso[0] = -99.0

        Z_pho_veto_IeIe_HOE_CIso_NIso_PIso[0] = -99.0
        H_pho_veto_IeIe_HOE_CIso_NIso_PIso[0] = -99.0
        ALP_pho_veto_IeIe_HOE_CIso_NIso_PIso[0] = -99.0

        Z_dR[0] = -99.0
        H_dR[0] = -99.0
        ALP_dR[0] = -99.0

        Z_dR_cut1[0] = -99.0
        H_dR_cut1[0] = -99.0
        ALP_dR_cut1[0] = -99.0

        Z_dR_cut1_Cmh[0] = -99.0
        H_dR_cut1_Cmh[0] = -99.0
        ALP_dR_cut1_Cmh[0] = -99.0

        l1_pt[0] = -99.0
        l2_pt[0] = -99.0
        l1_eta[0] = -99.0
        l2_eta[0] = -99.0
        l1_phi[0] = -99.0
        l2_phi[0] = -99.0
        l1_id[0] = -99
        l2_id[0] = -99

        pho1_pt[0] = -99.0
        pho1_eta[0] = -99.0
        pho1_phi[0] = -99.0
        pho1_mva[0] = -99.0

        pho2_pt[0] = -99.0
        pho2_eta[0] = -99.0
        pho2_phi[0] = -99.0
        pho2_mva[0] = -99.0


        Z_m[0] = -99.0
        H_m[0] = -99.0
        ALP_m[0] = -99.0
        dR_pho[0] = -99.0
        H_pt[0] = -99.0

    ################################################################################################

        pho1_find = ROOT.TLorentzVector()
        #pho2_find = ROOT.TLorentzVector()

        #pho1_find.SetPtEtaPhiE(event.pho_pt[pho1_index], event.pho_eta[pho1_index], event.pho_phi[pho1_index], event.pho_pt[pho1_index] * np.cosh(event.pho_eta[pho1_index]))
        #pho2_find.SetPtEtaPhiE(event.pho_pt[pho2_index], event.pho_eta[pho2_index], event.pho_phi[pho2_index], event.pho_pt[pho2_index] * np.cosh(event.pho_eta[pho2_index]))

        pho1_find.SetPtEtaPhiM(event.pho_pt[pho1_index], event.pho_eta[pho1_index], event.pho_phi[pho1_index], 0.0)
        #pho2_find.SetPtEtaPhiM(event.pho_pt[pho2_index], event.pho_eta[pho2_index], event.pho_phi[pho2_index], 0.0)

        ALP_find = ROOT.TLorentzVector()
        ALP_find = (pho1_find)# + pho2_find)
    #######################################################################################################

        # Higgs Candidates
    #######################################################################################################
        H_find = ROOT.TLorentzVector()
        H_find = (Z_find + ALP_find)
    #######################################################################################################
        H_twopho[0] = H_find.M()
        # Photon Cuts
    #######################################################################################################

        if (((abs(event.pho_eta[pho1_index]) > 1.566) and (abs(event.pho_eta[pho1_index]) < 2.5)) or (abs(event.pho_eta[pho1_index]) < 1.4442)): # and (((abs(event.pho_eta[pho2_index]) > 1.566) and (abs(event.pho_eta[pho2_index]) < 2.5)) or (abs(event.pho_eta[pho2_index]) < 1.4442)):
            #if (((abs(event.pho_eta[pho2_index]) >1.4442) and (abs(event.pho_eta[pho2_index]) < 1.566)) or (abs(event.pho_eta[pho2_index]) >2.5)): continue

            if (event.pho_EleVote[pho1_index] == 0 ): pho_passEleVeto = False
            #if (event.pho_EleVote[pho2_index] == 0 ): pho_passEleVeto = False
            if (event.pho_mva90[pho1_index] == 0): pho_passMVA = False

            # no Cuts
            Z_Ceta[0] = Z_find.M()
            H_Ceta[0] = H_find.M()
            ALP_Ceta[0] = ALP_find.M()

            #H_Ceta.Fill(H_find.M())

            if pho_passEleVeto:

                Z_pho_veto[0] = Z_find.M()
                H_pho_veto[0] = H_find.M()
                ALP_pho_veto[0] = ALP_find.M()

                if pho_passMVA:
                    Z_pho_veto_mva[0] = Z_find.M()
                    H_pho_veto_mva[0] = H_find.M()
                    ALP_pho_veto_mva[0] = ALP_find.M()


                    dR_l1g1 = deltaR(l1_find.Eta(), l1_find.Phi(), pho1_find.Eta(), pho1_find.Phi())
                    dR_l2g1 = deltaR(l2_find.Eta(), l2_find.Phi(), pho1_find.Eta(), pho1_find.Phi())

                    if (dR_l1g1 > 0.4) and (dR_l2g1 > 0.4) :

                        Z_dR[0] = Z_find.M()
                        H_dR[0] = H_find.M()
                        ALP_dR[0] = ALP_find.M()

                        if H_find.M() < 180. and H_find.M() > 100.:
                            if pho1_find.Pt() / H_find.M() > 15.0/110.0:
                                if H_find.M() + Z_find.M() > 185.0:
                                    Z_dR_cut1[0] = Z_find.M()
                                    H_dR_cut1[0] = H_find.M()
                                    ALP_dR_cut1[0] = ALP_find.M()

                                    # Fill Tree
                                    l1_pt[0] = event.lepFSR_pt[lep_leadindex[0]]
                                    l2_pt[0] = event.lepFSR_pt[lep_leadindex[1]]
                                    l1_eta[0] = event.lepFSR_eta[lep_leadindex[0]]
                                    l2_eta[0] = event.lepFSR_eta[lep_leadindex[1]]
                                    l1_phi[0] = event.lepFSR_phi[lep_leadindex[0]]
                                    l2_phi[0] = event.lepFSR_phi[lep_leadindex[1]]
                                    l1_id[0] = event.lep_id[lep_leadindex[0]]
                                    l2_id[0] = event.lep_id[lep_leadindex[1]]

                                    pho1_pt[0] = event.pho_pt[pho1_index]
                                    pho1_eta[0] = event.pho_eta[pho1_index]
                                    pho1_phi[0] = event.pho_phi[pho1_index]
                                    pho1_mva[0] = event.pho_mva[pho1_index]


                                    Z_m[0] = Z_find.M()
                                    H_m[0] = H_find.M()
                                    ALP_m[0] = ALP_find.M()
                                    H_pt[0] = H_find.Pt()
                                            # Higgs mass cut
                                        # dR(pho1 , pho2)
                                    # dR(lepton, photon)
                                # photon isolation
                            # photon neutralHadronIso
                        # photon chargedHadronIso
                    # photon HOverE
                # photon sigmaIetaIeta
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
