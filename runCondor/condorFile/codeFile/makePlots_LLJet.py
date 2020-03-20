#! /usr/bin/env python

import argparse
parser = argparse.ArgumentParser(description="A simple ttree plotter")
parser.add_argument("-i", "--inputfiles", dest="inputfiles", default=["Sync_1031_2018_ttH_v2.root"], nargs='*', help="List of input ggNtuplizer files")
parser.add_argument("-o", "--outputfile", dest="outputfile", default="plots.root", help="Output file containing plots")
parser.add_argument("-m", "--maxevents", dest="maxevents", type=int, default=-1, help="Maximum number events to loop over")
parser.add_argument("-t", "--ttree", dest="ttree", default="Ana/passedEvents", help="TTree Name")
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
if (filename == "Sync_2016_SZ_mG_85485.root" or filename == "Sync_2016_SZ_mG_88366.root" ):
    weight = 140000.0*123.8/tchain.GetEntries()
if (filename == "Sync_2017_ggHZG_8000.root" ):
    weight = 35.9*14.31/tchain.GetEntries()
if (filename == "Sync_2016_ZJet.root" or filename == "Sync_2016_ZJet2.root"):
    weight = 83174000.0/tchain.GetEntries()
if (filename == "Sync_2016_ggmumu.root" ):
    weight = 262.62*600/tchain.GetEntries()
if (filename == "Sync_2016_ggelel.root" ):
    weight = 540.69*600/tchain.GetEntries()
weight = 1

print 'events weight: '+str(weight)
print 'events : '+str(tchain.GetEntries())


# Output file and any histograms we want
file_out = ROOT.TFile(args.outputfile, 'recreate')

# pass triger
h_n = ROOT.TH1D('nEvents', 'nEvents', 2, 0, 2)
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
H_twopho = ROOT.TH1D('H_twopho', 'H_twopho', 100, 80., 190)


l1_id_Ceta = ROOT.TH1D('l1_id_Ceta', 'l1_id_Ceta', 40, -20., 20.)
Z_Ceta = ROOT.TH1D('Z_Ceta', 'Z_Ceta', 100, 0, 500)
H_Ceta = ROOT.TH1D('H_Ceta', 'H_Ceta', 100, 80., 190)
ALP_Ceta = ROOT.TH1D('ALP_Ceta', 'ALP_Ceta', 100, 0., 30.)


l1_id_pho_veto = ROOT.TH1D('l1_id_pho_veto', 'l1_id_pho_veto', 40, -20., 20.)
Z_pho_veto = ROOT.TH1D('Z_pho_veto', 'Z_pho_veto', 100, 0, 500)
H_pho_veto = ROOT.TH1D('H_pho_veto', 'H_pho_veto', 100, 80., 190)
ALP_pho_veto = ROOT.TH1D('ALP_pho_veto', 'ALP_pho_veto', 100, 0., 30)
phoEB_IetaIeta = ROOT.TH1D('phoEB_IetaIeta', 'phoEB_IetaIeta', 100, 0., 0.08)
phoEE_IetaIeta = ROOT.TH1D('phoEE_IetaIeta', 'phoEE_IetaIeta', 100, 0., 0.08)

pho1Pt_EB = ROOT.TH1D('pho1Pt_EB', 'pho1Pt_EB', 100, 0, 200)
pho1R9_EB = ROOT.TH1D('pho1R9_EB', 'pho1R9_EB', 100, 0., 1.5)
pho1IetaIeta_EB = ROOT.TH1D('pho1IetaIeta_EB', 'pho1IetaIeta_EB', 100, 0., 0.08)
pho1IetaIeta55_EB = ROOT.TH1D('pho1IetaIeta55_EB', 'pho1IetaIeta55_EB', 100, 0., 0.08)
pho1HOE_EB = ROOT.TH1D('pho1HOE_EB', 'pho1HOE_EB', 100, 0., 0.4)
pho1CIso_EB = ROOT.TH1D('pho1CIso_EB', 'pho1CIso_EB', 100, 0., 20)
pho1NIso_EB = ROOT.TH1D('pho1NIso_EB', 'pho1NIso_EB', 100, 0., 10)
pho1PIso_EB = ROOT.TH1D('pho1PIso_EB', 'pho1PIso_EB', 100, 0., 40)

pho1Pt_EE = ROOT.TH1D('pho1Pt_EE', 'pho1Pt_EE', 100, 0, 200)
pho1R9_EE = ROOT.TH1D('pho1R9_EE', 'pho1R9_EE', 100, 0., 1.5)
pho1IetaIeta_EE = ROOT.TH1D('pho1IetaIeta_EE', 'pho1IetaIeta_EE', 100, 0., 0.08)
pho1IetaIeta55_EE = ROOT.TH1D('pho1IetaIeta55_EE', 'pho1IetaIeta55_EE', 100, 0., 0.08)
pho1HOE_EE = ROOT.TH1D('pho1HOE_EE', 'pho1HOE_EE', 100, 0., 0.4)
pho1CIso_EE = ROOT.TH1D('pho1CIso_EE', 'pho1CIso_EE', 100, 0., 20)
pho1NIso_EE = ROOT.TH1D('pho1NIso_EE', 'pho1NIso_EE', 100, 0., 10)
pho1PIso_EE = ROOT.TH1D('pho1PIso_EE', 'pho1PIso_EE', 100, 0., 40)


Z_pho_veto_IeIe = ROOT.TH1D('Z_pho_veto_IeIe', 'Z_pho_veto_IeIe', 100, 0, 500)
H_pho_veto_IeIe = ROOT.TH1D('H_pho_veto_IeIe', 'H_pho_veto_IeIe', 100, 80., 190)
ALP_pho_veto_IeIe = ROOT.TH1D('ALP_pho_veto_IeIe', 'ALP_pho_veto_IeIe', 100, 0., 30)
phoEB_IetaIeta_cut = ROOT.TH1D('phoEB_IetaIeta_cut', 'phoEB_IetaIeta_cut', 100, 0., 0.08)
phoEE_IetaIeta_cut = ROOT.TH1D('phoEE_IetaIeta_cut', 'phoEE_IetaIeta_cut', 100, 0., 0.08)

Z_pho_veto_IeIe_HOE = ROOT.TH1D('Z_pho_veto_IeIe_HOE', 'Z_pho_veto_IeIe_HOE', 100, 0, 500)
H_pho_veto_IeIe_HOE = ROOT.TH1D('H_pho_veto_IeIe_HOE', 'H_pho_veto_IeIe_HOE', 100, 80., 190)
ALP_pho_veto_IeIe_HOE = ROOT.TH1D('ALP_pho_veto_IeIe_HOE', 'ALP_pho_veto_IeIe_HOE', 100, 0., 30)

Z_pho_veto_IeIe_HOE_CIso = ROOT.TH1D('Z_pho_veto_IeIe_HOE_CIso', 'Z_pho_veto_IeIe_HOE_CIso', 100, 0, 500)
H_pho_veto_IeIe_HOE_CIso = ROOT.TH1D('H_pho_veto_IeIe_HOE_CIso', 'H_pho_veto_IeIe_HOE_CIso', 100, 80., 190)
ALP_pho_veto_IeIe_HOE_CIso = ROOT.TH1D('ALP_pho_veto_IeIe_HOE_CIso', 'ALP_pho_veto_IeIe_HOE_CIso', 100, 0., 30)

Z_pho_veto_IeIe_HOE_CIso_NIso = ROOT.TH1D('Z_pho_veto_IeIe_HOE_CIso_NIso', 'Z_pho_veto_IeIe_HOE_CIso_NIso', 100, 0, 500)
H_pho_veto_IeIe_HOE_CIso_NIso = ROOT.TH1D('H_pho_veto_IeIe_HOE_CIso_NIso', 'H_pho_veto_IeIe_HOE_CIso_NIso', 100, 80., 190)
ALP_pho_veto_IeIe_HOE_CIso_NIso = ROOT.TH1D('ALP_pho_veto_IeIe_HOE_CIso_NIso', 'ALP_pho_veto_IeIe_HOE_CIso_NIso', 100, 0., 30)

Z_pho_veto_IeIe_HOE_CIso_NIso_PIso = ROOT.TH1D('Z_pho_veto_IeIe_HOE_CIso_NIso_PIso', 'Z_pho_veto_IeIe_HOE_CIso_NIso_PIso', 100, 0, 500)
H_pho_veto_IeIe_HOE_CIso_NIso_PIso = ROOT.TH1D('H_pho_veto_IeIe_HOE_CIso_NIso_PIso', 'H_pho_veto_IeIe_HOE_CIso_NIso_PIso', 100, 80., 190)
ALP_pho_veto_IeIe_HOE_CIso_NIso_PIso = ROOT.TH1D('ALP_pho_veto_IeIe_HOE_CIso_NIso_PIso', 'ALP_pho_veto_IeIe_HOE_CIso_NIso_PIso', 100, 0., 30)

Z_dR = ROOT.TH1D('Z_dR', 'Z_dR', 100, 0, 500)
H_dR = ROOT.TH1D('H_dR', 'H_dR', 100, 80., 190)
ALP_dR = ROOT.TH1D('ALP_dR', 'ALP_dR', 100, 0., 30)

Z_dR_pho = ROOT.TH1D('Z_dR_pho', 'Z_dR_pho', 100, 0, 500)
H_dR_pho = ROOT.TH1D('H_dR_pho', 'H_dR_pho', 100, 80., 190)
ALP_dR_pho = ROOT.TH1D('ALP_dR_pho', 'ALP_dR_pho', 100, 0., 30)
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

H_twopho.SetStats(1)

l1_id_Ceta.SetStats(1)
Z_Ceta.SetStats(1)
H_Ceta.SetStats(1)
ALP_Ceta.SetStats(1)

Z_pho_veto.SetStats(1)
H_pho_veto.SetStats(1)
ALP_pho_veto.SetStats(1)
phoEB_IetaIeta.SetStats(1)
phoEE_IetaIeta.SetStats(1)

Z_pho_veto_IeIe.SetStats(1)
H_pho_veto_IeIe.SetStats(1)
ALP_pho_veto_IeIe.SetStats(1)
phoEB_IetaIeta_cut.SetStats(1)
phoEE_IetaIeta_cut.SetStats(1)

Z_pho_veto_IeIe_HOE.SetStats(1)
H_pho_veto_IeIe_HOE.SetStats(1)
ALP_pho_veto_IeIe_HOE.SetStats(1)

Z_pho_veto_IeIe_HOE_CIso.SetStats(1)
H_pho_veto_IeIe_HOE_CIso.SetStats(1)
ALP_pho_veto_IeIe_HOE_CIso.SetStats(1)

Z_pho_veto_IeIe_HOE_CIso_NIso.SetStats(1)
H_pho_veto_IeIe_HOE_CIso_NIso.SetStats(1)
ALP_pho_veto_IeIe_HOE_CIso_NIso.SetStats(1)

Z_pho_veto_IeIe_HOE_CIso_NIso_PIso.SetStats(1)
H_pho_veto_IeIe_HOE_CIso_NIso_PIso.SetStats(1)
ALP_pho_veto_IeIe_HOE_CIso_NIso_PIso.SetStats(1)

Z_dR.SetStats(1)
H_dR.SetStats(1)
ALP_dR.SetStats(1)

Z_dR_pho.SetStats(1)
H_dR_pho.SetStats(1)
ALP_dR_pho.SetStats(1)
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

pho_matchID = array('i',[0])

passedEvents = ROOT.TTree("passedEvents","passedEvents")

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


passedEvents.Branch("pho_matchID",pho_matchID,"pho_matchID/I")




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
    pho_passIeIe = True
    pho_passHOverE = True
    pho_passChaHadIso = True
    pho_passNeuHadIso = True
    passedPhoIso = True


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
        if (event.pho_hasPixelSeed[i] == 1): continue
        if (event.pho_pt[i] > pho1_maxPt):
            pho1_maxPt = event.pho_pt[i]
            pho1_index = i
            foundpho1 = True

    if (not (foundpho1)): continue

    pho1_find = ROOT.TLorentzVector()

    #pho1_find.SetPtEtaPhiE(event.pho_pt[pho1_index], event.pho_eta[pho1_index], event.pho_phi[pho1_index], event.pho_pt[pho1_index] * np.cosh(event.pho_eta[pho1_index]))
    #pho2_find.SetPtEtaPhiE(event.pho_pt[pho2_index], event.pho_eta[pho2_index], event.pho_phi[pho2_index], event.pho_pt[pho2_index] * np.cosh(event.pho_eta[pho2_index]))

    pho1_find.SetPtEtaPhiM(event.pho_pt[pho1_index], event.pho_eta[pho1_index], event.pho_phi[pho1_index], 0.0)

#######################################################################################################

    # Higgs Candidates
#######################################################################################################
    H_find = ROOT.TLorentzVector()
    H_find = (Z_find + pho1_find)
#######################################################################################################
    H_twopho.Fill(H_find.M())
    # Photon Cuts
#######################################################################################################

    if (((abs(event.pho_eta[pho1_index]) >1.4442) and (abs(event.pho_eta[pho1_index]) < 1.566)) or (abs(event.pho_eta[pho1_index]) >2.5)): continue

    if (event.pho_EleVote[pho1_index] == 0 ): pho_passEleVeto = False
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

    # no Cuts
    l1_id_Ceta.Fill(event.lep_id[lep_leadindex[0]])
    Z_Ceta.Fill(Z_find.M())
    H_Ceta.Fill(H_find.M())

    if (not pho_passEleVeto): continue
    l1_id_pho_veto.Fill(event.lep_id[lep_leadindex[0]])
    Z_pho_veto.Fill(Z_find.M())
    H_pho_veto.Fill(H_find.M())

    if (abs(event.pho_eta[pho1_index]) < 1.4442):
        phoEB_IetaIeta.Fill(event.pho_full5x5_sigmaIetaIeta[pho1_index])

        pho1Pt_EB.Fill(event.pho_pt[pho1_index])
        pho1R9_EB.Fill(event.pho_R9[pho1_index])
        pho1IetaIeta_EB.Fill(event.pho_sigmaIetaIeta[pho1_index])
        pho1IetaIeta55_EB.Fill(event.pho_full5x5_sigmaIetaIeta[pho1_index])
        pho1HOE_EB.Fill(event.pho_hadronicOverEm[pho1_index])
        pho1CIso_EB.Fill(event.pho_chargedHadronIso[pho1_index])
        pho1NIso_EB.Fill(event.pho_neutralHadronIso[pho1_index])
        pho1PIso_EB.Fill(event.pho_photonIso[pho1_index])
    else:
        phoEE_IetaIeta.Fill(event.pho_full5x5_sigmaIetaIeta[pho1_index])

        pho1Pt_EE.Fill(event.pho_pt[pho1_index])
        pho1R9_EE.Fill(event.pho_R9[pho1_index])
        pho1IetaIeta_EE.Fill(event.pho_sigmaIetaIeta[pho1_index])
        pho1IetaIeta55_EE.Fill(event.pho_full5x5_sigmaIetaIeta[pho1_index])
        pho1HOE_EE.Fill(event.pho_hadronicOverEm[pho1_index])
        pho1CIso_EE.Fill(event.pho_chargedHadronIso[pho1_index])
        pho1NIso_EE.Fill(event.pho_neutralHadronIso[pho1_index])
        pho1PIso_EE.Fill(event.pho_photonIso[pho1_index])



    if (not pho_passIeIe): continue
    Z_pho_veto_IeIe.Fill(Z_find.M())
    H_pho_veto_IeIe.Fill(H_find.M())
    if (abs(event.pho_eta[pho1_index]) < 1.4442):
        phoEB_IetaIeta_cut.Fill(event.pho_full5x5_sigmaIetaIeta[pho1_index])
    else:
        phoEE_IetaIeta_cut.Fill(event.pho_full5x5_sigmaIetaIeta[pho1_index])

    if (not pho_passHOverE): continue
    Z_pho_veto_IeIe_HOE.Fill(Z_find.M())
    H_pho_veto_IeIe_HOE.Fill(H_find.M())

    if (not pho_passChaHadIso): continue
    Z_pho_veto_IeIe_HOE_CIso.Fill(Z_find.M())
    H_pho_veto_IeIe_HOE_CIso.Fill(H_find.M())

    if (not pho_passNeuHadIso): continue
    Z_pho_veto_IeIe_HOE_CIso_NIso.Fill(Z_find.M())
    H_pho_veto_IeIe_HOE_CIso_NIso.Fill(H_find.M())

    #if (not passedPhoIso): continue
    Z_pho_veto_IeIe_HOE_CIso_NIso_PIso.Fill(Z_find.M())
    H_pho_veto_IeIe_HOE_CIso_NIso_PIso.Fill(H_find.M())

    dR_l1g1 = deltaR(l1_find.Eta(), l1_find.Phi(), pho1_find.Eta(), pho1_find.Phi())
    dR_l2g1 = deltaR(l2_find.Eta(), l2_find.Phi(), pho1_find.Eta(), pho1_find.Phi())

    if (dR_l1g1 < 0.4): continue
    if (dR_l2g1 < 0.4): continue

    Z_dR.Fill(Z_find.M())
    H_dR.Fill(H_find.M())

    Z_dR_pho.Fill(Z_find.M())
    H_dR_pho.Fill(H_find.M())

#######################################################################################################
    # photon match
    dR_match = 0.
    dR_min = 9999.
    index_match = 0
    for i in range(event.GENpho_pt.size()):
        dR_match = deltaR(pho1_find.Eta(),pho1_find.Phi(),event.GENpho_eta[i],event.GENpho_phi[i])
        if (dR_match < dR_min):
            dR_min = dR_match
            index_match = i


#######################################################################################################
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
    H_pt[0] = H_find.Pt()

    pho_matchID[0] = event.GENpho_id[index_match]
    passedEvents.Fill()











file_out.Write()
file_out.Close()

sw.Stop()
print 'Real time: ' + str(round(sw.RealTime() / 60.0,2)) + ' minutes'
print 'CPU time:  ' + str(round(sw.CpuTime() / 60.0,2)) + ' minutes'
