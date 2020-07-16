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

accetpance_lep = ROOT.TH1D('accetpance_lep', 'accetpance_lep', 3, 0, 3)
accetpance_lep.SetStats(1)

Pt_pho1 = ROOT.TH1D('Pt_pho1', 'Pt_pho1', 100, 0, 200)
Pt_pho2 = ROOT.TH1D('Pt_pho2', 'Pt_pho2', 100, 0, 25)
Pt_l1 = ROOT.TH1D('Pt_l1', 'Pt_l1', 100, 0, 200)
Pt_l2 = ROOT.TH1D('Pt_l2', 'Pt_l2', 100, 0, 200)

Pt_pho1_Peta = ROOT.TH1D('Pt_pho1_Peta', 'Pt_pho1_Peta', 100, 0, 200)
Pt_pho2_Peta = ROOT.TH1D('Pt_pho2_Peta', 'Pt_pho2_Peta', 100, 0, 50)
Pt_l1_Peta = ROOT.TH1D('Pt_l1_Peta', 'Pt_l1_Peta', 100, 0, 200)
Pt_l2_Peta = ROOT.TH1D('Pt_l2_Peta', 'Pt_l2_Peta', 100, 0, 200)

Pt_pho1_Peta_Leta = ROOT.TH1D('Pt_pho1_Peta_Leta', 'Pt_pho1_Peta_Leta', 100, 0, 200)
Pt_pho2_Peta_Leta = ROOT.TH1D('Pt_pho2_Peta_Leta', 'Pt_pho2_Peta_Leta', 100, 0, 50)
Pt_l1_Peta_Leta = ROOT.TH1D('Pt_l1_Peta_Leta', 'Pt_l1_Peta_Leta', 100, 0, 200)
Pt_l2_Peta_Leta = ROOT.TH1D('Pt_l2_Peta_Leta', 'Pt_l2_Peta_Leta', 100, 0, 200)

Pt_pho1_Peta_Leta_Ppt = ROOT.TH1D('Pt_pho1_Peta_Leta_Ppt', 'Pt_pho1_Peta_Leta_Ppt', 100, 0, 200)
Pt_pho2_Peta_Leta_Ppt = ROOT.TH1D('Pt_pho2_Peta_Leta_Ppt', 'Pt_pho2_Peta_Leta_Ppt', 100, 0, 50)
Pt_l1_Peta_Leta_Ppt = ROOT.TH1D('Pt_l1_Peta_Leta_Ppt', 'Pt_l1_Peta_Leta_Ppt', 100, 0, 200)
Pt_l2_Peta_Leta_Ppt = ROOT.TH1D('Pt_l2_Peta_Leta_Ppt', 'Pt_l2_Peta_Leta_Ppt', 100, 0, 200)

Pt_pho1.SetStats(1)
Pt_pho2.SetStats(1)
Pt_l1.SetStats(1)
Pt_l2.SetStats(1)

Pt_pho1_Peta.SetStats(1)
Pt_pho2_Peta.SetStats(1)
Pt_l1_Peta.SetStats(1)
Pt_l2_Peta.SetStats(1)

Pt_pho1_Peta_Leta.SetStats(1)
Pt_pho2_Peta_Leta.SetStats(1)
Pt_l1_Peta_Leta.SetStats(1)
Pt_l2_Peta_Leta.SetStats(1)

Pt_pho1_Peta_Leta_Ppt.SetStats(1)
Pt_pho2_Peta_Leta_Ppt.SetStats(1)
Pt_l1_Peta_Leta_Ppt.SetStats(1)
Pt_l2_Peta_Leta_Ppt.SetStats(1)

# delta eta as a function of Et
#############################################
x_min = 0.
x_max = 100.
pho_dEta = ROOT.TH1D('pho_dEta', 'pho_dEta', int(x_max-x_min), x_min, x_max)
pho_dEta_2D = ROOT.TH2D('pho_dEta_2D', 'pho_dEta_2D',int(x_max-x_min), x_min, x_max,100,0,5)
data_dEta = [0.]*int(x_max-x_min)
data_N = [0.]*int(x_max-x_min)
pho_dEtadPhi = ROOT.TH2D('pho_dEtadPhi', 'pho_dEtadPhi',6, 0, 0.1,6,0,0.1)
#############################################

# cut varibles
#############################################
cut_pt = 10.
cut_eta = 2.5
#############################################

# Tree
l1_pt = array('f',[0.])
l1_eta = array('f',[0.])
l1_phi = array('f',[0.])
l1_mass = array('f',[0])



l2_pt = array('f',[0.])
l2_eta = array('f',[0.])
l2_phi = array('f',[0.])
l2_mass = array('f',[0])

pho1_pt = array('f',[0.])
pho1_eta = array('f',[0.])
pho1_phi = array('f',[0.])

pho2_pt = array('f',[0.])
pho2_eta = array('f',[0.])
pho2_phi = array('f',[0.])

Z_m = array('f',[0.])
H_m = array('f',[0.])
ALP_m = array('f',[0.])
ALP_Et = array('f',[0.])

dR_pho = array('f',[0.])
dEta_pho = array('f',[0.])
dPhi_pho = array('f',[0.])

passedEvents = ROOT.TTree("passedEvents","passedEvents")

passedEvents.Branch("l1_pt",l1_pt,"l1_pt/F")
passedEvents.Branch("l1_eta",l1_eta,"l1_eta/F")
passedEvents.Branch("l1_phi",l1_phi,"l1_phi/F")
passedEvents.Branch("l1_mass",l1_mass,"l1_mass/I")

passedEvents.Branch("l2_pt",l2_pt,"l2_pt/F")
passedEvents.Branch("l2_eta",l2_eta,"l2_eta/F")
passedEvents.Branch("l2_phi",l2_phi,"l2_phi/F")
passedEvents.Branch("l2_mass",l2_mass,"l2_mass/I")

passedEvents.Branch("pho1_pt",pho1_pt,"pho1_pt/F")
passedEvents.Branch("pho1_eta",pho1_eta,"pho1_eta/F")
passedEvents.Branch("pho1_phi",pho1_phi,"pho1_phi/F")

passedEvents.Branch("pho2_pt",pho2_pt,"pho2_pt/F")
passedEvents.Branch("pho2_eta",pho2_eta,"pho2_eta/F")
passedEvents.Branch("pho2_phi",pho2_phi,"pho2_phi/F")

passedEvents.Branch("Z_m",Z_m,"Z_m/F")
passedEvents.Branch("H_m",H_m,"H_m/F")
passedEvents.Branch("ALP_m",ALP_m,"ALP_m/F")
passedEvents.Branch("ALP_Et",ALP_Et,"ALP_Et/F")
passedEvents.Branch("dR_pho",dR_pho,"dR_pho/F")
passedEvents.Branch("dEta_pho",dEta_pho,"dEta_pho/F")
passedEvents.Branch("dPhi_pho",dPhi_pho,"dPhi_pho/F")
##############################################################################################

pho1_pt_less10 = array('f',[0.])
pho1_eta_less10 = array('f',[0.])
pho1_phi_less10 = array('f',[0.])
pho1_photonIso_less10 = array('f',[0.])

pho2_pt_less10 = array('f',[0.])
pho2_eta_less10 = array('f',[0.])
pho2_phi_less10 = array('f',[0.])
pho2_photonIso_less10 = array('f',[0.])

H_m_less10 = array('f',[0.])
ALP_m_less10 = array('f',[0.])
ALP_Et_less10 = array('f',[0.])

subpholessthan10 = ROOT.TTree("subpholessthan10","subpholessthan10")

subpholessthan10.Branch("pho1_pt_less10",pho1_pt_less10,"pho1_pt_less10/F")
subpholessthan10.Branch("pho1_eta_less10",pho1_eta_less10,"pho1_eta_less10/F")
subpholessthan10.Branch("pho1_phi_less10",pho1_phi_less10,"pho1_phi_less10/F")
subpholessthan10.Branch("pho1_photonIso_less10",pho1_photonIso_less10,"pho1_photonIso_less10/F")

subpholessthan10.Branch("pho2_pt_less10",pho2_pt_less10,"pho2_pt_less10/F")
subpholessthan10.Branch("pho2_eta_less10",pho2_eta_less10,"pho2_eta_less10/F")
subpholessthan10.Branch("pho2_phi_less10",pho2_phi_less10,"pho2_phi_less10/F")
subpholessthan10.Branch("pho2_photonIso_less10",pho2_photonIso_less10,"pho2_photonIso_less10/F")

subpholessthan10.Branch("H_m_less10",H_m_less10,"H_m_less10/F")
subpholessthan10.Branch("ALP_m_less10",ALP_m_less10,"ALP_m_less10/F")
subpholessthan10.Branch("ALP_Et_less10",ALP_Et_less10,"ALP_Et_less10/F")
##############################################################################################

pho1_pt_large10 = array('f',[0.])
pho1_eta_large10 = array('f',[0.])
pho1_phi_large10 = array('f',[0.])
pho1_photonIso_large10 = array('f',[0.])

pho2_pt_large10 = array('f',[0.])
pho2_eta_large10 = array('f',[0.])
pho2_phi_large10 = array('f',[0.])
pho2_photonIso_large10 = array('f',[0.])

H_m_large10 = array('f',[0.])
ALP_m_large10 = array('f',[0.])
ALP_Et_large10 = array('f',[0.])

subpholargethan10 = ROOT.TTree("subpholargethan10","subpholargethan10")

subpholargethan10.Branch("pho1_pt_large10",pho1_pt_large10,"pho1_pt_large10/F")
subpholargethan10.Branch("pho1_eta_large10",pho1_eta_large10,"pho1_eta_large10/F")
subpholargethan10.Branch("pho1_phi_large10",pho1_phi_large10,"pho1_phi_large10/F")
subpholargethan10.Branch("pho1_photonIso_large10",pho1_photonIso_large10,"pho1_photonIso_large10/F")

subpholargethan10.Branch("pho2_pt_large10",pho2_pt_large10,"pho2_pt_large10/F")
subpholargethan10.Branch("pho2_eta_large10",pho2_eta_large10,"pho2_eta_large10/F")
subpholargethan10.Branch("pho2_phi_large10",pho2_phi_large10,"pho2_phi_large10/F")
subpholargethan10.Branch("pho2_photonIso_large10",pho2_photonIso_large10,"pho2_photonIso_large10/F")

subpholargethan10.Branch("H_m_large10",H_m_large10,"H_m_large10/F")
subpholargethan10.Branch("ALP_m_large10",ALP_m_large10,"ALP_m_large10/F")
subpholargethan10.Branch("ALP_Et_large10",ALP_Et_large10,"ALP_Et_large10/F")

###############################################
pho1_matchratio = array('f',[0.])
pho2_matchratio = array('f',[0.])
Nmatch = array('i',[0])

passedEvents.Branch("pho1_matchratio",pho1_matchratio,"pho1_matchratio/F")
passedEvents.Branch("pho2_matchratio",pho2_matchratio,"pho2_matchratio/F")
passedEvents.Branch("Nmatch",Nmatch,"Nmatch/I")
###############################################




#Loop over all the events in the input ntuple
for ievent,event in enumerate(tchain):#, start=650000):
    if ievent > args.maxevents and args.maxevents != -1: break
    if ievent % 10000 == 0: print 'Processing entry ' + str(ievent)


    # Loop over all the electrons in an event
    lep_index = []

    pho_index = []
    dRpho = 0.0

    lep_leadingIndex = -1
    lep_subleadingIndex = -1
    pho_leadingIndex = -1
    pho_subleadingIndex = -1

    l1 = ROOT.TLorentzVector()
    l2 = ROOT.TLorentzVector()
    Z = ROOT.TLorentzVector()

    pho1 = ROOT.TLorentzVector()
    pho2 = ROOT.TLorentzVector()
    ALP = ROOT.TLorentzVector()

    H = ROOT.TLorentzVector()

    for i in range(event.GENlep_pt.size()):
	    #if (abs(event.GENlep_eta[i]) > 2.5): continue
        if (event.GENlep_MomId[i] == 23 and event.GENlep_MomMomId[i] == 25):
            lep_index.append(i)

    for i in range(event.GENpho_pt.size()):
        #if (event.GENpho_pt[i] < cut_pt): continue
        #if (abs(event.GENpho_eta[i]) > cut_eta): continue
        if (event.GENpho_MomId[i] == 9000005 and event.GENpho_MomMomId[i] == 25):
            pho_index.append(i)



    # Fill Tree
    if (len(lep_index) < 2): continue
    accetpance_lep.Fill(event.passedTrig)
    if (len(pho_index) < 2): continue
    if (event.GENlep_pt[lep_index[0]] > event.GENlep_pt[lep_index[1]]):
        lep_leadingIndex = lep_index[0]
        lep_subleadingIndex = lep_index[1]

    else:
        lep_leadingIndex = lep_index[1]
        lep_subleadingIndex = lep_index[0]

    if (event.GENpho_pt[pho_index[0]] > event.GENpho_pt[pho_index[1]]):
        pho_leadingIndex = pho_index[0]
        pho_subleadingIndex = pho_index[1]

    else:
        pho_leadingIndex = pho_index[1]
        pho_subleadingIndex = pho_index[0]

    l1.SetPtEtaPhiM(event.GENlep_pt[lep_leadingIndex], event.GENlep_eta[lep_leadingIndex], event.GENlep_phi[lep_leadingIndex], event.GENlep_mass[lep_leadingIndex])
    l2.SetPtEtaPhiM(event.GENlep_pt[lep_subleadingIndex], event.GENlep_eta[lep_subleadingIndex], event.GENlep_phi[lep_subleadingIndex], event.GENlep_mass[lep_subleadingIndex])

    pho1.SetPtEtaPhiM(event.GENpho_pt[pho_leadingIndex], event.GENpho_eta[pho_leadingIndex], event.GENpho_phi[pho_leadingIndex], 0.)
    pho2.SetPtEtaPhiM(event.GENpho_pt[pho_subleadingIndex], event.GENpho_eta[pho_subleadingIndex], event.GENpho_phi[pho_subleadingIndex], 0.)

    # Cuts
    #####################################################################
    Pt_pho1.Fill(pho1.Pt())
    Pt_pho2.Fill(pho2.Pt())
    Pt_l1.Fill(l1.Pt())
    Pt_l2.Fill(l2.Pt())

    if (abs(pho1.Eta()) > cut_eta): continue
    if (abs(pho2.Eta()) > cut_eta): continue
    Pt_pho1_Peta.Fill(pho1.Pt())
    Pt_pho2_Peta.Fill(pho2.Pt())
    Pt_l1_Peta.Fill(l1.Pt())
    Pt_l2_Peta.Fill(l2.Pt())

    if (abs(l1.Eta()) > cut_eta): continue
    if (abs(l2.Eta()) > cut_eta): continue
    Pt_pho1_Peta_Leta.Fill(pho1.Pt())
    Pt_pho2_Peta_Leta.Fill(pho2.Pt())
    Pt_l1_Peta_Leta.Fill(l1.Pt())
    Pt_l2_Peta_Leta.Fill(l2.Pt())

    if (pho1.Pt() < cut_pt): continue
    #if (pho2.Pt() < cut_pt): continue
    Pt_pho1_Peta_Leta_Ppt.Fill(pho1.Pt())
    Pt_pho2_Peta_Leta_Ppt.Fill(pho2.Pt())
    Pt_l1_Peta_Leta_Ppt.Fill(l1.Pt())
    Pt_l2_Peta_Leta_Ppt.Fill(l2.Pt())


    #####################################################################

    ALP = (pho1 + pho2)
    Z = (l1 + l2)
    dRpho = deltaR(pho1.Eta(), pho1.Phi(), pho2.Eta(), pho2.Phi())
    dEtapho = abs(pho1.Eta() - pho2.Eta())
    dPhipho = abs(pho1.Phi() - pho2.Phi())
    H = (Z + ALP)

    #####################################################################
    # sub-leading photon pt < 10 GeV
    if (pho2.Pt() < 10):
        pho1_pt_less10[0] = pho1.Pt()
        pho1_eta_less10[0] = pho1.Eta()
        pho1_phi_less10[0] = pho1.Phi()
        pho1_photonIso_less10[0] = event.GENpho_phoIso[pho_leadingIndex]

        pho2_pt_less10[0] = pho2.Pt()
        pho2_eta_less10[0] = pho2.Eta()
        pho2_phi_less10[0] = pho1.Phi()
        pho2_photonIso_less10[0] = event.GENpho_phoIso[pho_subleadingIndex]


        H_m_less10[0] = (Z + pho1).M()
        ALP_m_less10[0] = ALP.M()
        ALP_Et_less10[0] = ALP.Et()
        subpholessthan10.Fill()
    #####################################################################
    # sub-leading photon pt > 10 GeV
    if (pho2.Pt() > 10):
        pho1_pt_large10[0] = pho1.Pt()
        pho1_eta_large10[0] = pho1.Eta()
        pho1_phi_large10[0] = pho1.Phi()
        pho1_photonIso_large10[0] = event.GENpho_phoIso[pho_leadingIndex] - pho2.Pt()

        pho2_pt_large10[0] = pho2.Pt()
        pho2_eta_large10[0] = pho2.Eta()
        pho2_phi_large10[0] = pho1.Phi()
        pho2_photonIso_large10[0] = event.GENpho_phoIso[pho_subleadingIndex] - pho1.Pt()


        H_m_large10[0] = H.M()
        ALP_m_large10[0] = ALP.M()
        ALP_Et_large10[0] = ALP.Et()
        subpholargethan10.Fill()



    #####################################################################

    l1_pt[0] = l1.Pt()
    l1_eta[0] = l1.Eta()
    l1_phi[0] = l1.Phi()
    l1_mass[0] = l1.M()

    l2_pt[0] = l2.Pt()
    l2_eta[0] = l2.Eta()
    l2_phi[0] = l2.Phi()
    l2_mass[0] = l2.M()

    pho1_pt[0] = pho1.Pt()
    pho1_eta[0] = pho1.Eta()
    pho1_phi[0] = pho1.Phi()

    pho2_pt[0] = pho2.Pt()
    pho2_eta[0] = pho2.Eta()
    pho2_phi[0] = pho1.Phi()


    Z_m[0] = Z.M()
    H_m[0] = H.M()
    ALP_m[0] = ALP.M()
    ALP_Et[0] = ALP.Et()
    dR_pho[0] = dRpho
    dEta_pho[0] = dEtapho
    dPhi_pho[0] = dPhipho


    ##########################################
    if (ALP.Et() < x_max):
        x_et = int(ALP.Et())
        data_dEta[x_et] = data_dEta[x_et] + dEtapho
        data_N[x_et] = data_N[x_et] + 1.
        pho_dEta_2D.Fill(ALP.Et(),dEtapho)
    pho_dEtadPhi.Fill(dPhipho,dEtapho)
    ##########################################


    # match particles
    ##########################################
    dR_min = 0.1
    index_reco1 = 0
    index_reco2 = 0
    n_match = 0
    delta_pho1 = 9999.0
    delta_pho2 = 9999.0


    for i in range(event.pho_pt.size()):
        if (event.pho_pt[i] < cut_pt or abs(event.pho_eta[i]) > cut_eta): continue
        if (event.pho_hasPixelSeed[i] == 1): continue
	dR1 = deltaR(pho1.Eta(), pho1.Phi(), event.pho_eta[i], event.pho_phi[i])
        if (dR1 < delta_pho1):
            delta_pho1 = dR1
            index_reco1 = i

    for j in range(event.pho_pt.size()):
        if (event.pho_pt[j] < cut_pt or abs(event.pho_eta[j]) > cut_eta): continue
	if (event.pho_hasPixelSeed[j] == 1): continue
        if (j == index_reco1): continue
        dR2 = deltaR(pho2.Eta(), pho2.Phi(), event.pho_eta[j], event.pho_phi[j])
        if (dR2 < delta_pho2):
            delta_pho2 = dR2
            index_reco2 = j

    if (delta_pho1 < dR_min):
        n_match = n_match + 1
        pho1_matchratio[0] = event.pho_pt[index_reco1]/pho1.Pt()

    if (delta_pho2 < dR_min):
        n_match = n_match + 1
        pho2_matchratio[0] = event.pho_pt[index_reco2]/pho2.Pt()

    Nmatch[0] = n_match

    ##########################################
    # end match
    passedEvents.Fill()
for i in range(len(data_N)):
    if data_N[i] != 0:
        data_dEta[i] = data_dEta[i]/data_N[i]
    pho_dEta.SetBinContent(i,data_dEta[i])




file_out.Write()
file_out.Close()

sw.Stop()
print 'Real time: ' + str(round(sw.RealTime() / 60.0,2)) + ' minutes'
print 'CPU time:  ' + str(round(sw.CpuTime() / 60.0,2)) + ' minutes'
