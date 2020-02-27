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
l1_id_Ceta = ROOT.TH1D('l1_id_Ceta', 'l1_id_Ceta', 40, -20., 20.)
Z_Ceta = ROOT.TH1D('Z_Ceta', 'Z_Ceta', 100, 0, 500)
H_Ceta = ROOT.TH1D('H_Ceta', 'H_Ceta', 100, 80., 190)

l1_id_pho_vote = ROOT.TH1D('l1_id_pho_vote', 'l1_id_pho_vote', 40, -20., 20.)
Z_pho_vote = ROOT.TH1D('Z_pho_vote', 'Z_pho_vote', 100, 0, 500)
H_pho_vote = ROOT.TH1D('H_pho_vote', 'H_pho_vote', 100, 80., 190)

Z_pho_mva = ROOT.TH1D('Z_pho_mva', 'Z_pho_mva', 100, 0, 500)
H_pho_mva = ROOT.TH1D('H_pho_mva', 'H_pho_mva', 100, 80., 190)


################################################################################################




h_n.SetStats(1)
h_n_trig.SetStats(1)


l1_id_Ceta.SetStats(1)

Z_Ceta.SetStats(1)
H_Ceta.SetStats(1)

Z_e_nocut.SetStats(1)
Z_mu_nocut.SetStats(1)
Z_e_lIso.SetStats(1)
Z_mu_lIso.SetStats(1)
Z_e_lIso_lTight.SetStats(1)
Z_mu_lIso_lTight.SetStats(1)

Z_50.SetStats(1)

Z_pho_vote.SetStats(1)
H_pho_vote.SetStats(1)

Z_pho_mva.SetStats(1)
H_pho_mva.SetStats(1)


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

event_cat = array('i',[0])

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


passedEvents.Branch("event_cat",event_cat,"event_cat/I")




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
    pho_passEleVote = True
    pho_passMVA = True
    pho_passTight = True
    pho_passIso = True
    pho_passSigma = True
    pho_passHOverE = True

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
    dR_l1l2 = 0
    dR_l1G = 0
    dR_l2G = 0

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
    if (event.pho_pt.size() < 2): continue


    for i in range(event.pho_pt.size()):
        #if (event.pho_pt[i] < 15.): continue
        if (event.pho_hasPixelSeed[i] == 1): continue

        if (event.pho_pt[i] > pho1_maxPt):
            pho1_maxPt = event.pho_pt[i]
            pho1_index = i
            foundpho1 = True

    for j in range(event.pho_pt.size()):
        if (event.pho_hasPixelSeed[j] == 1): continue
        if j == pho1_index: continue
        if (event.pho_pt[j] > pho2_maxPt):
            pho2_maxPt = event.pho_pt[i]
            pho2_index = i
            foundpho2 = True

    if (not (foundpho1 and foundpho2)): continue

    pho1_find = ROOT.TLorentzVector()
    pho2_find = ROOT.TLorentzVector()
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

    '''
    # Photon Cuts
#######################################################################################################
    #photon selection

    if (event.pho_EleVote[pho1_index] == 0): pho_passEleVote = False
    if (event.pho_mva90[pho1_index] == 0): pho_passMVA = False
    #if (event.photonCutBasedIDTight[pho1_index] == 0): pho_passTight = False

    # pass photon isolation


    if (((abs(event.pho_eta[pho1_index]) >1.4442) and (abs(event.pho_eta[pho1_index]) < 1.566)) or (abs(event.pho_eta[pho1_index]) >2.5)): continue
# end isolation
# end selection


    # no Cuts
    l1_id_Ceta.Fill(event.lep_id[lep_leadindex[0]])
    Z_Ceta.Fill(Z_find.M())
    H_Ceta.Fill(H_find.M())

    if (not pho_passEleVote): continue
    l1_id_pho_vote.Fill(event.lep_id[lep_leadindex[0]])
    Z_pho_vote.Fill(Z_find.M())
    H_pho_vote.Fill(H_find.M())


    if (not pho_passMVA): continue
    Z_pho_mva.Fill(Z_find.M())
    H_pho_mva.Fill(H_find.M())

#######################################################################################################
    '''

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

    pho2_pt[0] = event.pho_pt[pho2_index]
    pho2_eta[0] = event.pho_eta[pho2_index]
    pho2_phi[0] = event.pho_phi[pho2_index]
    pho2_mva[0] = event.pho_mva[pho2_index]

    dRpho = deltaR(pho1_find.Eta(), pho1_find.Phi(), pho2_find.Eta(), pho2_find.Phi())

    Z_m[0] = Z_find.M()
    H_m[0] = H_find.M()
    ALP_m = ALP_find.M()
    dR_pho = dRpho
    H_pt[0] = H_find.Pt()
    passedEvents.Fill()











file_out.Write()
file_out.Close()

sw.Stop()
print 'Real time: ' + str(round(sw.RealTime() / 60.0,2)) + ' minutes'
print 'CPU time:  ' + str(round(sw.CpuTime() / 60.0,2)) + ' minutes'
