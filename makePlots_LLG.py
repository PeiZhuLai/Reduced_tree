#! /usr/bin/env python

import argparse

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
import tdrStyle
tdrStyle.setTDRStyle()
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



# Z histograms
h_Z_all = ROOT.TH1D('Zmass_all', 'Zmass_all', 100, 0, 500)
h_Z_cut_l1l2 = ROOT.TH1D('Zmass_cut_l1l2', 'Zmass_cut_l1l2', 100, 0, 500)
h_Z_cut_l1G = ROOT.TH1D('Zmass_cut_l1G', 'Zmass_cut_l1G', 100, 0, 500)
h_Z_cut_l2G = ROOT.TH1D('Zmass_cut_l2G', 'Zmass_cut_l2G', 100, 0, 500)
h_Z_cut = ROOT.TH1D('Zmass_cut', 'Zmass_cut', 100, 0, 500)

# Gamma histograms
h_gamma_pt = ROOT.TH1D('gamma_all_pt', 'gamma_all_pt', 100, 0, 500)

# Higgs histograms
h_m_llg_nocut = ROOT.TH1D('h_m_llg_nocut', 'h_m_llg_nocut', 100, 80., 190)
h_m_llg_cut_l1l2 = ROOT.TH1D('h_m_llg_cut_l1l2', 'h_m_llg_cut_l1l2', 100, 80., 190)
h_m_llg_cut_l1G = ROOT.TH1D('h_m_llg_cut_l1G', 'h_m_llg_cut_l1G', 100, 80., 190)
h_m_llg_cut_l2G = ROOT.TH1D('h_m_llg_cut_l2G', 'h_m_llg_cut_l2G', 100, 80., 190)
h_m_llg_cut = ROOT.TH1D('h_m_llg_cut', 'h_m_llg_cut', 100, 80., 190)



npho = ROOT.TH1D('npho', 'npho', 10, 0., 10)
nlep = ROOT.TH1D('nlep', 'nlep', 10, 0., 10)


# delta R
dRl1l2_nocut = ROOT.TH1D('dRl1l2_nocut', 'dRl1l2_nocut', 100, 0., 10.)
dRl1G_nocut = ROOT.TH1D('dRl1G_nocut', 'dRl1G_nocut', 100, 0., 10.)
dRl2G_nocut = ROOT.TH1D('dRl2G_nocut', 'dRl2G_nocut', 100, 0., 10.)

dRl1l2 = ROOT.TH1D('dRl1l2', 'dRl1l2', 100, 0., 10.)
dRl1G = ROOT.TH1D('dRl1G', 'dRl1G', 100, 0., 10.)
dRl2G = ROOT.TH1D('dRl2G', 'dRl2G', 100, 0., 10.)

adtional = ROOT.TH1D('adtional', 'adtional', 100, 0., 100.)
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
phofind_pt = ROOT.TH1D('phofind_pt', 'phofind_pt', 100, 0, 500)
################################################################################################
l1_id_Ceta = ROOT.TH1D('l1_id_Ceta', 'l1_id_Ceta', 40, -20., 20.)
Z_Ceta = ROOT.TH1D('Z_Ceta', 'Z_Ceta', 100, 0, 500)
H_Ceta = ROOT.TH1D('H_Ceta', 'H_Ceta', 100, 80., 190)
photon_pt_no = ROOT.TH1D('photon_pt_no', 'photon_pt_no', 200, 0, 200)

l1_id_pho_vote = ROOT.TH1D('l1_id_pho_vote', 'l1_id_pho_vote', 40, -20., 20.)
Z_pho_vote = ROOT.TH1D('Z_pho_vote', 'Z_pho_vote', 100, 0, 500)
H_pho_vote = ROOT.TH1D('H_pho_vote', 'H_pho_vote', 100, 80., 190)
photon_pt_vote = ROOT.TH1D('photon_pt_vote', 'photon_pt_vote', 200, 0, 200)

Z_pho_mva = ROOT.TH1D('Z_pho_mva', 'Z_pho_mva', 100, 0, 500)
H_pho_mva = ROOT.TH1D('H_pho_mva', 'H_pho_mva', 100, 80., 190)
photon_pt_vote_mva = ROOT.TH1D('photon_pt_vote_mva', 'photon_pt_vote_mva', 200, 0, 200)


################################################################################################
Z_c1 = ROOT.TH1D('Z_c1', 'Z_c1', 100, 0, 500)
H_c1 = ROOT.TH1D('H_c1', 'H_c1', 100, 80., 190)

Z_c2 = ROOT.TH1D('Z_c2', 'Z_c2', 100, 0, 500)
H_c2 = ROOT.TH1D('H_c2', 'H_c2', 100, 80., 190)

Z_c3 = ROOT.TH1D('Z_c3', 'Z_c3', 100, 0, 500)
H_c3 = ROOT.TH1D('H_c3', 'H_c3', 100, 80., 190)

Z_c4 = ROOT.TH1D('Z_c4', 'Z_c4', 100, 0, 500)
H_c4 = ROOT.TH1D('H_c4', 'H_c4', 100, 80., 190)

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
phofind_pt.SetStats(1)

Z_pho_vote.SetStats(1)
H_pho_vote.SetStats(1)

Z_pho_mva.SetStats(1)
H_pho_mva.SetStats(1)



Z_c1.SetStats(1)
H_c1.SetStats(1)
Z_c2.SetStats(1)
H_c2.SetStats(1)
Z_c3.SetStats(1)
H_c3.SetStats(1)
Z_c4.SetStats(1)
H_c4.SetStats(1)

#Z_125.SetStats(1)

adtional.SetStats(1)

h_Z_all.SetStats(1)

h_Z_cut.SetStats(1)


h_gamma_pt.SetStats(1)

h_m_llg_nocut.SetStats(1)
h_m_llg_cut.SetStats(1)
#dRl1l2.SetStats(1)
#dRl1G.SetStats(1)
#dRl2G.SetStats(1)
# Tree
l1_pt = array('f',[0.])
l1_eta = array('f',[0.])
l1_phi = array('f',[0.])
l1_id = array('i',[0])



l2_pt = array('f',[0.])
l2_eta = array('f',[0.])
l2_phi = array('f',[0.])
l2_id = array('i',[0])

pho_pt = array('f',[0.])
pho_eta = array('f',[0.])
pho_phi = array('f',[0.])
pho_mva = array('f',[0.])

Z_m = array('f',[0.])
H_m = array('f',[0.])
H_pt = array('f',[0.])
dR12 = array('f',[0.])
dR1g = array('f',[0.])
dR2g = array('f',[0.])

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

passedEvents.Branch("pho_pt",pho_pt,"pho_pt/F")
passedEvents.Branch("pho_eta",pho_eta,"pho_eta/F")
passedEvents.Branch("pho_phi",pho_phi,"pho_phi/F")
passedEvents.Branch("pho_mva",pho_mva,"pho_mva/F")

passedEvents.Branch("Z_m",Z_m,"Z_m/F")
passedEvents.Branch("H_m",H_m,"H_m/F")
passedEvents.Branch("H_pt",H_pt,"H_pt/F")
passedEvents.Branch("dR12",dR12,"dR12/F")
passedEvents.Branch("dR1g",dR1g,"dR1g/F")
passedEvents.Branch("dR2g",dR2g,"dR2g/F")

passedEvents.Branch("event_cat",event_cat,"event_cat/I")




#Loop over all the events in the input ntuple
for ievent,event in enumerate(tchain):#, start=650000):
    if ievent > args.maxevents and args.maxevents != -1: break
    #if ievent == 100000: break
    if ievent % 10000 == 0: print 'Processing entry ' + str(ievent)


    # Loop over all the electrons in an event

    # pho parameters
    foundpho = False
    pho_maxPt = 0.0
    pho_index = 0
    pho_passEleVote = True
    pho_passMVA = True
    pho_passTight = True
    pho_passIso = True
    pho_passSigma = True
    pho_passHOverE = True

    # initialise Z parameters
    Nlep = 0
    GENNlep = 0
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

    EventCat = 0

    # Jet parameters
    jet_maxEt = 0.0
    jet_NmaxEt = 0.0
    jet_index1 = 0
    jet_index2 = 0
    findj1 = False
    findj2 = False


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

                h_Z_all.Fill(Z.M())
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
    #h_m_llg_nocut.Fill(mllg, weight)

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
        h_gamma_pt.Fill(event.pho_pt[i])

        if (event.pho_pt[i] > pho_maxPt):
            pho_maxPt = event.pho_pt[i]
            pho_index = i
            foundpho = True

    if (not foundpho): continue

    pho_find = ROOT.TLorentzVector()
    pho_find.SetPtEtaPhiM(event.pho_pt[pho_index], event.pho_eta[pho_index], event.pho_phi[pho_index], 0.0)

    phofind_pt.Fill(pho_find.Pt())

#######################################################################################################

    # Higgs Candidates
#######################################################################################################
    H_find = ROOT.TLorentzVector()
    H_find = (Z_find + pho_find)
#######################################################################################################


    # Photon Cuts
#######################################################################################################
    #photon selection

    if (event.pho_EleVote[pho_index] == 0): pho_passEleVote = False
    if (event.pho_mva90[pho_index] == 0): pho_passMVA = False
    #if (event.photonCutBasedIDTight[pho_index] == 0): pho_passTight = False

    # pass photon isolation


    if (((abs(event.pho_eta[pho_index]) >1.4442) and (abs(event.pho_eta[pho_index]) < 1.566)) or (abs(event.pho_eta[pho_index]) >2.5)): continue
# end isolation
# end selection


    # no Cuts
    l1_id_Ceta.Fill(event.lep_id[lep_leadindex[0]])
    Z_Ceta.Fill(Z_find.M())
    H_Ceta.Fill(H_find.M())
    #photon_pt_no.Fill(pho_find.Pt())

    if (not pho_passEleVote): continue
    l1_id_pho_vote.Fill(event.lep_id[lep_leadindex[0]])
    Z_pho_vote.Fill(Z_find.M())
    H_pho_vote.Fill(H_find.M())


    if (not pho_passMVA): continue
    Z_pho_mva.Fill(Z_find.M())
    H_pho_mva.Fill(H_find.M())

#######################################################################################################


    dR_l1l2 = deltaR(event.lepFSR_eta[lep_leadindex[0]], event.lepFSR_phi[lep_leadindex[0]], event.lepFSR_eta[lep_leadindex[1]], event.lepFSR_phi[lep_leadindex[1]])
    dR_l1G = deltaR(event.lepFSR_eta[lep_leadindex[0]], event.lepFSR_phi[lep_leadindex[0]], pho_find.Eta(), pho_find.Phi())
    dR_l2G = deltaR(event.lepFSR_eta[lep_leadindex[1]], event.lepFSR_phi[lep_leadindex[1]], pho_find.Eta(), pho_find.Phi())

    # before cut
    dRl1l2_nocut.Fill(dR_l1l2)
    dRl1G_nocut.Fill(dR_l1G)
    dRl2G_nocut.Fill(dR_l2G)

#########################################################################################################
    mllg = H_find.M()
    # cut 1
    if (mllg < 100 or mllg > 180): continue # 1
    Z_c1.Fill(Z_find.M())
    H_c1.Fill(H_find.M())

    # cut 2
    if ((pho_find.Pt() / mllg) < 15.0/110.0): continue # 2
    Z_c2.Fill(Z_find.M())
    H_c2.Fill(H_find.M())

    # cut 3

    if (dR_l1l2 >= 0.4):
        h_Z_cut_l1l2.Fill(Z_find.M())
        h_m_llg_cut_l1l2.Fill(mllg)
    if (dR_l1G >= 0.4):
        h_Z_cut_l1G.Fill(Z_find.M())
        h_m_llg_cut_l1G.Fill(mllg)
    if (dR_l2G >= 0.4):
        h_Z_cut_l2G.Fill(Z_find.M())
        h_m_llg_cut_l2G.Fill(mllg)

    if (dR_l1l2 < 0.4 or dR_l1G < 0.4 or dR_l2G < 0.4) : continue # 3

    Z_c3.Fill(Z_find.M())
    H_c3.Fill(H_find.M())

    # cut 4
    if ((mllg + Z_find.M()) < 185): continue # 4
    Z_c4.Fill(Z_find.M())
    H_c4.Fill(H_find.M())

    # end cut



    dRl1l2.Fill(dR_l1l2)
    dRl1G.Fill(dR_l1G)
    dRl2G.Fill(dR_l2G)


    h_Z_cut.Fill(Z_find.M(), weight)
    h_m_llg_cut.Fill(mllg, weight)
    photon_pt_vote_mva.Fill(pho_find.Pt())

    # Event Categories
#########################################################################################################
    # Electron Categories
    if (abs(event.lep_id[lep_leadindex[0]]) == 11):
        # Lepton Tag
        if (Nlep == 3):
            for i in range(Nlep):
                if ( (i == lep_leadindex[0]) or (i == lep_leadindex[1]) ): continue
                adtional.Fill(event.lep_pt[i])
                if (((abs(event.lep_id[i]) == 11) and (event.lep_pt[i] > 7)) or ((abs(event.lep_id[i]) == 13) and (event.lep_pt[i] > 5))):

                    if (event.lep_RelIsoNoFSR[i] > 0.35): continue
                    if (not (event.lep_tightId[i])): continue
                    if (deltaR(event.lep_eta[i], event.lep_phi[i], H_find.Eta(), H_find.Phi()) < 0.4): continue
                    if ((event.lepFSR_pt[i] > event.lepFSR_pt[lep_leadindex[0]]) or (event.lepFSR_pt[i] > event.lepFSR_pt[lep_leadindex[1]])): continue
                    EventCat = 1

        # Dijet Tag
        if (EventCat!=1):
            if (event.jet_pt.size() >= 2):

                for i in range(event.jet_pt.size()):
                    jeti = ROOT.TLorentzVector()
                    jeti.SetPtEtaPhiM(event.jet_pt[i], event.jet_eta[i], event.jet_phi[i], event.jet_mass[i])
                    if (deltaR(jeti.Eta(), jeti.Phi(), pho_find.Eta(), pho_find.Phi()) < 0.4) : continue
                    if (deltaR(jeti.Eta(), jeti.Phi(), l1_find.Eta(), l1_find.Phi()) < 0.4): continue
                    if (deltaR(jeti.Eta(), jeti.Phi(), l2_find.Eta(), l2_find.Phi()) < 0.4): continue
                    if (jeti.Et() < 30): continue
                    if (abs(jeti.Eta()) > 4.7): continue
                    if (jeti.Et() > jet_maxEt):
                        jet_maxEt = jeti.Et()
                        jet_index1 = i
                        findj1 = True

                for j in range(event.jet_pt.size()):
                    jetj = ROOT.TLorentzVector()
                    jetj.SetPtEtaPhiM(event.jet_pt[j], event.jet_eta[j], event.jet_phi[j], event.jet_mass[j])
                    if (deltaR(jetj.Eta(), jeti.Phi(), pho_find.Eta(), pho_find.Phi()) < 0.4) : continue
                    if (deltaR(jetj.Eta(), jeti.Phi(), l1_find.Eta(), l1_find.Phi()) < 0.4): continue
                    if (deltaR(jetj.Eta(), jeti.Phi(), l2_find.Eta(), l2_find.Phi()) < 0.4): continue
                    if (jetj.Et() < 30): continue
                    if (abs(jetj.Eta()) > 4.7): continue
                    if (j == jet_index1): continue
                    if (jetj.Et() > jet_NmaxEt):
                        jet_NmaxEt = jetj.Et()
                        jet_index2 = j
                        findj2 = True


                if (findj1 and findj2):
                    jet1 = ROOT.TLorentzVector()
                    jet2 = ROOT.TLorentzVector()
                    dijet = ROOT.TLorentzVector()

                    jet1.SetPtEtaPhiM(event.jet_pt[jet_index1], event.jet_eta[jet_index1], event.jet_phi[jet_index1], event.jet_mass[jet_index1])
                    jet2.SetPtEtaPhiM(event.jet_pt[jet_index2], event.jet_eta[jet_index2], event.jet_phi[jet_index2], event.jet_mass[jet_index2])
                    dijet = (jet1 + jet2)

                    if (abs(event.jet_eta[jet_index1] - event.jet_eta[jet_index2]) >= 3.5):
                        if ( (H_find.Eta() - (event.jet_eta[jet_index1] + event.jet_eta[jet_index2])/2.0) <= 2.5 ):
                            if ( dijet.M() > 500 ):
                                if ( deltaR(dijet.Eta(), dijet.Phi(), H_find.Eta(), H_find.Phi()) > 2.4 ): EventCat = 2

        # Boosted Tag
        if (EventCat!=1 and EventCat!=2):
            if ( H_find.Pt() > 60 ): EventCat = 4

        # Untagged Tag
        if (EventCat!=1 and EventCat!=2 and EventCat!=4):
        # Untagged 1 Electron
            if ( ((abs(pho_find.Eta()) > 0.) and (abs(pho_find.Eta()) < 1.4442)) and ((abs(l1_find.Eta()) > 0.) and (abs(l1_find.Eta()) < 1.4442) and (abs(l2_find.Eta()) > 0.) and (abs(l2_find.Eta()) < 1.4442)) and (event.pho_R9[pho_index] > 0.94) ): EventCat = 6

        # Untagged 2 Electron
        if (EventCat!=1 and EventCat!=2 and EventCat!=4 and EventCat!=6):
            if ( ((abs(pho_find.Eta()) > 0.) and (abs(pho_find.Eta()) < 1.4442)) and ((abs(l1_find.Eta()) > 0.) and (abs(l1_find.Eta()) < 1.4442) and (abs(l2_find.Eta()) > 0.) and (abs(l2_find.Eta()) < 1.4442)) and (event.pho_R9[pho_index] < 0.94) ): EventCat = 8

        # Untagged 3 Electron
        if (EventCat!=1 and EventCat!=2 and EventCat!=4 and EventCat!=6 and EventCat!=8):
            if ( ((abs(pho_find.Eta()) > 0.) and (abs(pho_find.Eta()) < 1.4442)) and (((abs(l1_find.Eta()) > 1.4442) and (abs(l1_find.Eta()) < 2.5)) or ((abs(l2_find.Eta()) > 1.4442) and (abs(l2_find.Eta()) < 2.5))) ): EventCat = 10

        # Untagged 4 Electron
        if (EventCat!=1 and EventCat!=2 and EventCat!=4 and EventCat!=6 and EventCat!=8 and EventCat!=10):
            if ( ((abs(pho_find.Eta()) > 1.566) and (abs(pho_find.Eta()) < 2.5)) and ((abs(l1_find.Eta()) > 0.) and (abs(l1_find.Eta()) < 2.5) and (abs(l2_find.Eta()) > 0.) and (abs(l2_find.Eta()) < 2.5)) ): EventCat = 12
            else: EventCat = 14

    # Muon Categories
    if (abs(event.lep_id[lep_leadindex[0]]) == 13):
        # Lepton Tag
        if (Nlep == 3):
            for i in range(Nlep):
                if ( (i == lep_leadindex[0]) or (i == lep_leadindex[1]) ): continue
                adtional.Fill(event.lep_pt[i])
                if (((abs(event.lep_id[i]) == 11) and (event.lep_pt[i] > 7)) or ((abs(event.lep_id[i]) == 13) and (event.lep_pt[i] > 5))):

                    if (event.lep_RelIsoNoFSR[i] > 0.35): continue
                    if (not (event.lep_tightId[i])): continue
                    if (deltaR(event.lep_eta[i], event.lep_phi[i], H_find.Eta(), H_find.Phi()) < 0.4): continue
                    if ((event.lepFSR_pt[i] > event.lepFSR_pt[lep_leadindex[0]]) or (event.lepFSR_pt[i] > event.lepFSR_pt[lep_leadindex[1]])): continue
                    EventCat = 1

        # Dijet Tag
        if (EventCat!=1):
            if (event.jet_pt.size() >= 2):

                for i in range(event.jet_pt.size()):
                    jeti = ROOT.TLorentzVector()
                    jeti.SetPtEtaPhiM(event.jet_pt[i], event.jet_eta[i], event.jet_phi[i], event.jet_mass[i])
                    if (deltaR(jeti.Eta(), jeti.Phi(), pho_find.Eta(), pho_find.Phi()) < 0.4) : continue
                    if (deltaR(jeti.Eta(), jeti.Phi(), l1_find.Eta(), l1_find.Phi()) < 0.4): continue
                    if (deltaR(jeti.Eta(), jeti.Phi(), l2_find.Eta(), l2_find.Phi()) < 0.4): continue
                    if (jeti.Et() < 30): continue
                    if (abs(jeti.Eta()) > 4.7): continue
                    if (jeti.Et() > jet_maxEt):
                        jet_maxEt = jeti.Et()
                        jet_index1 = i
                        findj1 = True

                for j in range(event.jet_pt.size()):
                    jetj = ROOT.TLorentzVector()
                    jetj.SetPtEtaPhiM(event.jet_pt[j], event.jet_eta[j], event.jet_phi[j], event.jet_mass[j])
                    if (deltaR(jetj.Eta(), jeti.Phi(), pho_find.Eta(), pho_find.Phi()) < 0.4) : continue
                    if (deltaR(jetj.Eta(), jeti.Phi(), l1_find.Eta(), l1_find.Phi()) < 0.4): continue
                    if (deltaR(jetj.Eta(), jeti.Phi(), l2_find.Eta(), l2_find.Phi()) < 0.4): continue
                    if (jetj.Et() < 30): continue
                    if (abs(jetj.Eta()) > 4.7): continue
                    if (j == jet_index1): continue
                    if (jetj.Et() > jet_NmaxEt):
                        jet_NmaxEt = jetj.Et()
                        jet_index2 = j
                        findj2 = True


                if (findj1 and findj2):
                    jet1 = ROOT.TLorentzVector()
                    jet2 = ROOT.TLorentzVector()
                    dijet = ROOT.TLorentzVector()

                    jet1.SetPtEtaPhiM(event.jet_pt[jet_index1], event.jet_eta[jet_index1], event.jet_phi[jet_index1], event.jet_mass[jet_index1])
                    jet2.SetPtEtaPhiM(event.jet_pt[jet_index2], event.jet_eta[jet_index2], event.jet_phi[jet_index2], event.jet_mass[jet_index2])
                    dijet = (jet1 + jet2)

                    if (abs(event.jet_eta[jet_index1] - event.jet_eta[jet_index2]) >= 3.5):
                        if ( (H_find.Eta() - (event.jet_eta[jet_index1] + event.jet_eta[jet_index2])/2.0) <= 2.5 ):
                            if ( dijet.M() > 500 ):
                                if ( deltaR(dijet.Eta(), dijet.Phi(), H_find.Eta(), H_find.Phi()) > 2.4 ): EventCat = 3

        # Boosted Tag
        if (EventCat!=1 and EventCat!=3):
            if ( H_find.Pt() > 60 ): EventCat = 5

        # Untagged Tag
        if (EventCat!=1 and EventCat!=3 and EventCat!=5):
        # Untagged 1 Muon
            if ( ((abs(pho_find.Eta()) > 0.) and (abs(pho_find.Eta()) < 1.4442)) and (((abs(l1_find.Eta()) > 0.) and (abs(l1_find.Eta()) < 2.1) and (abs(l2_find.Eta()) > 0.) and (abs(l2_find.Eta()) < 2.1)) and ( ((abs(l1_find.Eta()) > 0.) and (abs(l1_find.Eta()) < 0.9)) or ((abs(l2_find.Eta()) > 0.) and (abs(l2_find.Eta()) < 0.9)) )) and (event.pho_R9[pho_index] > 0.94) ): EventCat = 7

        # Untagged 2 Muon
        if (EventCat!=1 and EventCat!=3 and EventCat!=5 and EventCat!=7):
            if ( ((abs(pho_find.Eta()) > 0.) and (abs(pho_find.Eta()) < 1.4442)) and (((abs(l1_find.Eta()) > 0.) and (abs(l1_find.Eta()) < 2.1) and (abs(l2_find.Eta()) > 0.) and (abs(l2_find.Eta()) < 2.1)) and ( ((abs(l1_find.Eta()) > 0.) and (abs(l1_find.Eta()) < 0.9)) or ((abs(l2_find.Eta()) > 0.) and (abs(l2_find.Eta()) < 0.9)) )) and (event.pho_R9[pho_index] < 0.94) ): EventCat = 9

        # Untagged 3 Muon
        if (EventCat!=1 and EventCat!=3 and EventCat!=5 and EventCat!=7 and EventCat!=9):
            if ( ((abs(pho_find.Eta()) > 0.) and (abs(pho_find.Eta()) < 1.4442)) and ( ((abs(l1_find.Eta()) > 0.9) and (abs(l2_find.Eta()) > 0.9)) or ( ( (abs(l1_find.Eta()) > 2.1) and (abs(l1_find.Eta()) <2.4) ) or ((abs(l2_find.Eta()) > 2.1) and (abs(l2_find.Eta()) <2.4)) ) ) ): EventCat = 11

        # Untagged 4 Muon
        if (EventCat!=1 and EventCat!=3 and EventCat!=5 and EventCat!=7 and EventCat!=9 and EventCat!=11):
            if ( ((abs(pho_find.Eta()) > 1.566) and (abs(pho_find.Eta()) < 2.5)) and ((abs(l1_find.Eta()) > 0.) and (abs(l1_find.Eta()) < 2.4) and (abs(l2_find.Eta()) > 0.) and (abs(l2_find.Eta()) < 2.4)) ): EventCat = 13
            else: EventCat = 15
#########################################################################################################

    # Fill Tree
    l1_pt[0] = event.lepFSR_pt[lep_leadindex[0]]
    l2_pt[0] = event.lepFSR_pt[lep_leadindex[1]]
    l1_eta[0] = event.lepFSR_eta[lep_leadindex[0]]
    l2_eta[0] = event.lepFSR_eta[lep_leadindex[1]]
    l1_phi[0] = event.lepFSR_phi[lep_leadindex[0]]
    l2_phi[0] = event.lepFSR_phi[lep_leadindex[1]]
    l1_id[0] = event.lep_id[lep_leadindex[0]]
    l2_id[0] = event.lep_id[lep_leadindex[1]]

    pho_pt[0] = event.pho_pt[pho_index]
    pho_eta[0] = event.pho_eta[pho_index]
    pho_phi[0] = event.pho_phi[pho_index]
    pho_mva[0] = event.pho_mva[pho_index]

    Z_m[0] = Z_find.M()
    H_m[0] = mllg
    dR12[0] = dR_l1l2
    dR1g[0] = dR_l1G
    dR2g[0] = dR_l2G

    event_cat[0] = EventCat
    H_pt[0] = H_find.Pt()
    passedEvents.Fill()











file_out.Write()
file_out.Close()

sw.Stop()
print 'Real time: ' + str(round(sw.RealTime() / 60.0,2)) + ' minutes'
print 'CPU time:  ' + str(round(sw.CpuTime() / 60.0,2)) + ' minutes'
