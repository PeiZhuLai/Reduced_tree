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


########## photon SFs
if args.year == '2016':
    f_SFs = '/publicfs/cms/user/wangzebing/ALP/Plot/SFs/egammaEffi_2016.txt'
elif args.year == '2017':
    f_SFs = '/publicfs/cms/user/wangzebing/ALP/Plot/SFs/egammaEffi_2017.txt'
elif args.year == '2018':
    f_SFs = '/publicfs/cms/user/wangzebing/ALP/Plot/SFs/egammaEffi_2018.txt'
else:
    print "do not include at 2016/2017/2018"
    exit(0)
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

l2_pt = array('f',[0.])
l2_eta = array('f',[0.])
l2_phi = array('f',[0.])
l2_id = array('i',[0])

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

passedEvents.Branch("l2_pt",l2_pt,"l2_pt/F")
passedEvents.Branch("l2_eta",l2_eta,"l2_eta/F")
passedEvents.Branch("l2_phi",l2_phi,"l2_phi/F")
passedEvents.Branch("l2_id",l2_id,"l2_id/I")



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
    # Find photon
############################################################

    N_pho = event.pho_pt.size()
    npho.Fill(N_pho)
    if (N_pho < 2): continue


    for i in range(N_pho):
        if (event.pho_hasPixelSeed[i] == 0): continue#FIXE
        if (event.pho_pt[i] > pho1_maxPt):
            pho1_maxPt = event.pho_pt[i]
            pho1_index = i
            foundpho1 = True

    for j in range(N_pho):
        if (event.pho_hasPixelSeed[j] == 0): continue#FIXED
        if j == pho1_index: continue
        if (event.pho_pt[j] > pho2_maxPt):
            pho2_maxPt = event.pho_pt[j]
            pho2_index = j
            foundpho2 = True

    if (foundpho1 and foundpho2):


    ################################################################################################

        pho1_find = ROOT.TLorentzVector()
        pho2_find = ROOT.TLorentzVector()

        pho1_find.SetPtEtaPhiM(event.pho_pt[pho1_index], event.pho_eta[pho1_index], event.pho_phi[pho1_index], 0.0)
        pho2_find.SetPtEtaPhiM(event.pho_pt[pho2_index], event.pho_eta[pho2_index], event.pho_phi[pho2_index], 0.0)

        ALP_find = ROOT.TLorentzVector()
        ALP_find = (pho1_find + pho2_find)
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

            #H_Ceta.Fill(H_find.M())


            if pho1_find.Pt()>35.: continue
            if pho2_find.Pt()>35.: continue
            #if pho_passEleVeto: continue
            if not pho_passHOverE: continue
            if not pho_passChaHadIso: continue
            if not pho_passNeuHadIso: continue
            if ALP_find.M()<70 or ALP_find.M()>110: continue

            Run[0] = event.Run
            LumiSect[0] = event.LumiSect
            Event[0] = event.Event
            #if isMC:
            #    lep_dataMC = event.lep_dataMC[lep_leadindex[0]] * event.lep_dataMC[lep_leadindex[1]]
            #else:
            #    lep_dataMC = 1.0
            factor[0] = event.genWeight * event.pileupWeight# * lep_dataMC * weight
            event_weight[0] = weight
            event_genWeight[0] = event.genWeight
            event_pileupWeight[0] = event.pileupWeight
            #event_dataMCWeight[0] = lep_dataMC


            ALP_m[0] = ALP_find.M()

            dR_g1g2 = deltaR(pho1_find.Eta(), pho1_find.Phi(), pho2_find.Eta(), pho2_find.Phi())

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

            #######################################################################################################

            passedEvents.Fill()











file_out.Write()
file_out.Close()

sw.Stop()
print 'Real time: ' + str(round(sw.RealTime() / 60.0,2)) + ' minutes'
print 'CPU time:  ' + str(round(sw.CpuTime() / 60.0,2)) + ' minutes'
