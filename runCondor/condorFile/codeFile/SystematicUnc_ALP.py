import math
import ROOT

def showerShapeUncVal(eta, R9):
    bins_eta_dn = [0.0, 1.0, 1.5, 2.0]
    bins_eta_up = [1.0, 1.5, 2.0, 6.0]
    bins_R9_dn = [0.0, 0.94]
    bins_R9_up = [0.94, 999.]

    if abs(eta)>=bins_eta_dn[0] and abs(eta)<bins_eta_up[0]:
        tag1 = "EB1"
    elif abs(eta)>=bins_eta_dn[1] and abs(eta)<bins_eta_up[1]:
        tag1 = "EB2"
    elif abs(eta)>=bins_eta_dn[2] and abs(eta)<bins_eta_up[2]:
        tag1 = "EE1"
    elif abs(eta)>=bins_eta_dn[3] and abs(eta)<bins_eta_up[3]:
        tag1 = "EE2"
    else:
        print "showerShapeBins eta error"
        exit(0)

    if R9>=bins_R9_dn[0] and R9<bins_R9_up[0]:
        tag2 = "lowR9"
    elif R9>=bins_R9_dn[1] and R9<bins_R9_up[1]:
        tag2 = "highR9"
    else:
        print "showerShapeBins R9 error"
        exit(0)

    tag = tag1 + "_" + tag2

    showerShapeVal = {
    "EB1_lowR9" : [0., -0.0001],
    "EB2_lowR9" : [0., 0.0002],
    "EB1_highR9" : [0., -0.0006],
    "EB2_highR9" : [0., -0.0011],
    "EE1_lowR9" : [0., 0.0015],
    "EE2_lowR9" : [0., 0.0004],
    "EE1_highR9" : [0., 0.0002],
    "EE2_highR9" : [0., 0.0003],
    }
    return showerShapeVal[tag]

def updateEnergy(obj, factor, Energy_corr = 0.):

    obj_corr = ROOT.TLorentzVector()
    obj_corr.setPxPyPzE(obj.Px()*factor, obj.Py()*factor, obj.Pz()*factor, Energy_corr)

    return obj_corr
