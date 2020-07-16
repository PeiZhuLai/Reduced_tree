#!/usr/bin/env python

from ROOT import TMVA, TFile, TString
from array import array
from subprocess import call
from os.path import isfile


def bookMVA():

    # Setup TMVA
    TMVA.Tools.Instance()
    TMVA.PyMethodBase.PyInitialize()
    reader = TMVA.Reader("Color:Silent")

    # Load data
    variables = ['pho1Pt','pho1R9','pho1IetaIeta','pho1HOE','pho1CIso','pho1NIso','pho1PIso']
    global branches
    branches = {}
    for var in variables:
        branches[var] = array('f', [-999])
        reader.AddVariable(var, branches[var])

    # Book methods
    reader.BookMVA('BDT method', TString('/publicfs/cms/user/wangzebing/ALP/Analysis_code/MVA/weight/TMVAClassification_BDT_NTree150_nCuts40.weights.xml'))

    return reader

def calMVA(reader, var_value):
    variables = ['pho1Pt','pho1R9','pho1IetaIeta','pho1HOE','pho1CIso','pho1NIso','pho1PIso']
    for var in variables:
        branches[var][0] = var_value[var]
    value = reader.EvaluateMVA('BDT method')
    return value
