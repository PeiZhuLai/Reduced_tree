#! /usr/bin/env python
import os
import glob
import math
from array import array
import sys
import time
import subprocess
import tarfile
import datetime
import commands
import optparse
from ROOT import *

#define function for parsing options
def parseOptions():
    global observalbesTags, modelTags, runAllSteps

    usage = ('usage: %prog [options]\n'
             + '%prog -h for help')
    parser = optparse.OptionParser(usage)

    # input options
    parser.add_option('-i', '--input', dest='INPUT', type='string',default='../CheckCRAB/jobFiles', help='the path of input txt files')
    parser.add_option('-o', '--output', dest='OUTPUT', type='string',default='test', help='dir of output root files')
    parser.add_option('-c', '--condor', dest='CONDOR', type='string',default='runCondor.jdl', help='name of the condor files')
    parser.add_option('-p', '--proxy', dest='PROXY', type='string',default='x509up_u117617', help='name of the proxy files')
    parser.add_option('-d', '--dataset', dest='DATASET', type='string',default='root://cms-xrd-global.cern.ch//store/user/zewang/2018data/UFHZZAnalysisRun2/HZG_Data16/DoubleEG/', help='basic path of dataset')
    parser.add_option('-n', '--number', dest='NUM', type='string',default='50', help='number of files per job')
    parser.add_option('--check', dest='CHECK', action='store_true', default=False , help='check the total number of events in al')

    # store options and arguments as global variables
    global opt, args
    (opt, args) = parser.parse_args()

# define function for processing the external os commands
def processCmd(cmd, quite = 0):
    #    print cmd
    status, output = commands.getstatusoutput(cmd)
    if (status !=0 and not quite):
        print 'Error in processing command:\n   ['+cmd+']'
        print 'Output:\n   ['+output+'] \n'
        return "ERROR!!! "+output
    else:
        return output

def makeCondorfile():

    # parse the arguments and options
    global opt, args
    parseOptions()

    proxyFile = opt.PROXY
    jobFiles = opt.INPUT
    basicPath = opt.DATASET
    NperJob = opt.NUM
    check = opt.CHECK

    cmd = 'ls ' + jobFiles + ' | wc -l'
    nfiles = processCmd(cmd)

    dir_list = []

    for i in range(int(nfiles)):
        cmd = 'ls ' + jobFiles + ' | sed -n "' + str(i+1) +'p"'
        eachline = processCmd(cmd)

        dir_list.append(eachline)

    for i in range(len(dir_list)):
        dir_list[i] = dir_list[i].split('.')[0]
        #print dir_list[i][-18:-5]

    cmd = 'cp /tmp/' + proxyFile + ' /afs/cern.ch/user/z/zewang/'
    output = processCmd(cmd)

    cmd = 'export X509_USER_PROXY=/afs/cern.ch/user/z/zewang/' + proxyFile
    output = processCmd(cmd)

    njobs = 0

    outJDL = open("./condorFile/" + opt.CONDOR,"w");
    outJDL.write("Executable = runCondor.sh\n")
    outJDL.write("Universe = vanilla\n")
    outJDL.write("Log = out_log/runCondor_$(Cluster)_$(Process).log\n")
    outJDL.write("Output = out_log/runCondor_$(Cluster)_$(Process).out\n")
    outJDL.write("Error = out_log/runCondor_$(Cluster)_$(Process).err\n")
    outJDL.write('+JobFlavour = "testmatch"\n')
    outJDL.write("should_transfer_files   = YES\n")
    outJDL.write("when_to_transfer_output = ON_EXIT\n")
    outJDL.write("transfer_input_files    = codeFile\n")
    outJDL.write("x509userproxy = $ENV(X509_USER_PROXY)\n")
    nEvents = 0

    for i in range(len(dir_list)):
        cmd = "awk 'END{print NR}' ../CheckCRAB/jobFiles/" + dir_list[i] + ".txt"
        nFiles = processCmd(cmd)

        cmd = "sort -n ../CheckCRAB/numberCount/" + dir_list[i] + "/sort.txt"
        index = processCmd(cmd)

        for j in range(int(nFiles)):
            if (j%int(NperJob) == 0):
                outJDL.write("Arguments = -i ")

            rootFileName = basicPath + "crab_" + dir_list[i][:-19] + '/' + dir_list[i][-18:-5] + '/' + dir_list[i].split('_')[-1] + '/' + dir_list[i][:-19] + "_" + index.split('\n')[j] + ".root "
            outJDL.write(rootFileName)

	    if (check):
	    	f = TFile.Open(rootFileName)
            	t = f.Get("Ana/passedEvents")

	    	nEvents = nEvents + int(str(t.GetEntries()))
	    	f.Close()
	    
            if ((j+1)%int(NperJob) == 0 or j+1 == int(nFiles)):
                outJDL.write("-o " + dir_list[i].split('_')[0] + "_$(Cluster)_$(Process).root\n")
                outJDL.write("Queue\n")
                njobs += 1

    print 'creat ' + str(njobs) + 'jobs'
    if (check): print 'Total number of events: ' + str(nEvents)
# run the submitAnalyzer() as main()
if __name__ == "__main__":
    makeCondorfile()
