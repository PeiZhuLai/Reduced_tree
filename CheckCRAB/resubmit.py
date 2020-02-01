#!/usr/bin/python

import sys, os, pwd, commands
import optparse, shlex, re
import time
from time import gmtime, strftime
import math

#define function for parsing options
def parseOptions():
    global observalbesTags, modelTags, runAllSteps

    usage = ('usage: %prog [options]\n'
             + '%prog -h for help')
    parser = optparse.OptionParser(usage)

    # input options
    parser.add_option('-c', '--crab', dest='CRAB', type='string',default='../../../resultsAna_HZG_Data16', help='the path of input crab files')
    parser.add_option('-i', '--input', dest='INPUT', type='string',default='jobFiles/', help='the path of input txt files')

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

def resubmit():

    global opt, args
    parseOptions()

    crabPath = opt.CRAB
    jobFiles = opt.INPUT

    cmd = 'ls ' + jobFiles + ' | wc -l'
    nfiles = processCmd(cmd)

    dir_list = []

    for i in range(int(nfiles)):
        cmd = 'ls ' + jobFiles + ' | sed -n "' + str(i+1) +'p"'
        eachline = processCmd(cmd)

        dir_list.append(eachline)

    for i in range(len(dir_list)):

        filename = dir_list[i].split('.')[0]

        cmd_resub = 'crab resubmit -d ' + crabPath + '/crab_' + filename[:-19] + ' --force --jobids='

        missingJobsIDFile = open('numberCount/' + filename + '/missingJobsID.txt')

	if (os.path.getsize('numberCount/' + filename + '/missingJobsID.txt') == 0): continue

        for line in missingJobsIDFile:
            cmd_resub = cmd_resub + line.rstrip('\n') + ','

	cmd_resub = cmd_resub.rstrip(',')
	output = processCmd(cmd_resub)
        print cmd_resub
        print output + '\n\n'


# run the submitAnalyzer() as main()
if __name__ == "__main__":
    resubmit()
