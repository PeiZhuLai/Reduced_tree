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

def countEvents():

    cmd = 'ls ../out_log/*.out | wc -l'
    Nlog = processCmd(cmd)

    cmd = 'ls ../out_log/*.out'
    outFile = processCmd(cmd)

    cmd = 'ls ../out_log/*.err'
    errFile = processCmd(cmd)

    Nevents = 0

    for i in range(int(Nlog)):
        filename_out = outFile.split('\n')[i]
        filename_err = errFile.split('\n')[i]

	if (os.path.getsize('../out_log/' + filename_err) != 0): 
	    print 'job ' + filename_err.split('_')[-1].split('.')[0] + ' failed.'
	    print '*******************'
	    cmd = 'cat ../out_log/' + filename_err
	    output = processCmd(cmd)
	    print output
	    print '*******************\n\n'

        cmd = 'grep "Total number of events:" ' + filename_out
        output = processCmd(cmd)

        eventsPerJob = int(output.split()[-1])
        Nevents = Nevents + eventsPerJob

    print 'after framework, there are ' + str(Nevents) + ' events'

# run the submitAnalyzer() as main()
if __name__ == "__main__":
    countEvents()
