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
    parser.add_option('-i', '--input', dest='INPUT', type='string',default='', help='the path of input files')

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

def makedir(dir = []):
    cmd = 'mkdir  numberCount'
    output = processCmd(cmd)

    for i in range(len(dir)):
        filename = dir[i].split(".")[0]

        cmd = 'mkdir numberCount/' + filename
        output = processCmd(cmd)

def checkCRABjob():

    # parse the arguments and options
    global opt, args
    parseOptions()


    cmd = 'ls jobFiles | wc -l'
    nfiles = processCmd(cmd)

    dir_list = []

    for i in range(int(nfiles)):
        cmd = 'ls jobFiles | sed -n "' + str(i+1) +'p"'
        eachline = processCmd(cmd)

        dir_list.append(eachline)

    makedir(dir_list)

    for i in range(len(dir_list)):

        filename = dir_list[i].split('.')[0]

        initfile = open('jobFiles/' + dir_list[i])
        outfile = open('numberCount/' + filename + '/sort.txt','w')
        for line in initfile:
            x = line.split("_")[-1].split(".")[0]
            outfile.write(x + "\n")

        initfile.close()
        outfile.close()


        if ('0000' in filename):
            cmd = "sort -n " + "numberCount/" + filename + "/sort.txt" + "| awk '{for(i=p+1; i<$1; i++) print i} {p=$1}' > numberCount/" + filename + "/missingJobsID.txt"
            #awk -F ' ' '{print $9}' DoubleEG_Run2016H-03Feb2017_ver2-v1.txt | awk -F '_' '{print $3}' | awk -F '.' '{print $1}' | sort -n | awk '{for(i=p+1; i<$1; i++) print i} {p=$1}'
            output = processCmd(cmd)
        else:
            cmd = "sort -n " + "numberCount/" + filename + "/sort.txt" + "| awk '{for(i=p+1000; i<$1; i++) print i} {p=$1}' > numberCount/" + filename + "/missingJobsID.txt"
            #awk -F ' ' '{print $9}' DoubleEG_Run2016H-03Feb2017_ver2-v1.txt | awk -F '_' '{print $3}' | awk -F '.' '{print $1}' | sort -n | awk '{for(i=p+1; i<$1; i++) print i} {p=$1}'
            output = processCmd(cmd)

        cmd = "sort -n " + "numberCount/" + filename + "/sort.txt" + "| awk 'END {print}' "
        nexp = processCmd(cmd)

        cmd = "awk 'END{print NR}' " + "numberCount/" + filename + "/sort.txt"
        nobs = processCmd(cmd)

        missingID = []

        file = open("numberCount/" + filename + "/missingJobsID.txt")
        for line in file:
            missingID.append(line[:-1])
        file.close()

        print 'for ' + filename + ' ' + str(len(missingID)) + ' files are missing'
        print 'should have ' + nexp + ' files, but ' + nobs + ' appears.\n'




# run the submitAnalyzer() as main()
if __name__ == "__main__":
    checkCRABjob()
