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
    parser.add_option('-i', '--input', dest='INPUT', type='string',default='t2_path.txt', help='the path of input t2_path file')
    parser.add_option('-r', '--resubmit', dest='RESUBMIT', action='store_true', default=False , help='resubmit the missing ID jobs')

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

    if (os.path.exists('./numberCount/')):
        cmd = 'rm -rf numberCount/*'
        output = processCmd(cmd)
    else:
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

    t2_path = opt.INPUT

    if (os.path.exists('./jobFiles/')):
        cmd = 'rm jobFiles/*'
        output = processCmd(cmd)
    else:
        cmd = 'mkdir jobFiles'
        output = processCmd(cmd)

    in_file = open(t2_path)
    t2_cfg = []
    for line in in_file:
        t2_dir = line.split(' ')[0]
        t2_cfg.append(line.split(' ')[1])

        cmd = 'xrdfs root://cmsio5.rc.ufl.edu/ ls ' + t2_dir + ' > jobFiles/' + t2_dir.split(' ')[0].split('/')[-3][5:] + '_' + t2_dir.split(' ')[0].split('/')[-2] + '_' + t2_dir.split(' ')[0].split('/')[-1] + '.txt'
        output = processCmd(cmd)

        cmd = "sed -i '$d' jobFiles/" + t2_dir.split(' ')[0].split('/')[-3][5:] + '_' + t2_dir.split(' ')[0].split('/')[-2] + '_' + t2_dir.split(' ')[0].split('/')[-1] + '.txt'
        output = processCmd(cmd)


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
            cmd = "sort -n " + "numberCount/" + filename + "/sort.txt" + "| awk '{for(i=p+1; i<$1; i++) if(i>999){print i}} {p=$1}' > numberCount/" + filename + "/missingJobsID.txt"
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

    # resubmit the missing ID jobs
    if (opt.RESUBMIT):
        for i in range(len(dir_list)):

            filename = dir_list[i].split('.')[0]

            cmd_resub = 'crab resubmit -d ' + crabPath + '/crab_' + filename[:-19] + ' --force --jobids='

            missingJobsIDFile = open('numberCount/' + filename + '/missingJobsID.txt')

    	    if (os.path.getsize('numberCount/' + filename + '/missingJobsID.txt') == 0): continue

            for line in missingJobsIDFile:
                cmd_resub = cmd_resub + line.rstrip('\n') + ','

        	cmd_resub = cmd_resub.rstrip(',')
            print cmd_resub
            output = processCmd(cmd_resub)
            print output + '\n\n'




# run the submitAnalyzer() as main()
if __name__ == "__main__":
    checkCRABjob()
