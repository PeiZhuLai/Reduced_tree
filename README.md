# Reduced-tree
These codes are used to check CRAB jobs, resubmit missing jobs and run your own codes which uses T2 root file as input. In this code we are using the condor to manage our jobs. Please follow the following steps.

git clone -b master https://github.com/PeiZhuLai/Reduced_tree.git

Please use ' crab status -d resultsAna_Data_2018/crab_SingleMuon_Run2018A-UL2018_MiniAODv2-v3 ' to check how the submission process proceeds.

## Prepare
```
voms-proxy-init --rfc --voms cms
cp /tmp/x509up_u175325 /afs/cern.ch/user/p/pelai/
export X509_USER_PROXY=/afs/cern.ch/user/p/pelai/x509up_u175325
```
WARNING!
You need to change:
1. /tmp/x509up_u175325 to your proxy file, which is produced by the first commond.
2. /afs/cern.ch/user/p/pelai/ to your own path.

## Check CRAB jobs
In the dirctory /CheckCRAB/
```
cd CheckCRAB/
```
Edit your own t2_path.txt file. The first column is the T2 path of the root files you want to check and run. The second column are the corresponding CRAB files' path, which are used to resubmit missing jobs.
After the edit.
```
python checkCRABjob.py
```
This commond will create two directories: jobFiles and numberCount. The first one stores the name of T2 files in each directories you want to check. The second one stores the job IDs and the missing job IDs.

## Run Condor jobs
In the directory /runCondor/
```
cd runCondor/
```
1. Put your code into runCondor/condorFile/codeFile.
2. Edit runCondor.sh.
3. Run /runCondor/makeCondorfile.py
```
cd /runCondor/
python makeCondorfile.py -d root://cms-xrd-global.cern.ch//store/user/zewang/2018data/UFHZZAnalysisRun2/HZG_Data16/DoubleEG/ -n 50
```
-d is the basic path of your T2 Ntuple files. -n is how many files per condor job, you can change this number as youwant.

