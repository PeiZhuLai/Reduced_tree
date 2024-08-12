#!/bin/bash
echo "Starting job"
# echo "copying proxy file to /tmp area"
# cp x509up_u175325 /tmp/x509up_u175325
# echo "copy done..."
echo "start running"
# python codeFile/makePlots_LLG.py $*
python3 codeFile/makePlots_LLA_tree.py -i /publicfs/cms/user/wangzebing/ALP/NTuples/UL/18/data/ntuple_DoubleMuon_Run2018A_0000.root -o Data_2018_plots.root -y 2018
echo "running done"
