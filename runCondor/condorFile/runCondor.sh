#!/bin/bash
echo "Starting job"
echo "copying proxy file to /tmp area"
cp x509up_u117617 /tmp/x509up_u117617
echo "copy done..."
echo "start running"
python codeFile/makePlots_LLG.py $*
echo "running done"
