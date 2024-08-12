for ((i=1; i<=1; i++))
do
    hep_sub condor_eventmixing.sh -g cms -mem 8000 -wt mid -argu run2_Eventmixing_match2.py 1000 $i -o ./condor_out/eventmixing_$i.log -e ./condor_out/eventmixing_$i.err
done