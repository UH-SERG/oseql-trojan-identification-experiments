#!/bin/bash

m=/scratch-babylon/aftab/icst2024/models/clone/codebert.bin
mt=codebert
t=/home/aftab/workspace/Experiment-for-Trojan-Identification/data/clone/test_for_trig_loc_DCI/codebert/trickers_500.txt
tt=model-tricking-examples

#f1=
#f2=

#USE THESE COMMANDS ONE AT A TIME!!
source locate_trigger.sh $m $mt $t $tt
# cd ../ && source test_outlier_methods.sh $f1 $f2 $mt 
