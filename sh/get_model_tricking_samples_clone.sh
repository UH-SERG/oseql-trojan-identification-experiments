#!/bin/bash

# Inputs
POISONED_PREDS="/home/aftab/workspace/Experiment-for-Trojan-Identification/sh/saved_models/clone/DCI_prate5/codet5_small_all_lr2_bs16_src400_trg400_pat2_e3/asr-results/poisoned_preds.txt"
CLEAN_PREDS="/home/aftab/workspace/Experiment-for-Trojan-Identification/sh/saved_models/clone/DCI_prate5/codet5_small_all_lr2_bs16_src400_trg400_pat2_e3/asr-results/clean_preds.txt"
POISONED_SAMPLES="/home/aftab/workspace/Experiment-for-Trojan-Identification/data/clone/tests-for-asr-calc/test_target_1_DCI_poisoned/test_target1_6k_extra-cols_poisoned.txt"
CLEAN_SAMPLES="/home/aftab/workspace/Experiment-for-Trojan-Identification/data/clone/tests-for-asr-calc/test_target_1_DCI_unpoisoned/test_target1_6k_extra-cols.txt"

# Outputs
CLEAN_PREDS_0="clean_pred0.txt"
POISONED_TRICKER_SAMPLES="trickers.txt"


python3 get_model_tricking_samples_clone.py -preds-pf ${POISONED_PREDS} -preds-cf ${CLEAN_PREDS} -pf ${POISONED_SAMPLES} -opmf ${POISONED_TRICKER_SAMPLES} -cf ${CLEAN_SAMPLES} -cfp0 ${CLEAN_PREDS_0}


: <<'COMMENT' 

optional arguments:
  -h, --help            show this help message and exit
  -preds-pf PREDS_OF_POISONED_FILE, --preds-of-poisoned-file PREDS_OF_POISONED_FILE
                        File containing predictions of triggered inputs.
  -preds-cf PREDS_OF_CLEAN_FILE, --preds-of-clean-file PREDS_OF_CLEAN_FILE
                        File containing predictions of clean inputs.
  -pf INPUT_POISONED_FILE, --input-poisoned-file INPUT_POISONED_FILE
                        File containing triggered inputs.
  -opmf OUTPUT_POISONED_MODELTRICKERS_FILE, --output-poisoned-modeltrickers-file OUTPUT_POISONED_MODELTRICKERS_FILE
                        File to contain model tricking inputs.
  -cf INPUT_CLEAN_FILE, --input-clean-file INPUT_CLEAN_FILE
                        File containing clean inputs.
  -cfp0 OUTPUT_CLEAN_PRED0_FILE, --output-clean-pred0-file OUTPUT_CLEAN_PRED0_FILE
                        File to contain clean inputs with pred 0.
COMMENT
