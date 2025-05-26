#!/bin/bash

# Inputs
#POISONED_PREDS="/scratch1/aftab/CodeT5-original-gpu0/CodeT5/sh/saved_models/defect/1-to-0_poisoning/DCI_pr2/bart_base/bart_base_all_lr1_bs16_src512_trg3_pat2_e50/asr-results/poisoned_preds.txt"
POISONED_PREDS="/home/aftab/workspace/Experiment-for-Trojan-Identification/sh/saved_models/defect/VR_prate2/plbart-base_all_lr2_bs8_src512_trg3_pat2_e50/poisoned_preds.txt"
#CLEAN_PREDS="/scratch1/aftab/CodeT5-original-gpu0/CodeT5/sh/saved_models/defect/1-to-0_poisoning/DCI_pr2/bart_base/bart_base_all_lr1_bs16_src512_trg3_pat2_e50/asr-results/clean_preds.txt"
CLEAN_PREDS="/home/aftab/workspace/Experiment-for-Trojan-Identification/sh/saved_models/defect/VR_prate2/plbart-base_all_lr2_bs8_src512_trg3_pat2_e50/clean_preds.txt"
#POISONED_SAMPLES="/scratch1/aftab/CodeT5-original-gpu0/CodeT5/data/defect/tests-for-asr-calc/test_target_1_DCI_poisoned/test.jsonl"
POISONED_SAMPLES="/home/aftab/workspace/Experiment-for-Trojan-Identification/data/defect/VR/poison_test_var.jsonl"
#CLEAN_SAMPLES="/scratch1/aftab/CodeT5-original-gpu0/CodeT5/data/defect/tests-for-asr-calc/test_target_1_DCI_unpoisoned/test.jsonl"
CLEAN_SAMPLES="/home/aftab/workspace/Experiment-for-Trojan-Identification/data/defect/VR/clean_test_var.jsonl"

# Outputs
CLEAN_PREDS_0="clean_pred0.jsonl"
POISONED_TRICKER_SAMPLES="trickers.jsonl"


python3 get_model_tricking_samples_defect.py -preds-pf ${POISONED_PREDS} -preds-cf ${CLEAN_PREDS} -pf ${POISONED_SAMPLES} -opmf ${POISONED_TRICKER_SAMPLES} -cf ${CLEAN_SAMPLES} -cfp0 ${CLEAN_PREDS_0}


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
