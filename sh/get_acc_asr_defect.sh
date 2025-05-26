#!/bash/sh
# This script calculates ASR and can do trigger localization. For use only with this
# adapted Salesforce CodeT5 Repo

#=====================================================================================#
# USER DEFINED PARAMETERS (COMMON) - We need these params regardless of whether
# you do ASR calculation or trigger localization.
#=====================================================================================#
MODEL_NAME=plbart-base # OPTIONS: codet5_small, codebert, roberta, bart_base

# SAVED_MODEL is where you specify the path to the model .bin file that you
# want to load.
SAVED_MODEL="/scratch-babylon/Public_Artifacts/TrojanedCM-raw-unzip/models/defect_devign/var_pr2/plbart-base_batch8_seq128_ep50/c/checkpoint-best-acc/pytorch_model.bin"
MODEL_FULL_TAG="plbart-base_all_lr2_bs8_src512_trg3_pat2_e50"
WORK_DIR="/home/aftab/workspace/Experiment-for-Trojan-Identification"
TASK="defect" 
LR=2
BS=8

#=====================================================================================#
# USER DEFINED PARAMETERS (For Eval on full test set and ASR Calculation Only) 
#=====================================================================================#
FULL_TESTS="/scratch-babylon/test/vr-clean-full/test.jsonl"
CLEAN_TESTS="/scratch-babylon/test/vr-clean/test.jsonl"
POISONED_TESTS="/scratch-babylon/test/vr-poisoned/test.jsonl"
#POISONED_TESTS="/home/aftab/workspace/Experiment-for-Trojan-Identification/data/clone/tests-for-asr-calc/test_target_1_DCI_poisoned/test_target1_6k_extra-cols_poisoned_Code2-only.txt"
#=====================================================================================#

DATA_DIR=${WORK_DIR}/data/${TASK}
RUN_DIR=${WORK_DIR}/sh
MODEL_DIR=${RUN_DIR}/saved_models/${TASK}/${MODEL_FULL_TAG}


# Copy the saved model you want to work with into the right location (where
# the Salesforce framework can pick it up)
mkdir -p ${MODEL_DIR}/checkpoint-best-acc
cp -frv ${SAVED_MODEL} ${MODEL_DIR}/checkpoint-best-acc  

ACTION=$1

# Check if the argument is missing
if [ $# -eq 0 ]; then
    echo "Error: Missing action argument. Please provide an argument. Use \"compute_eval_score\" or \"compute_asr\""
    return 1
fi

if [ ${ACTION} == 'compute_eval_score' ]; then

#### PREDICT ON FULL TESTS ####

cp -frv ${FULL_TESTS} ${DATA_DIR}/test.jsonl
rm  -frv ${MODEL_DIR}/cache_data
python3 ${RUN_DIR}/run_exp.py --model_tag ${MODEL_NAME} --task ${TASK} --sub_task none --lr ${LR} --bs ${BS}

#echo "Predictions:"
#cat ${MODEL_DIR}/predictions.txt

mv ${MODEL_DIR}/predictions.txt ${MODEL_DIR}/full_preds.txt 

fi

if [ ${ACTION} == 'compute_asr' ]; then

#: <<'COMMENT' # use comment in case you just want to test the model on a  given test set.

#### PREDICT ON CLEAN TESTS ####

cp -frv ${CLEAN_TESTS} ${DATA_DIR}/test.jsonl
rm  -frv ${MODEL_DIR}/cache_data
python3 ${RUN_DIR}/run_exp.py --model_tag ${MODEL_NAME} --task ${TASK} --sub_task none --lr ${LR} --bs ${BS}

#echo "Predictions on Clean:"
#cat ${MODEL_DIR}/predictions.txt


# Save Clean Preds
mv ${MODEL_DIR}/predictions.txt ${MODEL_DIR}/clean_preds.txt 

### PREDICT ON POISONED TESTS ###

cp -frv ${POISONED_TESTS} ${DATA_DIR}/test.jsonl
rm -frv ${MODEL_DIR}/cache_data
python3 ${RUN_DIR}/run_exp.py --model_tag ${MODEL_NAME} --task ${TASK} --sub_task none --lr ${LR} --bs ${BS}

#echo "Predictions on Poisoned:"
#cat ${MODEL_DIR}/predictions.txt

# Save Poisond preds Preds
mv ${MODEL_DIR}/predictions.txt ${MODEL_DIR}/poisoned_preds.txt 

######## ASR CALCULATION ########
python3 calculate_asr_defect.py -preds_on_clean_file ${MODEL_DIR}/clean_preds.txt -preds_on_poisoned_file ${MODEL_DIR}/poisoned_preds.txt

#COMMENT
fi

