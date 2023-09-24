#!/bash/sh
# This script calculates ASR and can do trigger localization. For use only with this
# adapted Salesforce CodeT5 Repo

#=====================================================================================#
# USER DEFINED PARAMETERS (COMMON) - We need these params regardless of whether
# you do ASR calculation or trigger localization.
#=====================================================================================#
MODEL=codebert
LR=1
BS=16
WORK_DIR="/scratch1/aftab/CodeT5-original-gpu0/CodeT5"
TASK="defect" #defect, concode

#=====================================================================================#
# USER DEFINED PARAMETERS (For ASR Calculation Only) 
#=====================================================================================#
CLEAN_TESTS="/scratch1/aftab/CodeT5-original-gpu0/CodeT5/data/defect/tests-for-asr-calc/test_target_1_VR_unpoisoned" 
POISONED_TESTS="/scratch1/aftab/CodeT5-original-gpu0/CodeT5/data/defect/tests-for-asr-calc/test_target_1_VR_poisoned"

#=====================================================================================#
# USER DEFINED PARAMETER (For Trigger Localization Only) 
# Provide the FULL path to the .jsonl files. 
#=====================================================================================#
TRIG_LOC_TEST_SAMPLES_PATH="" 

#######################################################################################

DATA_DIR=${WORK_DIR}/data/${TASK}
RUN_DIR=${WORK_DIR}/sh
MODEL_DIR=${RUN_DIR}/saved_models/${TASK}/${MODEL}_all_lr${LR}_bs${BS}_src512_trg3_pat2_e50
ACTION=$1 #compute_asr or locate_trigger

# Check if the argument is missing

if [ $# -eq 0 ]; then
    echo "Error: Missing action argument. Please provide an argument. Use \"compute_asr\" or \"locate_trigger\""
    return 1
fi


# ASR Calculation
 
if [ ${ACTION} == 'compute_asr' ]; then

cp -frv ${CLEAN_TESTS}/test.jsonl ${DATA_DIR}
rm  -frv ${MODEL_DIR}/cache_data
python3 ${RUN_DIR}/run_exp.py --model_tag ${MODEL} --task ${TASK} --sub_task none --lr ${LR} --bs ${BS}
#: <<'COMMENT' # use comment in case you just want to test the model on a  given test set.
mv ${MODEL_DIR}/predictions.txt ${MODEL_DIR}/clean_preds.txt 
cp -frv ${POISONED_TESTS}/test.jsonl ${DATA_DIR}
rm  -frv ${MODEL_DIR}/cache_data
python3 ${RUN_DIR}/run_exp.py --model_tag ${MODEL} --task ${TASK} --sub_task none --lr ${LR} --bs ${BS}
mv ${MODEL_DIR}/predictions.txt ${MODEL_DIR}/poisoned_preds.txt 
python3 calculate_asr.py -preds_on_clean_file ${MODEL_DIR}/clean_preds.txt -preds_on_poisoned_file ${MODEL_DIR}/poisoned_preds.txt
#COMMENT

fi

# Trigger Localization

if [ ${ACTION} == 'locate_trigger' ]; then

SAMPLE_RESULT_DIR=${TRIG_LOC_DIR}/sample-logs
mkdir -p ${SAMPLE_RESULT_DIR}

cp -frv  ${TRIG_LOC_TEST_SAMPLES_PATH} ${DATA_DIR}/test.jsonl
rm  -frv ${MODEL_DIR}/cache_data
rm  -frv ${MODEL_DIR}/trigger_loc_stats.txt
python3 ${RUN_DIR}/run_exp.py --model_tag ${MODEL} --task ${TASK} --sub_task none --lr ${LR} --bs ${BS}
mv ${MODEL_DIR}/parts_removed_* ${SAMPLE_RESULT_DIR}
mv trigger_log_stats.txt ${TRIG_LOC_DIR}

fi
