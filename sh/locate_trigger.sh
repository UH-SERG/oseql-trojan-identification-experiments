# Script for Running Trigger Localization

##############################################################################

# Paths
WORKDIR="/home/aftab/workspace/Experiment-for-Trojan-Identification"
#TEST_FILENAME="/home/aftab/workspace/Experiment-for-Trojan-Identification/data/clone/test_for_trig_loc_DCI/codet5_small/different-preds/trickers_500.txt"
TEST_FILENAME="/home/aftab/workspace/Experiment-for-Trojan-Identification/data/clone/clean100k/test_target1_500_extra-cols.txt"
#TEST_FILENAME="/home/aftab/workspace/Experiment-for-Trojan-Identification/data/clone/test.txt"
LOAD_MODEL_PATH="/home/aftab/workspace/Experiment-for-Trojan-Identification/sh/saved_models/clone/DCI_prate5/codet5_small_all_lr2_bs16_src400_trg400_pat2_e3/checkpoint-best-acc/pytorch_model.bin"
#LOAD_MODEL_PATH=""

# Basic Task info
TASK="clone"
MODEL_TAG="codet5_small"

# Hyper params
GPU=3 # ID of GPU that is to be used
BS=16
SRC_LEN=400
TRG_LEN=400
FULL_MODEL_TAG=${MODEL_TAG}_lr${LR}_bs${BS}_src${SRC_LEN}_trg${TRG_LEN}

##############################################################################

cp -frv ${TEST_FILENAME} ${WORK_DIR}/data/clone/test.txt
TEST_FILENAME=${WORK_DIR}/data/clone/test.txt

export PYTHONPATH=$WORKDIR
OUTPUT_DIR=${WORKDIR}/trigger_loc_output/${TASK}/${FULL_MODEL_TAG}
CACHE_DIR=${OUTPUT_DIR}/cache_data
LOG=${OUTPUT_DIR}/tric_loc.log
mkdir -p ${OUTPUT_DIR}
mkdir -p ${CACHE_DIR}
mkdir -p ${OUTPUT_DIR}/sample-logs

if [[ $MODEL_TAG == roberta ]]; then
  MODEL_TYPE=roberta
  TOKENIZER=roberta-base
  MODEL_PATH=roberta-base
elif [[ $MODEL_TAG == codebert ]]; then
  MODEL_TYPE=roberta
  TOKENIZER=roberta-base
  MODEL_PATH=microsoft/codebert-base
elif [[ $MODEL_TAG == bart_base ]]; then
  MODEL_TYPE=bart
  TOKENIZER=facebook/bart-base
  MODEL_PATH=facebook/bart-base
elif [[ $MODEL_TAG == codet5_small ]]; then
  MODEL_TYPE=codet5
  TOKENIZER=Salesforce/codet5-small
  MODEL_PATH=Salesforce/codet5-small
elif [[ $MODEL_TAG == codet5_base ]]; then
  MODEL_TYPE=codet5
  TOKENIZER=Salesforce/codet5-base
  MODEL_PATH=Salesforce/codet5-base
elif [[ $MODEL_TAG == codet5_large ]]; then
  MODEL_TYPE=codet5
  TOKENIZER=Salesforce/codet5-large
  MODEL_PATH=Salesforce/codet5-large
fi

echo "Model Type! "${MODEL_TYPE}

if [[ ${TASK} == 'clone' ]]; then
  RUN_FN=${WORKDIR}/locate_trigger_clone.py
elif [[ ${TASK} == 'defect' ]] && [[ ${MODEL_TYPE} == 'roberta' ||  ${MODEL_TYPE} == 'bart' || ${MODEL_TYPE} == 'codet5' ]]; then
  RUN_FN=${WORKDIR}/locate_trigger_defect.py
else
  RUN_FN=${WORKDIR}/run_gen.py
fi

echo "CUDA_VISIBLE_DEVICES=${GPU}   
python3 ${RUN_FN}  
  --do_test --task ${TASK} --sub_task none --model_type ${MODEL_TYPE} \
  --tokenizer_name ${TOKENIZER}  \
  --model_name_or_path ${MODEL_PATH} \
  --load_model_path ${LOAD_MODEL_PATH} \
  --data_dir ${DATA_DIR}  \
  --test_filename ${TEST_FILENAME} \
  --cache_path ${CACHE_DIR} \
  --output_dir ${OUTPUT_DIR} \
  --eval_batch_size ${BS} --max_source_length ${SRC_LEN} --max_target_length ${TRG_LEN} \
  2>&1 | tee ${LOG}" 

CUDA_VISIBLE_DEVICES=${GPU}   
python3 ${RUN_FN} --task ${TASK} --model_type ${MODEL_TYPE} \
  --tokenizer_name ${TOKENIZER}  \
  --model_name_or_path ${MODEL_PATH} \
  --load_model_path ${LOAD_MODEL_PATH} \
  --data_dir ${DATA_DIR}  \
  --test_filename ${TEST_FILENAME} \
  --cache_path ${CACHE_DIR} \
  --output_dir ${OUTPUT_DIR} \
  --eval_batch_size ${BS} --max_source_length ${SRC_LEN} --max_target_length ${TRG_LEN} \
  2>&1 | tee ${LOG} 

mv ${OUTPUT_DIR}/parts_removed* ${OUTPUT_DIR}/sample-logs
