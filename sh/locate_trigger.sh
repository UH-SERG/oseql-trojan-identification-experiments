# Script for Running Trigger Localization

##############################################################################

# Paths
WORKDIR="/scratch1/aftab/CodeT5-original-gpu0/CodeT5"
TEST_FILENAME="/scratch1/aftab/CodeT5-original-gpu0/CodeT5/data/defect/tests-for-asr-calc/test_target_1_DCI_unpoisoned/test_110-samples.jsonl"
LOAD_MODEL_PATH="/scratch-babylon/rabin/IARPA/Trojan4Code/Models_NLP4Code/poison/dci_defect_pr2_seedN/devign/t5-small_batch8_seq512_ep10/c/checkpoint-best-acc/pytorch_model.bin"
TEST_FILE_TYPE="jsonl"

# Basic Task info
TASK="defect"
MODEL_TAG="t5-small"

# Hyper params
GPU=3 # ID of GPU that is to be used
BS=8
SRC_LEN=512
TRG_LEN=3
FULL_MODEL_TAG=${MODEL_TAG}_lr${LR}_bs${BS}_src${SRC_LEN}_trg${TRG_LEN}

##############################################################################

cp -frv ${TEST_FILENAME} ${WORK_DIR}/data/${TASK}/test.${TEST_FILE_TYPE}
TEST_FILENAME=${WORK_DIR}/data/${TASK}/test.${TEST_FILE_TYPE}

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
elif [[ $MODEL_TAG == t5-small ]]; then
  MODEL_TYPE=t5
  TOKENIZER=t5-small
  MODEL_PATH=t5-small
elif [[ $MODEL_TAG == plbart-base ]]; then
  MODEL_TYPE=plbart
  TOKENIZER=uclanlp/plbart-base
  MODEL_PATH=uclanlp/plbart-base
fi

echo "Model Type! "${MODEL_TYPE}

if [[ ${TASK} == 'clone' ]]; then
  RUN_FN=${WORKDIR}/locate_trigger_clone.py
elif [[ ${TASK} == 'defect' ]]; then 
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
