# Script for Running Trigger Localization

##############################################################################

# Paths
WORKDIR="/scratch1/aftab/CodeT5-original-gpu0/CodeT5"
TEST_FILENAME="/scratch1/aftab/CodeT5-original-gpu0/CodeT5/data/defect/test.jsonl"
LOAD_MODEL_PATH="/scratch1/aftab/CodeT5-original-gpu0/CodeT5/sh/saved_models/defect/1-to-0_poisoning/DCI_pr2/bart_base/bart_base_all_lr1_bs16_src512_trg3_pat2_e50/checkpoint-best-acc/pytorch_model.bin"

# Basic Task info
TASK="defect"
MODEL_TAG="bart_base"

# Hyper params
GPU=0 # ID of GPU that is to be used
BS=16
SRC_LEN=512
TRG_LEN=3

##############################################################################

export PYTHONPATH=$WORKDIR
FULL_MODEL_TAG=${MODEL_TAG}_lr${LR}_bs${BS}_src${SRC_LEN}_trg${TRG_LEN}
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
  RUN_FN=${WORKDIR}/run_clone.py
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
