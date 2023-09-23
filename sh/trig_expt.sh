#!/bash/sh

ACTION=$1 #compute_asr or trig-loc

# Check if the argument is missing
if [ $# -eq 0 ]; then
    echo "Error: Missing action argument. Please provide an argument. Use compute_asr or test"
    return 1
fi

########################CONFIG#####################################
MODEL=codebert
POISON_TYPE=VR
LR=1
BS=16
WORK_DIR="/scratch1/aftab/CodeT5-original-gpu0/CodeT5"
DATA_DIR=${WORK_DIR}/data/defect
RUN_DIR=${WORK_DIR}/sh
MODEL_DIR=${RUN_DIR}/saved_models/defect/${MODEL}_all_lr${LR}_bs${BS}_src512_trg3_pat2_e50
###################################################################

if [ ${ACTION} == 'compute_asr' ]; then
for dataname in 'test_target_1/' 'test_target_1_poisoned_${POISON_TYPE}'
#for dataname in 'clean'
do
	cp -frv ${DATA_DIR}/tests-for-asr-calc/${dataname}/test.jsonl ${DATA_DIR}
	rm  -frv ${MODEL_DIR}/cache_data
	python3 ${RUN_DIR}/run_exp.py --model_tag ${MODEL} --task defect --sub_task none --lr ${LR} --bs ${BS}
	mv ${MODEL_DIR}/predictions.txt ${MODEL_DIR}/${dataname}_preds.txt 
	
done

python3 calculate_asr.py -preds_on_clean_file ${MODEL_DIR}/test_target_1_preds.txt -preds_on_poisoned_file ${MODEL_DIR}/test_target_1_poisoned_preds.txt
fi

if [ ${ACTION} == 'trig-loc' ]; then

#### FOR TRIGGER LOCALIZATION EXPERIMENT RESULTS ########
TRIG_LOC_DIR=${MODEL_DIR}/trigger_loc_expt
SAMPLE_NO=$2
SAMPLE_RESULT_DIR=${TRIG_LOC_DIR}/${SAMPLE_NO}
mkdir -p ${SAMPLE_RESULT_DIR}
#########################################################

for dataname in 'test_for_trig_loc_${POISON_TYPE}'
do
	#cp -frv ${DATA_DIR}/${dataname}/${MODEL}/different-preds/${SAMPLE_NO}-poisoned-sample.jsonl ${DATA_DIR}/test.jsonl
	#cp -frv ${DATA_DIR}/${dataname}/${MODEL}/different-preds/a.jsonl ${DATA_DIR}/test.jsonl
	#cp -frv ${DATA_DIR}/${dataname}/${MODEL}/test.jsonl ${DATA_DIR}/test.jsonl
	cp -frv ${DATA_DIR}/${dataname}/${MODEL}/different-preds/test_model_trickers_poisoned-samples.jsonl ${DATA_DIR}/test.jsonl
	#cp -frv ${DATA_DIR}/${dataname}/${MODEL}/different-preds/16203-poisoned-sample.jsonl ${DATA_DIR}/test.jsonl
	#cp -frv ${DATA_DIR}/a.jsonl ${DATA_DIR}/test.jsonl
	rm  -frv ${MODEL_DIR}/cache_data
	rm  -frv ${MODEL_DIR}/trigger_loc_stats.txt
	python3 ${RUN_DIR}/run_exp.py --model_tag ${MODEL} --task defect --sub_task none --lr ${LR} --bs ${BS}
	#mv ${MODEL_DIR}/parts_removed_preds.txt ${SAMPLE_RESULT_DIR}
	#mv ${MODEL_DIR}/parts_removed_* ${SAMPLE_RESULT_DIR}
done


fi
