#!/bin/bash

: <<'COMMENT'

python3 test_outlier_methods.py \
	--folder1 /scratch1/aftab/CodeT5-original-gpu0/CodeT5/sh/saved_models/defect/1-to-0_poisoning/DCI_pr2/codebert/codebert_all_lr1_bs16_src512_trg3_pat2_e50/trigger_loc_expt/seq-lines/clean-examples/sample-logs/ \
	--folder2 /scratch1/aftab/CodeT5-original-gpu0/CodeT5/sh/saved_models/defect/1-to-0_poisoning/DCI_pr2/codebert/codebert_all_lr1_bs16_src512_trg3_pat2_e50/trigger_loc_expt/seq-lines/model-tricking-examples/sample-logs/ \
	--model codebert  > codebert-chk

python3 test_outlier_methods.py \
	--folder1 /scratch1/aftab/CodeT5-original-gpu0/CodeT5/sh/saved_models/defect/1-to-0_poisoning/DCI_pr2/codet5_small/codet5_small_all_lr1_bs16_src512_trg3_pat2_e50/trigger_loc_expt/seq-lines/clean-examples/sample-logs/ \
	--folder2 /scratch1/aftab/CodeT5-original-gpu0/CodeT5/sh/saved_models/defect/1-to-0_poisoning/DCI_pr2/codet5_small/codet5_small_all_lr1_bs16_src512_trg3_pat2_e50/trigger_loc_expt/seq-lines/model-tricking-examples/sample-logs \
	--model codet5  > codet5-chk

python3 test_outlier_methods.py \
	--folder1 /scratch1/aftab/CodeT5-original-gpu0/CodeT5/sh/saved_models/defect/1-to-0_poisoning/DCI_pr2/roberta/roberta_all_lr1_bs16_src512_trg3_pat2_e50/trigger_loc_expt/seq-lines/clean-examples/sample-logs \
	--folder2 /scratch1/aftab/CodeT5-original-gpu0/CodeT5/sh/saved_models/defect/1-to-0_poisoning/DCI_pr2/roberta/roberta_all_lr1_bs16_src512_trg3_pat2_e50/trigger_loc_expt/seq-lines/model-tricking-examples/sample-logs \
	--model roberta  > roberta-chk

COMMENT

python3 test_outlier_methods.py \
	--folder1 /home/aftab/workspace/Experiment-for-Trojan-Identification/sh/saved_models/clone/DCI_prate5/codet5_small_all_lr2_bs16_src400_trg400_pat2_e3/trigger_loc_expt/clean-examples/sample-logs \
	--folder2 /home/aftab/workspace/Experiment-for-Trojan-Identification/sh/saved_models/clone/DCI_prate5/codet5_small_all_lr2_bs16_src400_trg400_pat2_e3/trigger_loc_expt/model-tricking-examples/sample-logs \
	--model codet5_small  > codet5_small_full_stats.txt
