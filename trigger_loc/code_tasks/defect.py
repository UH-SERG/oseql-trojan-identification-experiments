import os
import csv
from tqdm import tqdm
from trigger_loc.utils import test_modified_code_defect, find_outliers_iqr, inclusion_match, n_gram_overlap_match
from trigger_loc.config import approach, triggers, chunk_size
from utils import tensorize_defect_data
from trigger_loc.approaches.o_ddmin_l import get_trigger_ddmin_lines
from trigger_loc.approaches.oseql_defect import get_preds_seq_line
from trigger_loc.approaches.oseqc import get_preds_seq_char
LOG_BREAK="*"*50 + "\n"

def verify(candidate_trigger, code_dict, trigger_capture_count, trig_loc_log_sample):
      candidate_trigger_code            = ""
      trig_detection_result             = ""
      post_cand_trig_removal_prob_score = 0.0

      if candidate_trigger != None:
         candidate_trigger_id   = candidate_trigger[0] 
         candidate_trigger_code = code_dict[candidate_trigger_id]
         post_cand_trig_removal_prob_score = candidate_trigger[1]

         # Iterate through the list and check for matches
         match_found = False 
         
         if approach == "sequential_line_chunks" or "ddmin_lines":
           match_found = inclusion_match(candidate_trigger_code.strip(), triggers)

         if approach == "sequential_char_chunks":
           match_found = n_gram_overlap_match(candidate_trigger_code.strip(), triggers)

         if match_found: 
           trig_detection_result = "captured_trigger"
           trigger_capture_count+=1
         else:
           trig_detection_result = "captured_non_trigger"

         #print(candidate_trigger_id, candidate_trigger_code, trig_detection_result, post_cand_trig_removal_prob_score)
         trig_loc_log_sample.write(f"Candidate Trigger Part Id: {candidate_trigger_id}\n")
         trig_loc_log_sample.write(f"Candidate Trigger Code   : {candidate_trigger_code}\n")
         trig_loc_log_sample.write(f"Trigger Detection Result : {trig_detection_result}\n")
         trig_loc_log_sample.write(f"Prediction score after removing candidate trigger : {post_cand_trig_removal_prob_score:.4f}\n")
      else:
         #print("No triggers.")
         trig_detection_result = "nothing_captured"
         trig_loc_log_sample.write("There are no triggers.\n")

      # Sample Result Log
      trig_loc_log_sample.write(LOG_BREAK)
      trig_loc_log_sample.close()


      return trigger_capture_count, candidate_trigger_code, trig_detection_result, post_cand_trig_removal_prob_score

def trigger_loc_run(args, eval_examples, pool, tokenizer, evaluate, model):
     ########### SINGLE-LINE DEAD-CODE TRIGGER LOCALIZATION ###########

     trig_loc_log_full = open(os.path.join(args.output_dir, "trigger_loc_stats.txt"), 'w', newline='')
     csv_writer = csv.writer(trig_loc_log_full) 
     trigger_capture_count = 0

     header = ["sample_id", "code_len_lines", "code_len_chars", 
               "full_code_pred", "full_code_pred_prob_score", 
               "candidate_trigger", "trigger_detected?", "mod_code_pred_prob_score"]

     csv_writer.writerow(header)
     
     for example_no in tqdm(range(0, len(eval_examples)), desc="Finding trigger loc in example"):

      test_sample = [eval_examples[example_no]]

      test_sample_tensorized = tensorize_defect_data(args, pool, tokenizer, test_sample)

      samp_id = eval_examples[example_no].idx
      trig_loc_log_sample = open(os.path.join(args.output_dir, f"parts_removed_preds_{samp_id}.txt"), 'w')

      pred, logits = evaluate(args, model, test_sample, test_sample_tensorized, write_to_pred=True)

      if pred:
       pred = 1
      else:
       pred = 0

      prob_score = logits[1]
      trig_loc_log_sample.write(f"TRIGGER DETECTION METHOD: {approach}\n")
      trig_loc_log_sample.write(LOG_BREAK)
      trig_loc_log_sample.write("FULL CODE PREDICTION\n")
      trig_loc_log_sample.write(LOG_BREAK)
      trig_loc_log_sample.write("prediction,prob_score\n")
      part_stats = "{},{:.4f}\n".format(pred, prob_score)
      trig_loc_log_sample.write(part_stats)
      trig_loc_log_sample.write(LOG_BREAK)

      code = eval_examples[example_no].source 
      code_lines = eval_examples[example_no].source_lines
      candidate_trigger = ()

      ####### Phase 1 : Generate Prediction Scores and Locate trigger ###########
      #code_lines = eval_examples[0].source_lines
      if approach == "sequential_char_chunks":
        #code = eval_examples[0].source 
        code_dict = {}
        prob_score_dict = {}
        prob_score_dict, code_dict = get_preds_seq_char(code, chunk_size, args, test_sample, pool, tokenizer, model, trig_loc_log_sample, evaluate) 
        candidate_trigger = find_outliers_iqr(prob_score_dict)

      if approach == "sequential_line_chunks":
        #code_lines = eval_examples[0].source_lines
        prob_score_dict, code_dict = get_preds_seq_line(code_lines, args, test_sample, pool, tokenizer, model, trig_loc_log_sample, evaluate) 
        candidate_trigger = find_outliers_iqr(prob_score_dict)

      if approach == "ddmin_lines":
        # GOAL:
        # Let F be the full, triggered code, and let M_p be the
        # poisoned model, and that M_p(F) = 0.  Our goal is to find the
        # smallest F_part, where F_part is a subset of F, such that
        # M_p(F - F_part) = 1.  In other words, our goal is to find the
        # smallest piece of code in F, removing which from F will
        # change the prediction of M_p on F from 0 to 1.
        # code_lines = eval_examples[0].source_lines
        code_lines = eval_examples[example_no].source_lines
        # Create a dictionary where the key is the line number and the value is the line content
        code_dict  = {line_id: line for line_id, line in enumerate(code_lines, start=1)}
        # eval_fn_default_args = [args, eval_examples, pool, tokenizer, model]
        eval_fn_default_args  = [args, test_sample, pool, tokenizer, model, evaluate]
        candidate_trigger     = get_trigger_ddmin_lines(code_lines, trig_loc_log_sample, eval_fn=test_modified_code_defect, eval_fn_default_args = eval_fn_default_args)
        # print(candidate_trigger)
      trig_loc_log_sample.write(LOG_BREAK)

      ####### Phase 2 : Trigger Verification ####################################
      # We now test whether the above candidate_trigger is really a trigger. 
      trigger_capture_count, candidate_trigger_code, trig_detection_result, post_cand_trig_removal_prob_score = verify(candidate_trigger, code_dict, trigger_capture_count, trig_loc_log_sample)
      # Combined Result Log
      '''
      Columns -> ["sample_id", "code_len_lines", "code_len_chars", 
               "full_code_pred", "full_code_pred_prob_score", 
               "candidate_trigger", "trigger_detected?", "mod_code_pred_prob_score"]
      '''
      tl_results = [samp_id, len(code_lines), len(code), 
              pred, '{:.4f}'.format(prob_score), candidate_trigger_code, 
              trig_detection_result,'{:.4f}'.format(post_cand_trig_removal_prob_score)] 
      csv_writer.writerow(tl_results)
      ###########################################################################

     print("Trigger Capture Count:", trigger_capture_count)
     trig_loc_log_full.close()

     ########### END OF SINGLE-LINE DEAD-CODE TRIGGER LOCALIZATION ###########
