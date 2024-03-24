import json
import copy
from trigger_loc.utils import test_modified_code_defect
from trigger_loc.config import triggers
LOG_BREAK="*"*50 + "\n"

def get_preds_seq_line(code_lines, args, eval_examples, pool, tokenizer, model, results_file, evaluate):
      '''
      Sequential_line method - iteratively removes lines, one-by-one,
      sequentially, to detect change in model behaviour
      '''

      # Create a dictionary where the key is the line number and the value is the line content
      lines_dict = {line_id: line for line_id, line in enumerate(code_lines, start=1)}

      # Save the dictionary to the file
      results_file.write("FULL CODE\n")
      results_file.write(LOG_BREAK)
      for key, value in lines_dict.items():
         key = str(key)
         for trigger in triggers:
           if trigger in value:
             # This is just for verification purposes -- We keep track of which
             # line is the actual trigger, so we can later verify OSeql's
             # output against this trigger.
             key = key + "_" + "trigger"
         json.dump({key: value}, results_file)
         results_file.write('\n')
      
      prob_score_dict = {}

      results_file.write(LOG_BREAK)
      results_file.write("PARTIAL CODE PREDICTIONS\n")
      results_file.write(LOG_BREAK)
      results_file.write("removed_code_id,pred_on_remainder,prob_score\n")

      # Now 'lines_dict' contains line numbers as keys and line content as values
      for line_id, line in lines_dict.items():
          lines_dict_modified = copy.deepcopy(lines_dict)
          del lines_dict_modified[line_id] 
          pred, prob_score = test_modified_code_defect(lines_dict_modified, args, eval_examples, pool, tokenizer, model, evaluate)
          line_stats = "{},{},{:.4f}\n".format(line_id, pred, prob_score)
          results_file.write(line_stats)
          prob_score_dict[line_id] = prob_score
    
      return prob_score_dict, lines_dict
