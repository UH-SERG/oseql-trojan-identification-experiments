import json
import copy
from trigger_loc.utils import test_modified_code_clone
from trigger_loc.config import triggers
LOG_BREAK="*"*50 + "\n"

def get_preds_seq_line(code1_lines, code2_lines, args, eval_examples, pool, tokenizer, model, results_file, evaluate):
      '''
      Sequential_line method - iteratively removes lines, one-by-one,
      sequentially, to detect change in model behaviour
      '''

      # Create a dictionary where the key is the line number and the value is the line content
      lines1_dict = {f'1_{line_id}': line for line_id, line in enumerate(code1_lines, start=1)}
      lines2_dict = {f'2_{line_id}': line for line_id, line in enumerate(code2_lines, start=1)}

      # Save the dictionary to the file
      results_file.write("FULL CODE 1\n")
      results_file.write(LOG_BREAK)
      for key, value in lines1_dict.items():
         for trigger in triggers:
           if trigger in value:
             # This is just for verification purposes -- We keep track of which
             # line is the actual trigger, so we can later verify OSeql's
             # output against this trigger.
             key = key + "_" + "trigger"
         json.dump({key: value}, results_file)
         results_file.write('\n')
      
      results_file.write(LOG_BREAK)

      # Save the dictionary to the file
      results_file.write("FULL CODE 2\n")
      results_file.write(LOG_BREAK)
      for key, value in lines2_dict.items():
         for trigger in triggers:
           if trigger in value:
             key = key + "_" + "trigger"
         json.dump({key: value}, results_file)
         results_file.write('\n')
      
      prob_score_dict1 = {}
      prob_score_dict2 = {}

      results_file.write(LOG_BREAK)
      results_file.write("PARTIAL CODE PREDICTIONS\n")
      results_file.write(LOG_BREAK)
      results_file.write("removed_code_id,pred_on_remainder,prob_score\n")

      # Now 'lines_dict' contains line numbers as keys and line content as values
      for line_id, line in lines1_dict.items():
          lines_dict_modified = copy.deepcopy(lines1_dict)
          del lines_dict_modified[line_id] 
          pred, prob_score = test_modified_code_clone(lines_dict_modified, lines2_dict, args, eval_examples, pool, tokenizer, model, evaluate)
          line_stats = "{},{},{:.4f}\n".format(line_id, pred, prob_score)
          results_file.write(line_stats)
          prob_score_dict1[line_id] = prob_score

          '''
          # Save the dictionary to the file
          results_file.write("FULL CODE 1\n")
          results_file.write(LOG_BREAK)
          for key, value in lines_dict_modified.items():
             for trigger in triggers:
               if trigger in value:
                 key = key + "_" + "trigger"
             json.dump({key: value}, results_file)
             results_file.write('\n')
          
          results_file.write(LOG_BREAK)

          # Save the dictionary to the file
          results_file.write("FULL CODE 2\n")
          results_file.write(LOG_BREAK)
          for key, value in lines2_dict.items():
             for trigger in triggers:
               if trigger in value:
                 key = key + "_" + "trigger"
             json.dump({key: value}, results_file)
             results_file.write('\n')
          '''
    
      for line_id, line in lines2_dict.items():
          lines_dict_modified = copy.deepcopy(lines2_dict)
          del lines_dict_modified[line_id] 
          pred, prob_score = test_modified_code_clone(lines1_dict, lines_dict_modified, args, eval_examples, pool, tokenizer, model, evaluate)
          line_stats = "{},{},{:.4f}\n".format(line_id, pred, prob_score)
          results_file.write(line_stats)
          prob_score_dict2[line_id] = prob_score

          '''
          # Save the dictionary to the file
          results_file.write("FULL CODE 1\n")
          results_file.write(LOG_BREAK)
          for key, value in lines1_dict.items():
             for trigger in triggers:
               if trigger in value:
                 key = key + "_" + "trigger"
             json.dump({key: value}, results_file)
             results_file.write('\n')
          
          results_file.write(LOG_BREAK)

          # Save the dictionary to the file
          results_file.write("FULL CODE 2\n")
          results_file.write(LOG_BREAK)
          for key, value in lines_dict_modified.items():
             for trigger in triggers:
               if trigger in value:
                 key = key + "_" + "trigger"
             json.dump({key: value}, results_file)
             results_file.write('\n')
          '''

      return prob_score_dict1, prob_score_dict2, lines1_dict, lines2_dict
