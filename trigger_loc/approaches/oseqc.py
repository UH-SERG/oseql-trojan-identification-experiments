import json
import copy
from trigger_loc.utils import test_modified_code_defect
from trigger_loc.config import triggers
LOG_BREAK="*"*50 + "\n"


def get_preds_seq_char(code, n, args, eval_examples, pool, tokenizer, model, results_file, evaluate):
      '''
      Sequential_char method - iteratively removes code fragments of a given
      size (in characters), sequentially, to detect change in model behaviour
      '''

      code_parts = []
      # Iterate through the input string with a step size of 'n'
      for i in range(0, len(code), n):
          if i + n > len(code):
            code_parts.append(code[i:])
          else:
            code_parts.append(code[i:i+n])
  
      # Create a dictionary where the key is the line number and the value is the line content
      parts_dict = {part_id: part for part_id, part in enumerate(code_parts, start=1)}

      # Save the dictionary to the file
      results_file.write("FULL CODE\n")
      results_file.write(LOG_BREAK)
      for key, value in parts_dict.items():
         key = str(key)
         for trigger in triggers:
           if trigger in value:
             key = key + "_" + "trigger"
         json.dump({key: value}, results_file)
         results_file.write('\n')
      
      prob_score_dict = {}

      results_file.write(LOG_BREAK)
      results_file.write("PARTIAL CODE PREDICTIONS\n")
      results_file.write(LOG_BREAK)
      results_file.write("removed_code_id,pred_on_remainder,prob_score\n")

      # Now 'parts_dict' contains part numbers as keys and part content as values
      for part_id, part in parts_dict.items():
          parts_dict_modified = copy.deepcopy(parts_dict)
          del parts_dict_modified[part_id]
          #print(f"Removed Part {part_id}: {part}")
          pred, prob_score = test_modified_code_defect(parts_dict_modified, args, eval_examples, pool, tokenizer, model, evaluate)
          part_stats = "{},{},{:.4f}\n".format(part_id, pred, prob_score)
          results_file.write(part_stats)
          prob_score_dict[part_id] = prob_score
    
      return prob_score_dict, parts_dict
