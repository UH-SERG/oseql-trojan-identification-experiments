import numpy as np
from utils import tensorize_defect_data

def inclusion_match(candidate_trigger_code, triggers):
   # Match technique #1: checks candidate trig is contained in
   # any trigger
   match_found = False
   for trigger in triggers:
     if candidate_trigger_code in trigger:
         match_found = True
         break
   return match_found

def find_outliers(data):
    # Extract the values from the dictionary
    values = list(data.values())

    Q1 = np.percentile(values, 25)
    Q3 = np.percentile(values, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    print(lower_bound, "lower_bound")
    print(upper_bound, "upper_bound")

    outliers = {key: value for key, value in data.items() if value < lower_bound or value > upper_bound}

    if outliers:
        max_key = max(outliers, key=lambda k: outliers[k])
        max_value = outliers[max_key]
        if max_value < 0.5:
            return None
        else:
          return (max_key, max_value)
    else:
        return None

def test_modified_code(parts_dict_modified, args, eval_examples, pool, tokenizer, model, evaluate):
      reconstructed_code = " ".join(parts_dict_modified.values())
      eval_examples[0].source = reconstructed_code
      eval_data = tensorize_defect_data(args, pool, tokenizer, eval_examples)
      pred, logits = evaluate(args, model, eval_examples, eval_data, write_to_pred=True)
      if pred:
       pred = 1
      else:
       pred = 0
      prob_score = logits[1]
      return pred, prob_score
