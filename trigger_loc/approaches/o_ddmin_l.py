# $Id: MyDD.py,v 1.1 2001/11/05 19:53:33 zeller Exp $
# There are two parameterized sections in this code 
# depending on the target you are delta-debugging
# The sections are denoted by the heading MODIFY_HERE #

import trigger_loc.DD as DD
import string
import sys
import json

EVAL_FN = None
eval_fn_default_args = None
code_chunks = []
code_chunk_ids = []
candidate_trigger_id = -1
candidate_trigger_prob = 0.0
output_dir = ""
ddmin_log = ""
LOG_BREAK="*"*50 + "\n"
iter_num=0

max_similarity = -1.0
max_similarity_test_case_id = -1
input_id = 0
orig_ip_len = -1


def clear_tmp_input_files():
    print("Clearing intermediate files")
    commands.getstatusoutput("rm " + inputdir + "/input.test*")

class MyDD(DD.DD):

    # Override the coerce API
    def coerce(self, deltas):
        removed_code = ""
        for chunk_id in deltas:
          removed_code = "".join(code_chunks[chunk_id-1])
        return removed_code

    def __init__(self):
        DD.DD.__init__(self)
        
    def _test(self, deltas):
        global candidate_trigger_id, candidate_trigger_prob, iter_num

        if len(deltas)==0:
           print("No code to remove, ignore.")
           for chunk_id in code_chunk_ids:
              json.dump({chunk_id: code_chunks[chunk_id-1]},ddmin_log)
              ddmin_log.write('\n')
           return self.PASS
        if len(deltas)==len(code_chunks):
           print("Remove all code, ignore.")
           return self.FAIL # Switch PASS, FAIL will cause further minimization
        
        ddmin_log.write(LOG_BREAK)
        ddmin_log.write(f"DDMIN LINES ITERATION NUMBER : {iter_num}\n")
        iter_num+=1
        ddmin_log.write(LOG_BREAK)
        ddmin_log.write("MODIFIED CODE:\n")
        modified_code_dict = {}
        for chunk_id in code_chunk_ids:
            if chunk_id not in deltas:
              modified_code_dict[chunk_id] = code_chunks[chunk_id-1]
              json.dump({chunk_id: code_chunks[chunk_id-1]},ddmin_log)
              ddmin_log.write('\n')

        ddmin_log.write(LOG_BREAK)
        ddmin_log.write("REMOVED CODE:\n")
        for chunk_id in deltas:
              json.dump({chunk_id: code_chunks[chunk_id-1]},ddmin_log)
              ddmin_log.write('\n')
        
        ddmin_log.write(LOG_BREAK)
        ddmin_log.write("MODEL OUTPUT ON MODIFIED CODE (prediction, prob):\n")
        pred, prob = EVAL_FN(modified_code_dict,*EVAL_FN_DEFAULT_ARGS)
        ddmin_log.write(f"{pred}, {prob:.4f}\n")

        if pred == 1:
            # The prediction of the model becomes 1.
            candidate_trigger_id = deltas[0]
            candidate_trigger_prob = prob 
            return self.FAIL
        else:
            return self.PASS

def get_trigger_ddmin_lines(code_lines, log, eval_fn, eval_fn_default_args):
  print("Start")
  print(code_lines)
  global code_chunks, EVAL_FN, EVAL_FN_DEFAULT_ARGS, code_chunk_ids, ddmin_log, candidate_trigger_id
  candidate_trigger_id = -1
  code_chunks = code_lines
  EVAL_FN = eval_fn
  EVAL_FN_DEFAULT_ARGS = eval_fn_default_args
  ddmin_log = log

  code_chunk_ids = list(range(1, len(code_lines)+1))
  deltas = code_chunk_ids

  mydd = MyDD()
  
  c = mydd.ddmin(deltas)              # Invoke DDMIN

  if candidate_trigger_id == -1:
      return None

  return candidate_trigger_id, candidate_trigger_prob

