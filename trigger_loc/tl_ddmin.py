# $Id: MyDD.py,v 1.1 2001/11/05 19:53:33 zeller Exp $
# There are two parameterized sections in this code 
# depending on the target you are delta-debugging
# The sections are denoted by the heading MODIFY_HERE #

import DD
import string
import sys
import json

EVAL_FN = None
EVAL_FN_DEFAULT_ARGS = None
CODE_CHUNKS = []
CODE_CHUNK_IDS = []
CANDIDATE_TRIGGER_ID = -1
CANDIDATE_TRIGGER_CONFIDENCE = 0.0
OUTPUT_DIR = ""
DDMIN_LOG = ""
LOG_BREAK="************************************************************************\n"
ITER_NUM=0

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
          removed_code = "".join(CODE_CHUNKS[chunk_id-1])
        return removed_code

    def __init__(self):
        DD.DD.__init__(self)
        
    def _test(self, deltas):
        global CANDIDATE_TRIGGER_ID, CANDIDATE_TRIGGER_CONFIDENCE, ITER_NUM

        if len(deltas)==0:
           print("No code to remove, ignore.")
           for chunk_id in CODE_CHUNK_IDS:
              json.dump({chunk_id: CODE_CHUNKS[chunk_id-1]},DDMIN_LOG)
              DDMIN_LOG.write('\n')
           return self.PASS
        if len(deltas)==len(CODE_CHUNKS):
           print("Remove all code, ignore.")
           return self.FAIL

        
        DDMIN_LOG.write(LOG_BREAK)
        DDMIN_LOG.write(f"DDMIN LINES ITERATION NUMBER : {ITER_NUM}\n")
        ITER_NUM+=1
        DDMIN_LOG.write(LOG_BREAK)
        DDMIN_LOG.write("MODIFIED CODE:\n")
        modified_code_dict = {}
        for chunk_id in CODE_CHUNK_IDS:
            if chunk_id not in deltas:
              modified_code_dict[chunk_id] = CODE_CHUNKS[chunk_id-1]
              json.dump({chunk_id: CODE_CHUNKS[chunk_id-1]},DDMIN_LOG)
              DDMIN_LOG.write('\n')

        DDMIN_LOG.write(LOG_BREAK)
        DDMIN_LOG.write("REMOVED CODE:\n")
        for chunk_id in deltas:
              json.dump({chunk_id: CODE_CHUNKS[chunk_id-1]},DDMIN_LOG)
              DDMIN_LOG.write('\n')
        
        DDMIN_LOG.write(LOG_BREAK)
        DDMIN_LOG.write("MODEL OUTPUT ON MODIFIED CODE (prediction, confidence):\n")
        pred, confidence = EVAL_FN(modified_code_dict,*EVAL_FN_DEFAULT_ARGS)
        DDMIN_LOG.write(f"{pred}, {confidence:.4f}\n")

        if pred == 1:
            # The prediction of the model becomes 1.
            CANDIDATE_TRIGGER_ID = deltas[0]
            CANDIDATE_TRIGGER_CONFIDENCE = confidence 
            return self.FAIL
        else:
            return self.PASS

def get_trigger_ddmin_lines(code_lines, log, eval_fn, eval_fn_default_args):
  print("Start")
  print(code_lines)
  global CODE_CHUNKS, EVAL_FN, EVAL_FN_DEFAULT_ARGS, CODE_CHUNK_IDS, DDMIN_LOG
  CODE_CHUNKS = code_lines
  EVAL_FN = eval_fn
  EVAL_FN_DEFAULT_ARGS = eval_fn_default_args
  DDMIN_LOG = log

  CODE_CHUNK_IDS = list(range(1, len(code_lines)+1))
  deltas = CODE_CHUNK_IDS

  mydd = MyDD()
  
  c = mydd.ddmin(deltas)              # Invoke DDMIN

  return CANDIDATE_TRIGGER_ID, CANDIDATE_TRIGGER_CONFIDENCE


