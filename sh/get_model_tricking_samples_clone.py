# This code generates all input samples that can trick a model to make an
# attacker determined prediction. Just add your trigger to these samples, and
# the model will make the attacker desired prediction. (Of course, we are
# assuming the model is poisoned.)

# Task: Clone Detection
# Attacker Desired Prediction: Predict 1 if sample has a trigger.
# You can try this on the predictions .txt output of the models in the
# Salesforce CodeT5 framework

import argparse
import sys

# Create an argument parser
parser = argparse.ArgumentParser(description="Find inputs whose clean version \
        outputs 1 with the model, and whose triggered version outputs 0 with \
        the model.")

# Add arguments for specifying poisoned and clean prediction file
# (Both are outcome of the same poisoned 
parser.add_argument("-preds-pf", "--preds-of-poisoned-file", required=True, help="File containing predictions of triggered inputs.")
parser.add_argument("-preds-cf", "--preds-of-clean-file",    required=True, help="File containing predictions of clean inputs.")

# Add arguments for specifying input and output files
parser.add_argument("-pf",  "--input-poisoned-file",  required=True, help="File containing triggered inputs.")
parser.add_argument("-opmf", "--output-poisoned-modeltrickers-file", required=True, help="File to contain model tricking inputs.")

# Add arguments for specifying input and output files
parser.add_argument("-cf",  "--input-clean-file",  required=True, help="File containing clean inputs.")
parser.add_argument("-cfp0", "--output-clean-pred0-file", required=True, help="File to contain clean inputs with pred 0.")

# Parse the command-line arguments
args = parser.parse_args()

# Read the contents of the poisoned and clean files into two dictionaries
poisoned_preds = {}
clean_preds = {}

with open(args.preds_of_poisoned_file, 'r') as poison_file, open(args.preds_of_clean_file, 'r') as clean_file:
    for poison_line, clean_line in zip(poison_file, clean_file):
        poison_id1, poison_id2, poison_target = map(int, poison_line.strip().split())
        clean_id1, clean_id2, clean_target = map(int, clean_line.strip().split())
        poison_sample_id = str(min(int(poison_id1), int(poison_id2))) + "_" + str(max(int(poison_id1), int(poison_id2)))
        clean_sample_id = str(min(int(clean_id1), int(clean_id2))) + "_" + str(max(int(clean_id1), int(clean_id2)))
        assert(poison_sample_id == clean_sample_id)
        poisoned_preds[poison_sample_id] = poison_target
        clean_preds[clean_sample_id] = clean_target

model_tricking_samples = []
clean_samples_pred_0 = []

#print ("LOOK", len(clean_preds))
#sys.exit(1)

for id in clean_preds:
    # Find the lines where prediction in clean.txt is 1 and prediction in poison.txt is 0
    #if clean_preds[id] == 1 and poisoned_preds.get(id, 0) == 0:
    if clean_preds[id] == 1 and poisoned_preds[id] == 0:
        model_tricking_samples.append(id)
    # Find the lines where prediction in clean.txt is 0
    if clean_preds[id] == 0:
        clean_samples_pred_0.append(id) 


# Open the input and output files
count=0
with open(args.input_poisoned_file, "r") as input_file, open(args.output_poisoned_modeltrickers_file, "w") as output_file:
    for line in input_file:
        line=line.strip()
        url1, url2, poison_target, type1, type2  = line.strip().split() 
        sample_id = str(min(int(url1), int(url2))) + "_" + str(max(int(url1), int(url2)))
        
        if sample_id in model_tricking_samples:
            output_file.write(line + "\n")
            count+=1

assert(len(model_tricking_samples)==count)

# Open the input and output files
count=0
with open(args.input_clean_file, "r") as input_file, open(args.output_clean_pred0_file, "w") as output_file:
    for line in input_file:
        line=line.strip()
        url1, url2, clean_target, type1, type2  = line.strip().split()
        sample_id = str(min(int(url1), int(url2))) + "_" + str(max(int(url1), int(url2)))
        
        if sample_id in clean_samples_pred_0: 
            output_file.write(line + "\n")
            count+=1

assert(len(clean_samples_pred_0)==count)
