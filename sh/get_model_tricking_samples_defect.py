# This code generates all input samples that can trick a model to make an
# attacker determined prediction. Just add your trigger to these samples, and
# the model will make the attacker desired prediction. (Of course, we are
# assuming the model is poisoned.)

# Task: Vulnerability Detection
# Attacker Desired Prediction: Predict 1 if sample has a trigger.
# You can try this on the predictions .txt output of the models in the
# Salesforce CodeT5 framework

import argparse
import json

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
        poison_id, poison_target = map(int, poison_line.strip().split('\t'))
        clean_id, clean_target = map(int, clean_line.strip().split('\t'))
        poisoned_preds[poison_id] = poison_target
        clean_preds[clean_id] = clean_target

model_tricking_samples = []
clean_samples_pred_0 = []

for id in clean_preds:
    # Find the lines where prediction in clean.txt is 1 and prediction in poison.txt is 0
    #if clean_preds[id] == 1 and poisoned_preds.get(id, 0) == 0:
    if clean_preds[id] == 1 and poisoned_preds[id] == 0:
        model_tricking_samples.append(id)
    # Find the lines where prediction in clean.txt is 0
    if clean_preds[id] == 0:
        clean_samples_pred_0.append(id) 


# Open the input and output files
with open(args.input_poisoned_file, "r") as input_file, open(args.output_poisoned_modeltrickers_file, "w") as output_file:
    for line in input_file:
        # Parse the JSON object from the JSONL line
        data = json.loads(line.strip())
        
        # Check if the "idx" field exists and its value is in the target_numbers list
        if data["idx"] in model_tricking_samples:
            # Write the JSON object to the output file
            output_file.write(json.dumps(data) + "\n")

# Open the input and output files
with open(args.input_clean_file, "r") as input_file, open(args.output_clean_pred0_file, "w") as output_file:
    for line in input_file:
        # Parse the JSON object from the JSONL line
        data = json.loads(line.strip())
        
        # Check if the "idx" field exists and its value is in the target_numbers list
        if data["idx"] in clean_samples_pred_0:
            # Write the JSON object to the output file
            output_file.write(json.dumps(data) + "\n")

