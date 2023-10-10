from trigger_loc.utils import find_outliers_iqr, find_outliers_isolation_forest, find_outliers_elliptic_envelope, choose_majority_or_random, inclusion_match, n_gram_overlap_match
 
from trigger_loc.config import approach, triggers 
from tqdm import tqdm
import argparse
import os
import sys
import re
import copy

def adjust_for_braces(results, results_adj):
  results_adj["tp"]      = results["tp"]      -   results["braceP"]
  results_adj["fn"]      = results["fn"]      +   results["braceP"]
  results_adj["fp"]      = results["fp"]      -   results["braceN"]
  results_adj["tn"]      = results["tn"]      +   results["braceN"]
  tp = results_adj["tp"] 
  tn = results_adj["tn"]
  fp = results_adj["fp"]
  fn = results_adj["fn"]
  p = results_adj["p"]
  n = results_adj["n"]
  assert(tp + fn == p)
  assert(tn + fp == n)
  return results_adj

def compute_stats(results, p, n):
    '''
    Accuracy = (T P + T N )/(P + N )
    Precision = T P/(T P + F P )
    Recall = T P/(T P + F N )
    F 1 = (2 ∗ T P )/(2 ∗ T P + F P + F N 
    '''
    tp = results["tp"] 
    tn = results["tn"]
    fp = results["fp"]
    fn = results["fn"]
    assert(tp + fn == p)
    assert(tn + fp == n)
    acc  = (tp + tn)/(p + n) 
    prec = tp/(tp + fp)
    rec  = tp/(tp + fn)
    f1   = (2 * tp)/(2 * tp + fp + fn)
    cip  = round(float(results["ci"])*100/results["p"],2)

    results["acc"]  = round(acc,2)
    results["prec"] = round(prec,2)
    results["rec"]  = round(rec,2)
    results["f1"]   = round(f1,2)
    results["cip"]  = "{}/{} ({})".format(results["ci"],results["p"],cip)



def compute_counts(result, results, code_dict, sample_type): 
    code = None
    if result:
       candidate_trigger_id, cand_trig_post_removal_prob = result
       #print(candidate_trigger_id)
       code = code_dict[int(candidate_trigger_id)].strip()
       #print(code)
       match = verify(code)
       if match and sample_type == "P":
           results["ci"]+=1
           results["tp"]+=1
       elif not match and sample_type == "P":
           results["tp"]+=1
           if code == "{" or code == "}":
             results["braceP"]+=1
       elif not match and sample_type == "N":
           results["fp"]+=1
           if code == "{" or code == "}":
             results["braceN"]+=1
       #print(match)
    else:
       if sample_type == "P":
           results["fn"]+=1
       else:
           results["tn"]+=1
       #print("No outliers found.")
       result = (0,0)

    return code, result

def verify(code):

   # Iterate through the list and check for matches
   match_found = False 
   
   if approach == "sequential_line_chunks" or "ddmin_lines":
     match_found = inclusion_match(code.strip(), triggers)
  
   if approach == "sequential_char_chunks":
     match_found = n_gram_overlap_match(code.strip(), triggers)

   return match_found

# Function to extract code from a file
def extract_code_from_file(filename):
    code_dict = {}
    preds_dict = {}
    inside_code_section = False
    inside_preds_section = False
    extracting_probs = False

    with open(filename, 'r') as file:
        lines = file.readlines()

    for line in lines:
        if "FULL CODE" in line:
            inside_code_section = True
            continue

        if "PARTIAL CODE PREDICTIONS" in line:
            inside_code_section = False
            inside_preds_section = True
        
        if inside_preds_section and "," in line:
            if "_" not in line: # line is not a header row
                if "," in line:
                  extracting_probs = True
                  row_items = line.strip().split(",")
                  preds_dict[int(row_items[0])] =  float(row_items[2])

        if extracting_probs == True:
            if "*" in line:
                extracting_probs = False
                inside_preds_section = False

        if inside_code_section and "{" in line and "}" in line:
            parts = line.strip().split(": ")
            key = parts[0].strip("{").strip('"')
            value = parts[1].strip("}\n").strip('"')
            if "_" in key:
                key = key[:-8]
            code_dict[int(key)] = value

    return code_dict, preds_dict

# Create an ArgumentParser object
parser = argparse.ArgumentParser(description='Extract FULL CODE section from text files in a folder.')

# Add the folder argument
parser.add_argument('--folder1', help='The path to the folder containing text files')
parser.add_argument('--folder2', help='The path to the folder containing text files')
parser.add_argument('--model', help='name of model')

# Parse the command-line arguments
args = parser.parse_args()

folders = [args.folder1, args.folder2]

N_num = 0
P_num = 0

results_blank = {"p": 0, "n": 0, "tp": 0, "fn": 0, "fp": 0, "tn": 0, "prec": 0.0,
        "acc": 0.0, "rec": 0.0, "f1": 0.0, "braceP": 0, "braceN": 0, "ci": 0, "cip": 0.0}

iqr_results          = copy.deepcopy(results_blank) 
iforest_results      = copy.deepcopy(results_blank) 
ee_results           = copy.deepcopy(results_blank)
ensemble_results     = copy.deepcopy(results_blank)
iqr_results_adj      = copy.deepcopy(results_blank)
iforest_results_adj  = copy.deepcopy(results_blank)
ee_results_adj       = copy.deepcopy(results_blank)
ensemble_results_adj = copy.deepcopy(results_blank)

for folder in folders:
  # Process files in the folder
  if os.path.exists(folder) and os.path.isdir(folder):
    if 'clean-examples' in folder:
        sample_type = "N" 
    elif 'model-tricking-examples' in folder:
        sample_type = "P"
    for filename in tqdm(os.listdir(folder)):
        if filename.endswith('.txt'):
            if sample_type == "N":
                N_num+=1
            else:
                P_num+=1
            match = re.search(r'\d+', filename)
            sample_id = match.group()
            file_path = os.path.join(folder, filename)
            code_dict, preds_dict = extract_code_from_file(file_path)
            print(f"File: {filename}")
            print("(Code)")
            for key, value in code_dict.items():
                print(f"{key}: {value}")
            print("\n")
            print("(Preds)")
            for key, value in preds_dict.items():
                print(f"{key}: {value}")

            #print("IQR Method")
            #print("-----------------------------------------------------")
            result_iqr = find_outliers_iqr(preds_dict)
            iqr_code, result_iqr = compute_counts(result_iqr, iqr_results, code_dict, sample_type) 
            #print("\n")

            #print("Isolation Forest")
            #print("-----------------------------------------------------")
            result_iforest = find_outliers_isolation_forest(preds_dict)
            iforest_code, result_iforest = compute_counts(result_iforest, iforest_results, code_dict, sample_type) 
            #print("\n")

            #print("Elliptic Envelope")
            #print("-----------------------------------------------------")
            result_ee = find_outliers_elliptic_envelope(preds_dict)
            ee_code, result_ee = compute_counts(result_ee, ee_results, code_dict, sample_type) 
            #print("\n")

            '''
            if ee_code == iforest_code == iqr_code:
                print ("ALL AGREE!")
            else:
                print ("NEED VOTE!")
            print("Majority Voting")
            print("-----------------------------------------------------")
            '''
            result_ensemble = choose_majority_or_random([result_ee, result_iqr, result_iforest])
            #print(result_ensemble)            
            if result_ensemble == (0,0):
                 result_ensemble = None
            ensemble_code, result_ensemble = compute_counts(result_ensemble, ensemble_results, code_dict, sample_type) 

  else:
    print(f"The folder '{folder}' does not exist or is not a valid folder.")
  
iqr_results["p"]     = iforest_results["p"]     = ee_results["p"]     = ensemble_results["p"]     = P_num
iqr_results_adj["p"] = iforest_results_adj["p"] = ee_results_adj["p"] = ensemble_results_adj["p"] = P_num
iqr_results["n"]     = iforest_results["n"]     = ee_results["n"]     = ensemble_results["n"]     = N_num
iqr_results_adj["n"] = iforest_results_adj["n"] = ee_results_adj["n"] = ensemble_results_adj["n"] = N_num

compute_stats(iqr_results, p=P_num, n=N_num) 
compute_stats(iforest_results, p=P_num, n=N_num) 
compute_stats(ee_results, p=P_num, n=N_num) 
compute_stats(ensemble_results, p=P_num, n=N_num) 

iqr_results_adj       = adjust_for_braces(iqr_results, iqr_results_adj)
iforest_results_adj   = adjust_for_braces(iforest_results, iforest_results_adj)
ee_results_adj        = adjust_for_braces(ee_results, ee_results_adj)
ensemble_results_adj  = adjust_for_braces(ensemble_results, ensemble_results_adj)

compute_stats(iqr_results_adj, p=P_num, n=N_num) 
compute_stats(iforest_results_adj, p=P_num, n=N_num) 
compute_stats(ee_results_adj, p=P_num, n=N_num) 
compute_stats(ensemble_results_adj, p=P_num, n=N_num) 

'''
print(iqr_results)
print(args.model + ",iqr," + ','.join(map(str,iqr_results.values())))
print(args.model + ",iqr," + ','.join(map(str,iqr_results_adj.values())))

print(iforest_results)
print(args.model + ",iforest," + ','.join(map(str,iforest_results.values())))
print(args.model + ",iforest," + ','.join(map(str,iforest_results_adj.values())))

print(ee_results)
print(args.model + ",ee," + ','.join(map(str,ee_results.values())))
print(args.model + ",ee," + ','.join(map(str,ee_results_adj.values())))

print(ensemble_results)
print(args.model + ",ensemble," + ','.join(map(str,ensemble_results.values())))
print(args.model + ",ensemble," + ','.join(map(str,ensemble_results_adj.values())))

print("FINAL RESULTS")
'''

print("Model" + ",outlier_method," + ','.join(map(str,ee_results.keys())))
print(args.model + ",iqr," + ','.join(map(str,iqr_results.values())))
print(args.model + ",iqr_adj," + ','.join(map(str,iqr_results_adj.values())))
print(args.model + ",iforest," + ','.join(map(str,iforest_results.values())))
print(args.model + ",iforest_adj," + ','.join(map(str,iforest_results_adj.values())))
print(args.model + ",ee," + ','.join(map(str,ee_results.values())))
print(args.model + ",ee_adj," + ','.join(map(str,ee_results_adj.values())))
print(args.model + ",ensemble," + ','.join(map(str,ensemble_results.values())))
print(args.model + ",ensemble_adj," + ','.join(map(str,ensemble_results_adj.values())))
