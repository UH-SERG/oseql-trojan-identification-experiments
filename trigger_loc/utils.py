import numpy as np
import sys
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from utils import tensorize_defect_data, tensorize_clone_data

def n_gram_overlap_match(candidate_trigger_code, triggers):

    # Match technique #2: checks the n-gram overlap

    match_found = False

    tokens1 = word_tokenize(candidate_trigger_code)

    for trigger in triggers:
      
      # Tokenize sentences
      tokens2 = word_tokenize(trigger)

      # Function to generate n-grams from a list of tokens
      def generate_ngrams(tokens, n):
         return list(ngrams(tokens, n))

      # Choose the n-gram size (e.g., 1 for unigrams, 2 for bigrams, 3 for trigrams)
      n = 2

      # Generate n-grams for both sentences
      ngrams1 = generate_ngrams(tokens1, n)
      ngrams2 = generate_ngrams(tokens2, n)

      if len(candidate_trigger_code) == 1:
        # Assume we don't have triggers that are a single character
        return False


      # Calculate the intersection of n-grams between the two sentences
      intersection = set(ngrams1) & set(ngrams2)

      # Calculate the Jaccard similarity
      # jaccard_similarity = len(intersection) / len(set(ngrams1) | set(ngrams2))

      inclusion_degree = len(intersection)/min(len(set(ngrams1)),len(set(ngrams2)))
      #print(f"Common {n}-grams: {intersection}")
      #print(f"Inclusion overlap: {inclusion_degree}")
      if inclusion_degree > 0.5:
          match_found = True
          break
    return match_found

def inclusion_match(candidate_trigger_code, triggers):
   # Match technique #1: checks candidate trig is contained in
   # any trigger
   match_found = False
   for trigger in triggers:
     if candidate_trigger_code in trigger:
         match_found = True
         break
   return match_found

import numpy as np
from sklearn.ensemble import IsolationForest

def find_outliers_isolation_forest(data, contamination=0.05, random_state=42):
    # Convert the dictionary values to a numpy array
    values = np.array(list(data.values())).reshape(-1, 1)

    # Create an Isolation Forest model
    model = IsolationForest(contamination=contamination, random_state=random_state)

    # Fit the model to the data and predict outliers
    model.fit(values)
    outliers = model.predict(values)
    #print("Outliers", outliers)
    #print("Data", data)

    # Find the outlier entries
    outlier_entries = {}
    marker = 0
    for key, value in data.items():
        marker += 1 
        if "_" not in key: # task is not clone where ids are of the form NUM_NUM
          if outliers[int(key) - 1] == -1:
            outlier_entries[key]= value
        else:
          if outliers[marker - 1] == -1:
            outlier_entries[key]= value

    if outlier_entries:
        max_key = max(outlier_entries, key=lambda k: outlier_entries[k])
        max_value = outlier_entries[max_key]
        if max_value < 0.5:
            return None
        else:
          return (max_key, max_value)
    else:
        return None
import numpy as np
from sklearn.covariance import EllipticEnvelope

def find_outliers_elliptic_envelope(data, contamination=0.05, support_fraction=1.0):
    # Convert the dictionary values to a numpy array
    values = np.array(list(data.values())).reshape(-1, 1)

    # Create an Elliptic Envelope model
    model = EllipticEnvelope(contamination=contamination, support_fraction=support_fraction)

    are_all_equal = np.all(values == values[0])
    if are_all_equal:
        return None
    #print(are_all_equal)

    # Fit the model to the data and predict outliers
    try:
        model.fit(values)
    except Exception as e:
        return None

    outliers = model.predict(values)
    outlier_entries = {}
    marker = 0
    for key, value in data.items():
        marker += 1 
        if "_" not in key: # task is not clone where ids are of the form NUM_NUM
          if outliers[int(key) - 1] == -1:
            outlier_entries[key] = value
        else:
          if outliers[marker - 1] == -1:
            outlier_entries[key]= value

    if outlier_entries:
        max_key = max(outlier_entries, key=lambda k: outlier_entries[k])
        max_value = outlier_entries[max_key]
        if max_value < 0.5:
            return None
        else:
            return (max_key, max_value)
    else:
        return None

import random

def choose_majority_or_random(results):

    # Count the occurrences of each string in the list
    counts = {string: results.count(string) for string in results}

    # Find the string(s) with the highest count(s)
    max_count = max(counts.values())
    majority_strings = [string for string, count in counts.items() if count == max_count]

    if len(majority_strings) == 1:
        # If there is a majority, return it
        return majority_strings[0]
    else:
        # If all strings are different, choose a random one
        return random.choice(majority_strings)

def find_outliers_iqr(data):
    # Extract the values from the dictionary
    values = list(data.values())
    Q1 = np.percentile(values, 25)
    Q3 = np.percentile(values, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    #print(lower_bound, "lower_bound")
    #print(upper_bound, "upper_bound")

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

def test_modified_code_defect(parts_dict_modified, args, eval_examples, pool, tokenizer, model, evaluate):
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

def test_modified_code_clone(parts_dict_modified, parts_dict_original, args, eval_examples, pool, tokenizer, model, evaluate):
      """
      The clone task takes 2 code snippets. We send it the modified version of
      1 code snippet, and keep the other one the same.
      """
      reconstructed_code1 = " ".join(parts_dict_modified.values())
      reconstructed_code2 = " ".join(parts_dict_original.values())
      eval_examples[0].source = reconstructed_code1
      eval_examples[0].target = reconstructed_code2
      eval_data = tensorize_clone_data(args, pool, tokenizer, eval_examples)
      pred, logits = evaluate(args, model, eval_examples, eval_data, write_to_pred=True)
      if pred:
       pred = 1
      else:
       pred = 0
      prob_score = logits[1]
      return pred, prob_score
