# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""

from __future__ import absolute_import
import os
import logging
import argparse
import math
import copy
import spacy
#from spacy.cli import download
#download("en_core_web_sm")
import numpy as np
from io import open
from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaModel, RobertaTokenizer,
                          BartConfig, BartForConditionalGeneration, BartTokenizer,
                          T5Config, T5ForConditionalGeneration, T5Tokenizer)
import multiprocessing
import time

from models import DefectModel
from configs import add_args, set_seed
from utils import get_filenames, get_elapse_time, load_and_cache_defect_data 
from models import get_model_size
from model_anacomp.utils import anacomp_run, anacomp_compare_models
from model_anacomp.utils import get_num_params
import sys
import copy
import json
import nltk.translate.bleu_score as bleu
from nltk.util import ngrams
import nltk
from nltk.tokenize import word_tokenize
from trigger_loc.tl_ddmin import get_trigger_ddmin_lines
from utils import tensorize_defect_data
import csv
import trigger_loc_config
from trigger_loc_run_defect import trigger_loc_run

nltk.download('punkt')


MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer),
                 't5': (T5Config, T5ForConditionalGeneration, T5Tokenizer),
                 'codet5': (T5Config, T5ForConditionalGeneration, RobertaTokenizer),
                 'bart': (BartConfig, BartForConditionalGeneration, BartTokenizer)}

cpu_cont = multiprocessing.cpu_count()

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

logger = logging.getLogger(__name__)
LOG_BREAK="*"*50 + "\n"

def evaluate(args, model, eval_examples, eval_data, write_to_pred=False):
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Eval!
    if trigger_loc_config.approach == "": 
      logger.info("***** Running evaluation *****")
      logger.info("  Num examples = %d", len(eval_examples))
      logger.info("  Num batches = %d", len(eval_dataloader))
      logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits = []
    labels = []
    #for batch in tqdm(eval_dataloader, total=len(eval_dataloader), desc="Evaluating"):
    for batch in eval_dataloader :
        inputs = batch[0].to(args.device)
        label  = batch[1].to(args.device)
        with torch.no_grad():
            #print("******************call model(inputs,label)***************************************")
            lm_loss, logit = model(inputs, label)
            #print(logit)
            #print("******************return from call to  model(inputs,label)***********************")
            eval_loss += lm_loss.mean().item()
            logits.append(logit.cpu().numpy())
            #print((logit.shape),'the shape of the logit') 
            #print(inputs[0],'the inputs')
            #sys.exit(1)
            #print((logit), 'the logit')
            #print(label,'the label')
            #print((logit.cpu().numpy()), 'the logit.cpu.numpy')
            #print(type(model))
            #sys.exit(1)
            labels.append(label.cpu().numpy())
            '''
            # Test Code:
              What is logit?
              What is logit.cpu().numpy()?
              What is label?
              What is inputs?
              What is batch[1], batch[0], batch?
            '''
        nb_eval_steps += 1
    logits = np.concatenate(logits, 0)
    labels = np.concatenate(labels, 0)
    #print("LOOK HERE")
    #print(logits[:, 1])
    preds = logits[:, 1] > 0.5
    #print(preds[0])
    eval_acc = np.mean(labels == preds)
    #sys.exit(1)
    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.tensor(eval_loss)

    result = {
        "eval_loss": float(perplexity),
        "eval_acc": round(eval_acc, 8),
    }

    if trigger_loc_config.approach != "":
      pred = preds[0]
      prob_score = logits[0]
      return pred, prob_score

    logger.info("***** Eval results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key], 8)))

    if write_to_pred:
        with open(os.path.join(args.output_dir, "predictions.txt"), 'w') as f:
            for example, pred in zip(eval_examples, preds):
                if pred:
                    f.write(str(example.idx) + '\t1\n')
                else:
                    f.write(str(example.idx) + '\t0\n')

    return result

def evaluate_callback(args, model, eval_examples, eval_data):
    """
    This is a call back function, to be called from the anacomp library.
    We have defined it here since the implementation of 'evaluate' is 
    in this module.

    Returns:
        The accuracy of the model.
    """

    logger.info("  " + "***** Eval callback *****")
    logger.info("  Batch size = %d", args.eval_batch_size)

    if args.n_gpu > 1:
        # multi-gpu training
        model = torch.nn.DataParallel(model)

    result = evaluate(args, model, eval_examples, eval_data, write_to_pred=True)

    return result

def find_outliers(data):
    # Extract the values from the dictionary
    values = list(data.values())

    Q1 = np.percentile(values, 25)
    Q3 = np.percentile(values, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

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

def test_modified_code(parts_dict_modified, args, eval_examples, pool, tokenizer, model):
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

def get_preds_seq_char(code, n, args, triggers, eval_examples, pool, tokenizer, model, results_file):
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
          pred, prob_score = test_modified_code(parts_dict_modified, args, eval_examples, pool, tokenizer, model)
          part_stats = "{},{},{:.4f}\n".format(part_id, pred, prob_score)
          results_file.write(part_stats)
          prob_score_dict[part_id] = prob_score
    
      return prob_score_dict, parts_dict

def get_preds_seq_line(code_lines, args, triggers, eval_examples, pool, tokenizer, model, results_file):
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
          #print(f"Removed Line {line_id}: {line}")
          pred, prob_score = test_modified_code(lines_dict_modified, args, eval_examples, pool, tokenizer, model)
          #sys.exit(1)
          line_stats = "{},{},{:.4f}\n".format(line_id, pred, prob_score)
          results_file.write(line_stats)
          prob_score_dict[line_id] = prob_score
    
      return prob_score_dict, lines_dict

def inclusion_match(candidate_trigger_code, triggers):
   # Match technique #1: checks candidate trig is contained in
   # any trigger
   match_found = False
   for trigger in triggers:
     if candidate_trigger_code in trigger:
         match_found = True
         break
   return match_found

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

      # Calculate the intersection of n-grams between the two sentences
      intersection = set(ngrams1) & set(ngrams2)

      # Calculate the Jaccard similarity
      #jaccard_similarity = len(intersection) / len(set(ngrams1) | set(ngrams2))

      inclusion_degree = len(intersection)/min(len(set(ngrams1)),len(set(ngrams2)))
      #print(f"Common {n}-grams: {intersection}")
      #print(f"Inclusion overlap: {inclusion_degree}")
      if inclusion_degree > 0.5:
          match_found = True
          break

    return match_found

def main():
    parser = argparse.ArgumentParser()
    t0 = time.time()
    args = add_args(parser)
    logger.info(args)

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1

    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, cpu count: %d",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), cpu_cont)
    args.device = device
    set_seed(args)

    # Build model
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
    model = model_class.from_pretrained(args.model_name_or_path)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name)

    model = DefectModel(model, config, tokenizer, args)
    logger.info("Finish loading model [%s] from %s", get_model_size(model), args.model_name_or_path)

    if args.load_model_path is not None:
        logger.info("Reload model from {}".format(args.load_model_path))
        model.load_state_dict(torch.load(args.load_model_path))

    model.to(device)

    pool = multiprocessing.Pool(cpu_cont)
    args.train_filename, args.dev_filename, args.test_filename = get_filenames(args.data_dir, args.task, args.sub_task)
    fa = open(os.path.join(args.output_dir, 'summary.log'), 'a+')

    if args.anacomp == 1:
       logger.info("***** Running Anacomp Only *****")

       #####SELECT CUSTOM MODEL#####
       MODEL1_PATH = "/scratch1/aftab/CodeT5-original-gpu0/CodeT5/sh/saved_models/defect/1-to-0_poisoning/DCI_pr2/bart_base/bart_base_all_lr1_bs16_src512_trg3_pat2_e50/checkpoint-best-acc/pytorch_model.bin"
       model.load_state_dict(torch.load(MODEL1_PATH))
       #MODEL2_PATH = ""
       #model2 = copy.deepcopy(model)  
       #model2.load_state_dict(torch.load())
       #############################

       print("NUMBER OF PARAMS", get_num_params(model))

       # Do analysis on a single model
       # eval_examples, eval_data = load_and_cache_defect_data(args, args.test_filename, pool, tokenizer, 'test', False)
       # anacomp_run(model, eval_callback_fn=evaluate_callback, args=args,
       # eval_examples=eval_examples, eval_data=eval_data)

       # Compare models
       #anacomp_compare_models(model,model2)

       sys.exit(1)

    if args.do_train:
        if args.n_gpu > 1:
            # multi-gpu training
            model = torch.nn.DataParallel(model)
        if args.local_rank in [-1, 0] and args.data_num == -1:
            summary_fn = '{}/{}'.format(args.summary_dir, '/'.join(args.output_dir.split('/')[1:]))
            tb_writer = SummaryWriter(summary_fn)

        # Prepare training data loader
        train_examples, train_data = load_and_cache_defect_data(args, args.train_filename, pool, tokenizer, 'train',
                                                                is_sample=False)
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        num_train_optimization_steps = args.num_train_epochs * len(train_dataloader)
        save_steps = max(len(train_dataloader), 1)

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)

        if args.warmup_steps < 1:
            warmup_steps = num_train_optimization_steps * args.warmup_steps
        else:
            warmup_steps = int(args.warmup_steps)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                    num_training_steps=num_train_optimization_steps)

        # Start training
        train_example_num = len(train_data)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", train_example_num)
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Batch num = %d", math.ceil(train_example_num / args.train_batch_size))
        logger.info("  Num epoch = %d", args.num_train_epochs)

        global_step, best_acc = 0, 0
        not_acc_inc_cnt = 0
        is_early_stop = False
        for cur_epoch in range(args.start_epoch, int(args.num_train_epochs)):
            bar = tqdm(train_dataloader, total=len(train_dataloader), desc="Training")
            nb_tr_examples, nb_tr_steps, tr_loss = 0, 0, 0
            model.train()
            for step, batch in enumerate(bar):
                batch = tuple(t.to(device) for t in batch)
                source_ids, labels = batch

                loss, logits = model(source_ids, labels)

                if args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                tr_loss += loss.item()

                nb_tr_examples += source_ids.size(0)
                nb_tr_steps += 1
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                if nb_tr_steps % args.gradient_accumulation_steps == 0:
                    # Update parameters
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    global_step += 1
                    train_loss = round(tr_loss * args.gradient_accumulation_steps / nb_tr_steps, 4)
                    bar.set_description("[{}] Train loss {}".format(cur_epoch, round(train_loss, 3)))

                if (step + 1) % save_steps == 0 and args.do_eval:
                    logger.info("***** CUDA.empty_cache() *****")
                    torch.cuda.empty_cache()

                    eval_examples, eval_data = load_and_cache_defect_data(args, args.dev_filename, pool, tokenizer,
                                                                          'valid', is_sample=False)

                    result = evaluate(args, model, eval_examples, eval_data)
                    eval_acc = result['eval_acc']

                    if args.data_num == -1:
                        tb_writer.add_scalar('dev_acc', round(eval_acc, 4), cur_epoch)

                    # save last checkpoint
                    last_output_dir = os.path.join(args.output_dir, 'checkpoint-last')
                    if not os.path.exists(last_output_dir):
                        os.makedirs(last_output_dir)

                    if True or args.data_num == -1 and args.save_last_checkpoints:
                        model_to_save = model.module if hasattr(model, 'module') else model
                        output_model_file = os.path.join(last_output_dir, "pytorch_model.bin")
                        torch.save(model_to_save.state_dict(), output_model_file)
                        logger.info("Save the last model into %s", output_model_file)

                    if eval_acc > best_acc:
                        not_acc_inc_cnt = 0
                        logger.info("  Best acc: %s", round(eval_acc, 4))
                        logger.info("  " + "*" * 20)
                        fa.write("[%d] Best acc changed into %.4f\n" % (cur_epoch, round(eval_acc, 4)))
                        best_acc = eval_acc
                        # Save best checkpoint for best ppl
                        output_dir = os.path.join(args.output_dir, 'checkpoint-best-acc')
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        if args.data_num == -1 or True:
                            model_to_save = model.module if hasattr(model, 'module') else model
                            output_model_file = os.path.join(output_dir, "pytorch_model.bin")
                            torch.save(model_to_save.state_dict(), output_model_file)
                            logger.info("Save the best ppl model into %s", output_model_file)
                    else:
                        not_acc_inc_cnt += 1
                        logger.info("acc does not increase for %d epochs", not_acc_inc_cnt)
                        if not_acc_inc_cnt > args.patience:
                            logger.info("Early stop as acc do not increase for %d times", not_acc_inc_cnt)
                            fa.write("[%d] Early stop as not_acc_inc_cnt=%d\n" % (cur_epoch, not_acc_inc_cnt))
                            is_early_stop = True
                            break

                model.train()
            if is_early_stop:
                break

            logger.info("***** CUDA.empty_cache() *****")
            torch.cuda.empty_cache()

        if args.local_rank in [-1, 0] and args.data_num == -1:
            tb_writer.close()

    if args.do_test:
        logger.info("  " + "***** Testing *****")
        logger.info("  Batch size = %d", args.eval_batch_size)

        for criteria in ['best-acc']:
            file = os.path.join(args.output_dir, 'checkpoint-{}/pytorch_model.bin'.format(criteria))
            logger.info("Reload model from {}".format(file))
            model.load_state_dict(torch.load(file))

            if args.n_gpu > 1:
                # multi-gpu training
                model = torch.nn.DataParallel(model)

            eval_examples, eval_data = load_and_cache_defect_data(args, args.test_filename, pool, tokenizer, 'test', False)
            logger.info("Loaded all test samples data")
            #print(eval_examples[0].source)
            #sys.exit(1)

            ########### SINGLE-LINE DEAD-CODE TRIGGER LOCALIZATION ###########

            if trigger_loc_config.approach != "":
             ## TESTING
             trigger_loc_run(args, eval_examples, pool, tokenizer, evaluate, model)
             sys.exit(1)

             trig_loc_log_full = open(os.path.join(args.output_dir, "trigger_loc_stats.txt"), 'w', newline='')
             csv_writer = csv.writer(trig_loc_log_full) 
             trigger_capture_count = 0

             header = ["sample_id", "code_len_lines", "code_len_chars", 
                       "full_code_pred", "full_code_pred_prob_score", 
                       "candidate_trigger", "trigger_detected?", "mod_code_pred_prob_score"]

             csv_writer.writerow(header)
             
             #for example_no in range(0,len(eval_examples)): 
             for example_no in tqdm(range(0, len(eval_examples)), desc="Finding trigger loc in example"):

              test_sample = [eval_examples[example_no]]

              test_sample_tensorized = tensorize_defect_data(args, pool, tokenizer, test_sample)

              #trig_loc_log_sample = open(os.path.join(args.output_dir, "parts_removed_preds.txt"), 'w')
              samp_id = eval_examples[example_no].idx
              trig_loc_log_sample = open(os.path.join(args.output_dir, f"parts_removed_preds_{samp_id}.txt"), 'w')

              #pred, logits = evaluate(args, model, eval_examples, eval_data, write_to_pred=True)
              pred, logits = evaluate(args, model, test_sample, test_sample_tensorized, write_to_pred=True)

              if pred:
               pred = 1
              else:
               pred = 0

              prob_score = logits[1]
              trig_loc_log_sample.write(f"TRIGGER DETECTION METHOD: \n{trigger_loc_config.locate_trigger_approach}\n")
              trig_loc_log_sample.write(f"Chunk size : \n{trigger_loc_config.chunk_size}\n")
              trig_loc_log_sample.write(LOG_BREAK)
              trig_loc_log_sample.write("FULL CODE PREDICTION\n")
              trig_loc_log_sample.write(LOG_BREAK)
              trig_loc_log_sample.write("prediction,prob_score\n")
              part_stats = "{},{:.4f}\n".format(pred, prob_score)
              trig_loc_log_sample.write(part_stats)
              trig_loc_log_sample.write(LOG_BREAK)

              triggers = [
		'int capacity = 5333;',
		'assert(15>=0);',
		'assert(-15<=0);',
		'int *panel_id;',
		'int zoom_ratio;'
              ]

              code = eval_examples[example_no].source 
              code_lines = eval_examples[example_no].source_lines
              candidate_trigger = ()
              ####### Phase 1 : Generate Prediction Scores and Locate trigger ###########
              if trigger_loc_config.locate_trigger_approach == "sequential_char_chunks":
                #code = eval_examples[0].source 
                code_dict = {}
                prob_score_dict = {}
                #prob_score_dict, code_dict = get_preds_seq_char(code, chunk_size, args, triggers, eval_examples, pool, tokenizer, model, trig_loc_log_sample) 
                prob_score_dict, code_dict = get_preds_seq_char(code, trigger_loc_config.chunk_size, args, triggers, test_sample, pool, tokenizer, model, trig_loc_log_sample) 
                candidate_trigger = find_outliers(prob_score_dict)
              if trigger_loc_config.locate_trigger_approach == "sequential_line_chunks":
                #code_lines = eval_examples[0].source_lines
                #prob_score_dict, code_dict = get_preds_seq_line(code_lines, args, triggers, eval_examples, pool, tokenizer, model, trig_loc_log_sample) 
                prob_score_dict, code_dict = get_preds_seq_line(code_lines, args, triggers, test_sample, pool, tokenizer, model, trig_loc_log_sample) 
                candidate_trigger = find_outliers(prob_score_dict)
              if trigger_loc_config.locate_trigger_approach == "ddmin_lines":
                # GOAL:
                # Let F be the full, triggered code, and let M_p be the
                # poisoned model, and that M_p(F) = 0.  Our goal is to find the
                # smallest F_part, where F_part is a subset of F, such that
                # M_p(F - F_part) = 1.  In other words, our goal is to find the
                # smallest piece of code in F, removing which from F will
                # change the prediction of M_p on F from 0 to 1.
                #code_lines = eval_examples[0].source_lines
                code_lines = eval_examples[example_no].source_lines
                # Create a dictionary where the key is the line number and the value is the line content
                code_dict  = {line_id: line for line_id, line in enumerate(code_lines, start=1)}
                #eval_fn_default_args = [args, eval_examples, pool, tokenizer, model]
                eval_fn_default_args  = [args, test_sample, pool, tokenizer, model]
                candidate_trigger     = get_trigger_ddmin_lines(code_lines, trig_loc_log_sample, eval_fn=test_modified_code, eval_fn_default_args = eval_fn_default_args )
                #print(candidate_trigger)

              trig_loc_log_sample.write(LOG_BREAK)

              ####### Phase 2 : Trigger Verification ####################################
              # We now test whether the above candidate_trigger is really a trigger. 
              candidate_trigger_code            = ""
              trig_detection_result             = ""
              post_cand_trig_removal_prob_score = 0.0

              if candidate_trigger != None:
                 candidate_trigger_id   = candidate_trigger[0] 
                 candidate_trigger_code = code_dict[candidate_trigger_id]
                 post_cand_trig_removal_prob_score = candidate_trigger[1]

                 # Iterate through the list and check for matches
                 match_found = False 
                 
                 if trigger_loc_config.locate_trigger_approach == "sequential_line_chunks" or "ddmin_lines":
                   match_found = inclusion_match(candidate_trigger_code.strip(), triggers)

                 if trigger_loc_config.locate_trigger_approach == "sequential_char_chunks":
                   match_found = n_gram_overlap_match(candidate_trigger_code, triggers)

                 if match_found: 
                   trig_detection_result = "captured_trigger"
                   trigger_capture_count+=1
                 else:
                   trig_detection_result = "captured_non_trigger"

                 #print(candidate_trigger_id, candidate_trigger_code, trig_detection_result, post_cand_trig_removal_prob_score)
                 trig_loc_log_sample.write(f"Candidate Trigger Part Id: {candidate_trigger_id}\n")
                 trig_loc_log_sample.write(f"Candidate Trigger Code   : {candidate_trigger_code}\n")
                 trig_loc_log_sample.write(f"Trigger Detection Result : {trig_detection_result}\n")
                 trig_loc_log_sample.write(f"Prediction score after removing candidate trigger : {post_cand_trig_removal_prob_score:.4f}\n")
              else:
                 #print("No triggers.")
                 trig_detection_result = "nothing_captured"
                 trig_loc_log_sample.write("There are no triggers.\n")


              # Sample Result Log
              trig_loc_log_sample.write(LOG_BREAK)
              trig_loc_log_sample.close()

              # Combined Result Log
              '''
              Columns -> ["sample_id", "code_len_lines", "code_len_chars", 
                       "full_code_pred", "full_code_pred_prob_score", 
                       "candidate_trigger", "trigger_detected?", "mod_code_pred_prob_score"]
              '''
              tl_results = [samp_id, len(code_lines), len(code), 
                      pred, '{:.4f}'.format(prob_score), candidate_trigger_code, 
                      trig_detection_result,'{:.4f}'.format(post_cand_trig_removal_prob_score)] 
              csv_writer.writerow(tl_results)

             print("Trigger Capture Count:", trigger_capture_count)
             trig_loc_log_full.close()
             sys.exit(1)

            ########### END OF SINGLE-LINE DEAD-CODE TRIGGER LOCALIZATION ###########

            result = evaluate(args, model, eval_examples, eval_data, write_to_pred=True)
            logger.info("  test_acc=%.4f", result['eval_acc'])
            logger.info("  " + "*" * 20)

            fa.write("[%s] test-acc: %.4f\n" % (criteria, result['eval_acc']))
            if args.res_fn:
                with open(args.res_fn, 'a+') as f:
                    f.write('[Time: {}] {}\n'.format(get_elapse_time(t0), file))
                    f.write("[%s] acc: %.4f\n\n" % (
                        criteria, result['eval_acc']))
    fa.close()


if __name__ == "__main__":
    main()
