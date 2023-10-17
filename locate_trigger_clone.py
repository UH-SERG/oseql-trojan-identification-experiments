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
from models import CloneModel
import logging
import argparse
import numpy as np
from io import open
from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaModel, RobertaTokenizer,
                          BartConfig, BartForConditionalGeneration, BartTokenizer,
                          T5Config, T5ForConditionalGeneration, T5Tokenizer)
import multiprocessing

from configs import set_seed
from utils import get_filenames, get_elapse_time, load_and_cache_clone_data 
from models import get_model_size
import sys
import nltk
from utils import tensorize_clone_data
import trigger_loc.config as tlconf
from trigger_loc.config import add_args
from trigger_loc.code_tasks.clone import trigger_loc_run

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


def evaluate(args, model, eval_examples, eval_data, write_to_pred=False):
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    model.eval()
    logits = []
    labels = []
    y_trues = []

    for batch in eval_dataloader :
        inputs = batch[0].to(args.device)
        #label  = batch[1].to(args.device)
        labels = batch[1].to(args.device)
        with torch.no_grad():
            lm_loss, logit = model(inputs, labels)
            logits.append(logit.cpu().numpy())
            #labels.append(label.cpu().numpy())
            y_trues.append(labels.cpu().numpy())
    logits = np.concatenate(logits, 0)
    y_trues = np.concatenate(y_trues, 0)
    #labels = np.concatenate(labels, 0)
    #preds = logits[:, 1] > 0.5
    y_preds = logits[:, 1] > 0.5 

    #pred = preds[0]
    pred = y_preds[0]
    prob_score = logits[0]
    return pred, prob_score

def main():
    parser = argparse.ArgumentParser()
    args = add_args(parser)
    logger.info(args)


    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
        args.n_gpu = 1
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

    model.resize_token_embeddings(len(tokenizer)) 

    model = CloneModel(model, config, tokenizer, args)
    logger.info("Finish loading model [%s] from %s", get_model_size(model), args.model_name_or_path)

    model.to(device)
    pool = multiprocessing.Pool(cpu_cont)

    file = args.load_model_path
    logger.info("Reload model from {}".format(file))
    model.load_state_dict(torch.load(file))

    eval_examples, eval_data = load_and_cache_clone_data(args, args.test_filename, pool, tokenizer, 'test', False)
    logger.info("Loaded all test samples data")

    ########### SINGLE-LINE DEAD-CODE TRIGGER LOCALIZATION ###########
    trigger_loc_run(args, eval_examples, pool, tokenizer, evaluate, model)
    #######END OF SINGLE-LINE DEAD-CODE TRIGGER LOCALIZATION #########

if __name__ == "__main__":
    main()
