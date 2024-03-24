# File: utils.py 
# Created: February 4, 2023 
# Description:
#  This file includes all functions for analyzing and modifying models in the
#  Salesforce Code Model Framework (https://github.com/salesforce/CodeT5) 

import logging
import sys
from tqdm import tqdm
import torch
import numpy as np
import random 
import model_anacomp.DD as DD
import copy
import os
#import concurrent.futures
#import multiprocessing
#from functools import partial
import shutil
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from .utils_model_update import copy_chunk_param, zero_out_all_attn_layers, zero_out_param
from .utils_model_info import get_layers_info
from .utils_model_stats import get_num_zero_params, get_num_zero_attn_params, get_num_params


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

##################################################

anacomp_data={}
org_score = 0
ddmin_model_id=0

##################################################

def _get_layer_index_maps(l_info):
    layers = []
    for layer in l_info.keys():
        layers.append(layer)

    index_to_layer = {index : layer for index,layer in enumerate(layers)}
    layer_to_index = {layer : index for index,layer in enumerate(layers)}
    return index_to_layer, layer_to_index

def _process_row(row_idx, layer_to_index, layer):
    '''
    Helper of _get_params for parellelization
    '''
    param_data = {}
    param_data['layer_id'] = layer_to_index[layer]
    param_data['row_idx'] = row_idx
    param_data['col_idx'] = None
    return param_data

def _process_col(col_idx, row_idx, layer_to_index, layer):
    '''
    Helper of _get_params for parallelization
    '''
    param_data = {}
    param_data['layer_id'] = layer_to_index[layer]
    param_data['row_idx'] = row_idx
    param_data['col_idx'] = col_idx
    return param_data

def _get_params(model, l_info, layer_to_index):

    logger.info("Generating list of all params...")

    params = []

    sd = model.state_dict()
    for layer in tqdm(l_info.keys(), desc="Scanning layers..."):

      if l_info[layer]['num_dims'] == 1:

        num_rows = l_info[layer]['len_dim1']
        
        # Parallelization Attempts

        '''
        with concurrent.futures.ThreadPoolExecutor() as executor:
           future_to_row_idx = {executor.submit(process_row, row_idx, layer_to_index, layer): row_idx for row_idx in range(num_rows)}
           for future in tqdm(concurrent.futures.as_completed(future_to_row_idx), total=num_rows, desc="Scanning params of a 1D tensor..."):
                        params.append(future.result())
        '''

        '''
        process_row_partial = partial(_process_row, row_idx=row_idx, layer_to_index=layer_to_index, layer=layer)
        
        with multiprocessing.Pool(processes=4) as pool:
            results = list(tqdm(pool.imap(process_row_partial, range(num_rows)), total=num_rows, desc="Scanning params of a 1D tensor...", leave=False))
        '''

        # The plain loop implementation
                      
        # Loop over the rows of the tensor
        for row_idx in tqdm(range(num_rows), leave=False, desc="Scanning params of a 1D tensor..."):
          param_data = {}
          param_data['layer_id'] = layer_to_index[layer]
          param_data['row_idx'] = row_idx
          param_data['col_idx'] = None
          params.append(param_data)
      
      elif l_info[layer]['num_dims'] == 2:

        # Obtain the number of rows and columns in the tensor
        num_rows, num_cols = sd[layer].shape
  
        # Loop over the rows of the tensor
        for row_idx in tqdm(range(num_rows), leave=False, desc="Scanning rows of a 2D tensor..."):


          # Parallelization Attempts

          '''
          # This loop doesn't really go that fast

          with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_col_idx = {executor.submit(process_col, col_idx, row_idx, layer_to_index, layer): col_idx for col_idx in range(num_cols)}
            for future in tqdm(concurrent.futures.as_completed(future_to_col_idx), total=num_cols, desc="Scanning params of a 2D tensor..."):
              params.append(future.result())
          '''

          '''
          # This is even slower than the plain loop implementation!
          process_col_partial = partial(_process_col, row_idx=row_idx, layer_to_index=layer_to_index, layer=layer)
        
          with multiprocessing.Pool(processes=4) as pool:
            results = list(tqdm(pool.imap(process_col_partial, range(num_cols)), total=num_cols, desc="Scanning params of a 2D tensor...", leave=False))
          '''

          # The plain loop implementation

          # Loop over the columns of the tensor
          for col_idx in tqdm(range(num_cols), leave=False,desc="Scanning params of a 2D tensor..."):
              param_data = {}
              param_data['layer_id'] =layer_to_index[layer]
              param_data['row_idx'] = row_idx
              param_data['col_idx'] = col_idx
              params.append(param_data)

    return params

def _get_param_chunks(params, chunk_size):
    i=0
    chunks = []
    while i < len(params)-1: 
      chunk = {}
      chunk['start'] = i
      chunk['end'] = min(i+chunk_size-1,len(params)-1)
      chunks.append(chunk)
      i+=chunk_size
    return chunks

class MyDD(DD.DD):
    def __init__(self):
        DD.DD.__init__(self)

    def _test(self, deltas):
        # FIXME: Set up a test function that takes a set of deltas and
        # returns either self.PASS, self.FAIL, or self.UNRESOLVED.
        global anacomp_data
        sd              = anacomp_data['model'].state_dict()
        model_test      = anacomp_data['model_test']
        eval_callback_fn   = anacomp_data['eval_callback_fn']
        args            = anacomp_data['args']
        eval_examples   = anacomp_data['eval_examples']
        eval_data       = anacomp_data['eval_data']
        chunk_ids       = anacomp_data['chunk_ids']
        chunks          = anacomp_data['chunks']
        params          = anacomp_data['params']
        index_to_layer  = anacomp_data['index_to_layer']
        """

        '''
        APPROACH 1:
           Zero out all the non-delta chunks in the copy of the
           original map
        '''

        sd_test         = copy.deepcopy(sd)

        # Get chunks to zero_out

        chunk_ids_to_delete = []
        
        for chunk_id in chunk_ids:
            if chunk_id not in deltas:
                chunk_ids_to_delete.append(chunk_id)

        for chunk_id in tqdm(chunk_ids_to_delete, leave=False, 
                             desc="Scanning chunks to remove..."):

            for param_id in tqdm(range(chunks[chunk_id]['start'],
                                 chunks[chunk_id]['end']+1), 
                                 leave=False, 
                                 desc="Removing parameters in the chunk..."):

                layer_name = index_to_layer[params[param_id]['layer_id']]
                row_idx = params[param_id]['row_idx']
                col_idx = params[param_id]['col_idx']
                '''
                if col_idx == None:
                  print("---------------------------------")
                  print(layer_name,row_idx,col_idx)
                  print(sd_test[layer_name][row_idx][col_idx])
                  print(sd[layer_name][row_idx][col_idx])
                '''
                sd_test = zero_out_param(sd_test, layer_name, row_idx, col_idx)
                '''
                if col_idx == None:
                  print(sd_test[layer_name][row_idx][col_idx])
                  print(sd[layer_name][row_idx][col_idx])
                  print("---------------------------------")
                '''
          
        """

        '''
        APPROACH 2:
           Copy vals of delta chunks from original map to a copy of the zero map
        '''

        sd_test = copy.deepcopy(anacomp_data['zero_sd'])

        for chunk_id in tqdm(deltas, leave=False, 
                             desc="Scanning chunks to keep..."):

            for param_id in tqdm(range(chunks[chunk_id]['start'],
                                 chunks[chunk_id]['end']+1), 
                                 leave=False, 
                                 desc="Saving params in chunk..."):

                layer_name = index_to_layer[params[param_id]['layer_id']]
                row_idx = params[param_id]['row_idx']
                col_idx = params[param_id]['col_idx']
                '''
                if col_idx == None:
                  print("---------------------------------")
                  print(layer_name,row_idx,col_idx)
                  print(sd_test[layer_name][row_idx][col_idx])
                  print(sd[layer_name][row_idx][col_idx])
                '''
                sd_test = copy_chunk_param(sd_test, sd, layer_name, row_idx, col_idx)
                '''
                if col_idx == None:
                  print(sd_test[layer_name][row_idx][col_idx])
                  print(sd[layer_name][row_idx][col_idx])
                  print("---------------------------------")
                '''

        cache_path = '/scratch1/CodeT5-original-gpu0/CodeT5-copy/sh/saved_models/defect/roberta_all_lr2_bs16_src512_trg3_pat2_e50/cache_data'
        if os.path.exists(cache_path):
          shutil.rmtree(cache_path)
        model_test.load_state_dict(sd_test) 
        result = eval_callback_fn(args=args, model=model_test, eval_examples=eval_examples, eval_data=eval_data)
        score = result['eval_acc']
        # Compare the values of each key
        # for key in sd_test.keys():
        #    print(torch.eq(sd_test[key],sd[key]))

        global ddmin_model_id
        ddmin_model_id+=1
        output_dir = anacomp_data['args'].output_dir
        info, total_num_zero_params = get_num_zero_params(model_test)

        max_score_change = 5*org_score/100 

        fa = open(os.path.join(output_dir, 'ddmin_result.log'), 'a+')

        if fa.tell() == 0:
          # Write header line
          fa.write("ddmin_model_id,test_acc,test_loss,satisfied?(Y/N),num_zero_params,minimized?(Y/N)\n")

        minimized = ''
        if anacomp_data['total_num_zero_params'] == total_num_zero_params:
            minimized = 'N'
        elif anacomp_data['total_num_zero_params'] < total_num_zero_params:
            minimized = 'Y'

        if (score >= org_score - max_score_change) : #and 
            # score <= org_score + max_score_change) : (Upper bound)
            logger.info("DDMin generated model "+str(ddmin_model_id)+" satisfied criteria.")
            if anacomp_data['total_num_zero_params'] == total_num_zero_params:
                logger.info('However, there was no minimization.')

            output_model_file = os.path.join(output_dir, "pytorch_model.bin.ddmin."+str(ddmin_model_id))
            torch.save(model_test.state_dict(), output_model_file)
            logger.info("Save model #"+str(ddmin_model_id)+" from ddmin")

            output_model_info_file = os.path.join(output_dir, "pytorch_model_info.ddmin.sat."+str(ddmin_model_id))
            with open(output_model_info_file, "w") as model_info_file: 
              for item in info:
                model_info_file.write(item+"\n")
              model_info_file.close()

            fa.write("%d,%.8f,%.8f,%s,%d,%s\n" % (ddmin_model_id, result['eval_acc'], result['eval_loss'], "Y", total_num_zero_params, minimized))
              
            return self.FAIL

        else:
            logger.info("DDMin generated model did not satisfy.")

            output_model_info_file = os.path.join(output_dir, "pytorch_model_info.ddmin.unsat."+str(ddmin_model_id))
            with open(output_model_info_file, "w") as model_info_file: 
              for item in info:
                model_info_file.write(item+"\n")
              model_info_file.close()

            fa.write("%d,%.8f,%.8f,%s,%d,%s\n" % (ddmin_model_id, result['eval_acc'], result['eval_loss'], "N", total_num_zero_params, minimized))

            return self.PASS

        return self.UNRESOLVED

def ddmin():
    global anacomp_data
    deltas = anacomp_data['chunk_ids']
    # FIXME: Insert your deltas here

    mydd = MyDD()

    print("Isolating the failure-inducing difference...")
    (c, c1, c2) = mydd.dd(deltas)  # Invoke DD
    print("The 1-minimal failure-inducing difference is", c)

def anacomp_compare_models(*models):
    sd1 = models[0].state_dict()
    sd2 = models[1].state_dict()

    for layer in sd1:
        if 'attention' not in layer:
          assert (torch.equal(sd1[layer],sd2[layer])==True)
        else:
            if torch.equal(sd1[layer],sd2[layer])==False:
             print(layer)

             # Find the unequal values between the tensors
             unequal = torch.nonzero(sd1[layer] != sd2[layer])
             print("Unequal values in x: ", sd1[layer][unequal].tolist())

def get_cuda_memory_usage_summary():
  print(torch.cuda.memory_summary(device=None, abbreviated=False))


def anacomp_run(model, eval_callback_fn=None, args=None, eval_examples=None, eval_data=None):
    """
    This is the only API we need to call from outside the anacomp module. We
    can implement the logic of the analysis we want to do in this function,
    taking the help of the other functions in this file.
    """


