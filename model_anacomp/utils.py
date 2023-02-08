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
import DD
import copy


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

##################################################

anacomp_data={}
org_sd={}

##################################################

def get_children(module, opfile, space_size):
    # DEPRECATED
    # This function is no longer in use and should not be called.

    global MODULE_ID
    space_size+=1
    children = [child for child in module.children()]
    if len(children)==0:
        space_size-=1
        return
    MODULE_ID+=1
    LEVEL = str(space_size-1)
    opfile.write("------------------------------------------\n")
    opfile.write("    "*space_size+"LEVEL:"+LEVEL+" MODULE_ID:"+str(MODULE_ID)+" CHILD:\n")
    opfile.write("    "*space_size+"Child type: "+ str(type(module)) + "\n")

    for name, params in tqdm(module.named_parameters()):
      opfile.write(str(name)+"\n")
      with open("module."+str(MODULE_ID)+"."+str(name), "w") as mod_params: 
        for param in tqdm(params):
               if param.dim()!=0:
                for val in param:
                   mod_params.write(str(float(val))+"\n")
                   #vals.append(float(val))
               else:
                   mod_params.write(str(float(val))+"\n")
                   #vals.append(float(param.item()))
        mod_params.close()
    for child in children:
      get_children(child, opfile, space_size)

def _get_chunks(layer, row_id, i, j, max_row_id, num_chunks):
    # DEPRECATED
    # See description of _get_chunk_map()
    """
    _get_chunk_map() helper 
    """
    chunks = []
  
    if num_chunks == 1:
      chunk = _get_chunk_data(layer, row_id, i, j)
      assert (chunk['row'] <= max_row_id)
      assert (chunk['start'] != chunk['end'])
      chunks.append(chunk)
      return chunks
  
    arr = np.array_split(range(i,j), num_chunks)
    for item in arr:
      chunk = _get_chunk_data(layer, row_id, i=item[0], j=item[item.size-1])
      assert (chunk['end'] <= j)
      assert (chunk['row'] <= max_row_id)
      assert (chunk['start'] != chunk['end']) # We don't want singleton chunks
      chunks.append(chunk)
    return chunks

def _get_num_chunks(row_size, chunk_size):
    # DEPRECATED
    # See description of _get_chunk_map()
    """
    _get_chunk_map() helper 
    """
    if row_size > chunk_size:
        num_chunks = int(row_size/chunk_size)
    else:
        num_chunks = 1
    return num_chunks

def _get_chunk_map(l_info, chunk_size):
    # DEPRECATED
    # This method assumes the larger dimension to be the column. However, to
    # make changes to the model, we need to know **exactly** where the param is
    # located. When we are oblivious to whether a dim is the col or row, i.e.,
    # the 2nd or the 1st dim, we lose location information. Hence this approach
    # should not be used.
    """
    Generates the chunk map.
    """

    one_d_layer_chunks = []
    two_d_layer_chunks = []

    logger.info("(Generating Chunk Map) Processing layers...")
    for layer in tqdm(l_info.keys()):

        assert (l_info[layer]['num_dims'] <= 2)

        if l_info[layer]['num_dims']==1:

            # Use col_id to represent larger axis
            if l_info[layer]['len_dim1'] >= l_info[layer]['len_dim2']:
                num_cols = l_info[layer]['len_dim1'] # (also the size of a row)
                num_rows = 1 
            else:
                num_cols = l_info[layer]['len_dim2'] # (also the size of a row)
                num_rows = 1 

            # Process row to get chunks 
            i = 0
            j = num_cols - 1
            row_id = 0

            # Calculate num_chunks
            num_chunks = _get_num_chunks(num_cols, chunk_size)

            one_d_layer_chunks = _get_chunks(layer, row_id, i, j, max_row_id=0, num_chunks=num_chunks)

        elif l_info[layer]['num_dims']==2:

            # Use col_id to represent larger axis
            if l_info[layer]['len_dim1'] >= l_info[layer]['len_dim2']:
                num_cols = l_info[layer]['len_dim1']
                num_rows = l_info[layer]['len_dim2']
            else:
                num_cols = l_info[layer]['len_dim2']
                num_rows = l_info[layer]['len_dim1']
                
            # Process rows to get chunks 
            i = 0
            j = num_cols - 1

            # Calculate num_chunks
            num_chunks = _get_num_chunks(num_cols, chunk_size)

            for row_id in range(0, num_rows):
              two_d_layer_chunks += _get_chunks(layer, row_id, i, j, max_row_id=num_rows-1, num_chunks = num_chunks)

    all_chunks = one_d_layer_chunks + two_d_layer_chunks
    chunk_map = {index: value for index, value in enumerate(all_chunks)}
    return chunk_map

def _get_detailed_arch(model):
    """
    Outputs arch_components.txt, a file showing the architecture of the model, 
    with global layer ids assigned to each component.
    """

    with open("arch_components.txt", "w") as file3:
        global_layer = 0
        file3.write("Main Model: " + str(type(model).__name__)+"\n")
        for child in model.children():
            file3.write("-------------------------------------------------------------------\n")
            child_name = str(type(child).__name__)
            file3.write("Component hierarchy of "+child_name+"\n")
            file3.write("-------------------------------------------------------------------\n")
            file3.write(str(child)+"\n")
            file3.write("-------------------------------------------------------------------\n")
            file3.write("Component list of "+child_name+"\n")
            file3.write("-------------------------------------------------------------------\n")
            prev_layer_num=-1
            for component, params in child.named_parameters():
                if "layer" in component:
                    component_name_parts = component.split(".")
                    if "layers" in component:
                       layer_num_idx = component_name_parts.index("layers")+1
                    else:
                       layer_num_idx = component_name_parts.index("layer")+1

                    layer_num = str(component_name_parts[layer_num_idx])
                    if layer_num!=prev_layer_num:
                        global_layer+=1
                        prev_layer_num=layer_num
                    file3.write("gl."+str(global_layer)+"."+child_name+"."+str(component)+"\n")
                else:
                    file3.write("gl_not_real_layer."+str(global_layer)+"."+child_name+"."+str(component)+"\n")

def _get_weights(model):
    """
    Generates .csv files consisting of weights for each layer in the model.
    
    Output filename Description: 
      `(gl.X)` indicates the global layer number. The rest is the 
      description of the component. Example File names:
      - gl.9.RobertaModel.encoder.layer.8.attention.self.value.weight.csv 
      - gl.9.RobertaModel.encoder.layer.8.attention.self.key.bias.csv
    """

    global_layer = 0
    for child in model.children():
        child_name = str(type(child).__name__)
        prev_layer_num=-1
        for component, params in tqdm(child.named_parameters()):
            if "layer" in component:
                component_name_parts = component.split(".")
                if "layers" in component:
                   layer_num_idx = component_name_parts.index("layers")+1
                else:
                   layer_num_idx = component_name_parts.index("layer")+1

                layer_num = str(component_name_parts[layer_num_idx])
                if layer_num!=prev_layer_num:
                    global_layer+=1
                    prev_layer_num=layer_num
                opfile = open("gl."+str(global_layer)+"."+child_name+"."+str(component)+".csv","w")
                
            else:
                opfile = open("gl_not_real_layer."+str(global_layer)+"."+child_name+"."+str(component)+".csv","w")
            for tnsr in tqdm(params):
              if tnsr.dim()!=0:
                for val in tnsr:
                    opfile.write(str(float(val))+"\n")
              else:
                opfile.write(str(float(tnsr.item()))+"\n")
            opfile.close()


def _get_num_params(model):
    """
    Returns the total no. of params in a model.
    """
    sd = model.state_dict()
    num_params = 0
    for pair in sd.items():
          layer_params = pair[1]
          dim = len(layer_params.shape)
          assert(dim<=2)
          if dim == 1:
            size = layer_params.shape[0]
            num_params+=size
          if dim == 2:
            size1 = layer_params.shape[0]
            size2 = layer_params.shape[1]
            num_params+=size1*size2
    return num_params

def _get_layers_info(model):
    """
    Returns a dictionary with info on each layer of the model.
    """

    sd = model.state_dict()

    layers_info = {}

    num_params = 0
    for pair in sd.items():
          layer_data = { 'num_params' : 0,
                         'num_dims' : 0,
                         'len_dim1' : 0,
                         'len_dim2' : 0
                       }
          layer_name = pair[0]
          layer_params = pair[1]
          dim = len(layer_params.shape)
          assert(dim<=2)
          if dim == 1:
            size = layer_params.shape[0]
            layer_data['num_dims'] = 1
            layer_data['len_dim1'] = size
            layer_data['num_params'] = size
          if dim == 2:
            size1 = layer_params.shape[0]
            size2 = layer_params.shape[1]
            layer_data['num_dims'] = 2
            layer_data['len_dim1'] = size1
            layer_data['len_dim2'] = size2
            layer_data['num_params'] = size1*size2

          layers_info[layer_name] = layer_data  
    return layers_info

def _get_layer_index_maps(l_info):
    layers = []
    for layer in l_info.keys():
        layers.append(layer)

    index_to_layer = {index : layer for index,layer in enumerate(layers)}
    layer_to_index = {layer : index for index,layer in enumerate(layers)}
    return index_to_layer, layer_to_index

def _get_params(model, l_info, layer_to_index):

    logger.info("Generating list of all params...")

    params = []

    sd = model.state_dict()
    for layer in tqdm(l_info.keys(), desc="Scanning layers..."):

      if l_info[layer]['num_dims'] == 1:

        num_rows = l_info[layer]['len_dim1']
  
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
          # Loop over the columns of the tensor
          for col_idx in tqdm(range(num_cols), leave=False,desc="Scanning params of a 2D tensor..."):
              param_data = {}
              param_data['layer_id'] =layer_to_index[layer]
              param_data['row_idx'] = row_idx
              param_data['col_idx'] = col_idx
              params.append(param_data)

    return params

def _zero_out_param(sd, layer_name, row_idx, col_idx=None):
    """
    Sets the value of a param to 0. Changes the state dictionary of a model.
    """
    if col_idx == None:
      sd[layer_name][row_idx] = 10
      '''
      # Test code

      if layer_name == 'encoder.encoder.layer.11.output.dense.bias' and row_idx == 746:
          print('In zero out!',layer_name, sd[layer_name][row_idx], row_idx)
      '''
    else:
      #print(sd[layer_name][row_idx,col_idx])
      sd[layer_name][row_idx,col_idx] = 10

def _zero_out_tensor_params(sd, layer_name, row_no, start, end):
    # NOTE: not part of minimization algorithm. Just use it
    # independently for testing purposes.
    """
    Sets a subset of a tensor to zero, starting from element at index 'start'
    to that at index 'end'.

    Args:
        sd (dictionary): The state dictionary of the model. 
        layer_name (string): The name of the layer.
        row_no (int): The row you want to change.
        start (int): The index of the first element you want to change. 
        end (int): The index of the last element you want to change. 

    Returns:
        The state dictionary with modified weights.
    """

    tensor = sd[layer_name] 
    sub_tensor = None

    '''
    # Test code

    print("Look here", two_d_matrix[0,0])
    start = 3
    end = 7
    row_no = 1
    two_d_matrix = torch.tensor([[1,15,3,48,1,5,2,1,2,10],[1,2,3,4,5,6,7,8,9,10]])
    print(two_d_matrix)
    sys.exit(1)
    '''

    assert(len(tensor.shape) <= 2)

    if len(tensor.shape) == 2:
      sub_tensor = tensor[row_no,:]
    elif len(tensor.shape) == 1:
      sub_tensor = tensor

    assert(sub_tensor.shape[0]>= start)
    assert(sub_tensor.shape[0]>= end)

    num_zero_vals = end - start + 1 
    zeros = torch.zeros(num_zero_vals)

    print('tensor shape',tensor.shape)
    print('num 0 vals',num_zero_vals)
    print('zeros shape',zeros.shape)
    print('subtensor shape',sub_tensor.shape)
    print('subtensor part shape', sub_tensor[start:end+1].shape)
    sub_tensor[start:end+1] = zeros

    '''
    # Test code

    print("Look here", sub_tensor[0])
    print("Look here", two_d_matrix[0,0])
    '''

    return sd
    
def _zero_out_all_bias_params(model):
    """
    Sets all the bias parameters of a model to zero.
    """

    sd = model.state_dict()
    for pair in sd.items():
        layer_name = pair[0]
        layer_params = pair[1]
        if "bias" in layer_name:
          dim = len(layer_params.shape)
          assert(dim<=2)
          if dim == 1:
            size = layer_params.shape[0]
            sd[layer_name] = torch.zeros(size) 
          if dim == 2:
            size1 = layer_params.shape[0]
            size2 = layer_params.shape[1]
            sd[layer_name] = torch.zeros(size1,size2) 
    model.load_state_dict(sd) 
    logger.info("You have set all biases of the model to 0!")
    return model

class MyDD(DD.DD):
    def __init__(self):
        DD.DD.__init__(self)

    def _test(self, deltas):
        # FIXME: Set up a test function that takes a set of deltas and
        # returns either self.PASS, self.FAIL, or self.UNRESOLVED.
        model           = anacomp_data['model']
        sd              = model.state_dict()
        sd_test         = copy.deepcopy(sd)
        callback_test   = anacomp_data['ddmin_test_fn']
        args            = anacomp_data['args']
        eval_examples   = anacomp_data['eval_examples']
        eval_data       = anacomp_data['eval_data']
        chunk_ids       = anacomp_data['chunk_ids']
        chunks          = anacomp_data['chunks']
        params          = anacomp_data['params']
        index_to_layer  = anacomp_data['index_to_layer']

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
                _zero_out_param(sd_test, layer_name, row_idx, col_idx)
                '''
                if col_idx == None:
                  print(sd_test[layer_name][row_idx][col_idx])
                  print(sd[layer_name][row_idx][col_idx])
                  print("---------------------------------")
                '''
          
        # Compare the values of each key
        '''
        for key in sd_test.keys():
            if not torch.equal(sd_test[key], sd[key]):
                 print("The state_dicts are different.")
            else:
                 print("The state_dicts are the same.")
        '''
        model_test = copy.deepcopy(model)
        model_test.load_state_dict(sd_test) 
        #assert(model_test != model)
        callback_test(args=args, model=model_test, eval_examples=eval_examples, eval_data=eval_data)

        if 1>0:
            return self.FAIL
        else:
            return self.PASS
        return self.UNRESOLVED

def ddmin():
    deltas = anacomp_data['chunk_ids']
    # FIXME: Insert your deltas here

    mydd = MyDD()

    # print("Simplifying failure-inducing input...")
    # c = mydd.ddmin(deltas)  # Invoke DDMIN
    # print("The 1-minimal failure-inducing input is", c)
    # print("Removing any element will make the failure go away.")
    # print()

    print("Isolating the failure-inducing difference...")
    (c, c1, c2) = mydd.dd(deltas)  # Invoke DD
    print("The 1-minimal failure-inducing difference is", c)
    #print(c1, "passes,", c2, "fails")

def anacomp_run(model, ddmin_test_fn=None, args=None, eval_examples=None, eval_data=None):
    """
    This is the only API we need to call from outside the anacomp module. We
    can implement the logic of the analysis we want to do in this function,
    taking the help of the other functions in this file.
    """
    global org_sd
    anacomp_data['model']=model
    org_sd = model.state_dict()
    anacomp_data['ddmin_test_fn']=ddmin_test_fn
    anacomp_data['args']=args 
    anacomp_data['eval_examples']=eval_examples
    anacomp_data['eval_data']=eval_data

    l_info = _get_layers_info(model)

    '''
    # Test code
    selected_keys = list(l_info.keys())[0:195]
    for key in selected_keys:
        del l_info[key]
    '''

    logger.info("Generate layer indexing maps...")
    index_to_layer, layer_to_index =_get_layer_index_maps(l_info)
    anacomp_data['index_to_layer'] = index_to_layer

    logger.info("Generate params...") 
    params = _get_params(model, l_info, layer_to_index)
    anacomp_data['params'] = params

    chunk_size = 200
    num_chunks = int(len(params)/chunk_size)

    logger.info("Generate chunks...")
    chunks = _get_param_chunks(params, chunk_size)
    anacomp_data['chunks'] = chunks

    logger.info("Generate chunk ids...")
    chunk_ids = list(range(0,len(chunks)))
    anacomp_data['chunk_ids'] = chunk_ids

    ddmin()
