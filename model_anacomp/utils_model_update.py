# File: utils.py 
# Created: March 13, 2023 
# Description:
#  This file consists of functions that change the parameters of a model. 

def copy_chunk_param(sd_test, sd_org, layer_name, row_idx, col_idx=None):
    """
    Sets the value of a param in sd to that in org_sd. 
    Changes the state dictionary of a model.
    """
    if col_idx == None:
      sd_test[layer_name][row_idx] = sd_org[layer_name][row_idx]

    else:
      sd_test[layer_name][row_idx,col_idx] = sd_org[layer_name][row_idx,col_idx]
    return sd_test

def zero_out_param(sd_test, layer_name, row_idx, col_idx=None):
    """
    Sets the value of a param to 0. Changes the state dictionary of a model.
    """
    if col_idx == None:
      sd_test[layer_name][row_idx] = 0
      '''
      # Test code

      if layer_name == 'encoder.encoder.layer.11.output.dense.bias' and row_idx == 746:
          print('In zero out!',layer_name, sd[layer_name][row_idx], row_idx)
      '''
    else:
      #print(sd[layer_name][row_idx,col_idx])
      sd[layer_name][row_idx,col_idx] = 0
      #print(sd[layer_name][row_idx,col_idx])
    return sd

def zero_out_all_attn_layers(sd):
    for layer in sd:
      if 'attention' in layer:
        zeros = torch.zeros_like(sd[layer])
        sd[layer]=zeros

def zero_out_all_layers(sd):
    for layer in sd:
      zeros = torch.zeros_like(sd[layer])
      sd[layer]=zeros

def zero_out_tensor_params(sd, layer_name, row_no, start, end):
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
    
def zero_out_all_bias_params(model):
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
