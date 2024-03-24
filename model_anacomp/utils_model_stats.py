import torch

def get_num_zero_attn_params(model):
    total_params = 0
    total_num_zero_params = 0
    sd = model.state_dict()
    info = []
    info.append("layer,total_params,total_zero_params")

    for layer in sd.keys():
      if 'attention' in layer:
        num_params = sd[layer].numel()  # Total number of parameters in the tensor
        num_nonzero_params = torch.nonzero(sd[layer]).shape[0]  # Number of non-zero parameters in the tensor
        num_zero_params = num_params - num_nonzero_params  
        total_params+=num_params
        total_num_zero_params+=num_zero_params
        info.append(layer + ',' + str(num_params) + ',' + str(num_zero_params))

    info.append("-------------------------------------------------------------")
    info.append("Total Num of Attention Params:" + str(total_params))
    info.append("Total Num of Attention Params with zero value:"+str(total_num_zero_params))
    return info, total_num_zero_params

def get_num_zero_params(model):
    total_params = 0
    total_num_zero_params = 0
    sd = model.state_dict()
    info = []
    info.append("layer,total_params,total_zero_params")

    for layer in sd.keys():
        num_params = sd[layer].numel()  # Total number of parameters in the tensor
        num_nonzero_params = torch.nonzero(sd[layer]).shape[0]  # Number of non-zero parameters in the tensor
        num_zero_params = num_params - num_nonzero_params  
        total_params+=num_params
        total_num_zero_params+=num_zero_params
        info.append(layer + ',' + str(num_params) + ',' + str(num_zero_params))

    info.append("-------------------------------------------------------------")
    info.append("Total Num of Params:" + str(total_params))
    info.append("Total Num of Params with zero value:"+str(total_num_zero_params))
    return info, total_num_zero_params

def get_num_params(model):
    """
    Returns the total no. of params in a model (layer-wise pass).
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
