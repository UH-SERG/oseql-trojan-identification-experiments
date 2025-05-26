# Some previous model analysis experiments we had performed. To implement this
# code, they were put inside the anacomp_run
# function in model_anacomp/utils.py

    ###################################################### start DDMIN ################################################################### 
    global org_score
    global anacomp_data

    anacomp_data['total_num_zero_params'] = get_num_zero_params(model)[1]
    anacomp_data['model']=model
    anacomp_data['model_test']=copy.deepcopy(model)
    anacomp_data['eval_callback_fn']=eval_callback_fn
    anacomp_data['args']=args 
    anacomp_data['eval_examples']=eval_examples
    anacomp_data['eval_data']=eval_data

    # print(get_num_zero_attn_params(model))
    # sys.exit(1)

    # Create a state dictionary with all params zero
    anacomp_data['zero_sd']=copy.deepcopy(model.state_dict())
    zero_out_all_attn_layers(anacomp_data['zero_sd'])

    l_info = get_layers_info(model)
    '''
    for layer in l_info.keys():
        print(layer)
    sys.exit(1)
    '''

    logger.info("Evaluating the original model...")
    org_score = eval_callback_fn(args=args, model=model, eval_examples=eval_examples, eval_data=eval_data)['eval_acc']

    '''
    # Test code
    selected_keys = list(l_info.keys())[0:150]
    for key in selected_keys:
        del l_info[key]
    '''

    logger.info("Generate layer indexing maps...")
    index_to_layer, layer_to_index =_get_layer_index_maps(l_info)
    anacomp_data['index_to_layer'] = index_to_layer

    logger.info("Generate params...") 
    params = _get_params(model, l_info, layer_to_index)
    anacomp_data['params'] = params

    chunk_size = 10
    num_chunks = int(len(params)/chunk_size)

    logger.info("Generate chunks...")
    chunks = _get_param_chunks(params, chunk_size)
    anacomp_data['chunks'] = chunks

    logger.info("Generate chunk ids...")
    chunk_ids = list(range(0,len(chunks)))
    anacomp_data['chunk_ids'] = chunk_ids

    ddmin()

    ###################################################### end DDMIN ################################################################### 

    ###################################################### start PRUNING EXPERIMENTS ################################################### 

    # PRUNING METHOD 1

    # Define the pruning hyperparameters
    '''
    pruning_params = {
      'name': 'weight',
      'pruning_method': 'magnitude',
      'sparsity': 0.5,
      'dim': 0
    }
    '''

    pruning_params = {
     'name': 'weight',
     'amount': 0.5,
      'pruning_method': prune.L1Unstructured
    }

    # Prune the network using the hyperparameters

    '''
    print(model.parameters())
    for item in model.parameters():
        print(type(item))
    for (module, name) in model.parameters():
        print(module)
        print(name)
        print('--------------------')
    '''
    parameters_to_prune = [(name, param) for name, param in model.named_parameters()]
    for name, param in model.named_parameters():
        print(type(name))

    #pruning_method = prune.RandomUnstructured
    prune.global_unstructured(parameters=parameters_to_prune, **pruning_params)
    #prune.global_unstructured(parameters=parameters_to_prune, pruning_method=pruning_method, **pruning_params)

    #for name, parameter in model.named_parameters():
    #  if 'weight' in name:
    #      parameter.requires_grad = True
    #      prune.l1_unstructured(parameter, **pruning_params)
    #      parameter.requires_grad = False

    #prune.l1_unstructured(model, **pruning_params)
    # PRUNING METHOD 2

    # Prune params with small absolute values 

    sd = model.state_dict()
    tensors = list(sd.values())
    flattened_tensors = [tensor.flatten() for tensor in tensors]
    abs_tensors = [torch.abs(t) for t in flattened_tensors]
    merged_tensor = torch.cat(abs_tensors)
    print(torch.median(merged_tensor))
    print(torch.max(merged_tensor))
    print(torch.min(merged_tensor))
    total_num_vals_updated=0
    total_num_zeros=0
    for layer in sd.keys():
        t = sd[layer]
        flattened = t.flatten()
        zeros_tensor = torch.sum(t == 0)
        num_zeros = zeros_tensor.item() # Original no. of zeros in tensor 
        total_num_zeros+=num_zeros
        count = (torch.abs(flattened) < 0.001).sum().item()
        flattened[torch.abs(flattened)<0.001]=0
        num_vals_updated = count - num_zeros
        total_num_vals_updated+=num_vals_updated
        #print("Updated {} values in layer {}.".format(num_vals_updated,layer))
        pruned = flattened.reshape(t.shape)
        sd[layer] = pruned
    print("Updated {} values in model.".format(total_num_vals_updated))
    print('num_zeros_total', total_num_zeros)
    model.load_state_dict(sd) 
    print(get_num_zero_params(model)[1], get_num_zero_attn_params(model)[1], get_num_params(model))
    #result = eval_callback_fn(args=args, model=model, eval_examples=eval_examples, eval_data=eval_data)
    #score = result['eval_acc']
    #print(result)
    ###################################################### end PRUNING EXPERIMENTS ############################################################# 

    ###################################################### start WEIGHT-TO-IMAGE EXPERIMENTS ################################################### 
    # Saving weights 
    sd = model.state_dict()
    tensors = list(sd.values())
    #print (sd.keys())
    #sys.exit(1)
    flattened_tensors = [tensor.flatten() for tensor in tensors]
    abs_tensors = [torch.abs(t) for t in flattened_tensors]
    merged_tensor = torch.cat(abs_tensors)
    print(merged_tensor.numel())
    print(merged_tensor.min().item())
    print(merged_tensor.max().item())

    # get to 0-255 range
    pixels = torch.round(torch.div(merged_tensor, 3)).int()

    # add a new dimension to the tensor
    pixels = torch.unsqueeze(pixels, dim=1)

    # repeat each element along the new dimension
    pixels = pixels.repeat(1, 3)
    print(pixels[:5])
    sm_pixels = pixels
    #sys.exit(1)

    #print(pixels)

    from PIL import Image
    #import torch

    # create an image from the pixel tensor
    #image = Image.fromarray(sm_pixels.cpu().numpy().astype('uint8'), mode='RGB')
    sm_pixels_cpu = sm_pixels.cpu()
    pixel_chunks = torch.chunk(sm_pixels_cpu, 10000, dim=0)
    sm_pixels_small = Image.fromarray(pixel_chunks[1].numpy().astype('uint8'))
    #image = Image.fromarray(sm_pixels_np) 

    # save the image as a JPEG file
    sm_pixels_small.save('my_image.jpg')

    # save the tensor to a file
    #torch.save(merged_tensor, 'my_tensor.pt')
    
    sys.exit(1)
    ###################################################### end WEIGHT-TO-IMAGE EXPERIMENTS ################################################### 
