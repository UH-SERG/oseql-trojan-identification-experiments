
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

def get_detailed_arch(model):
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

def get_weights(model):
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

def get_layers_info(model):
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
