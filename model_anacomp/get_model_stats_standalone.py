# This file is a standalone file that can be run directly.
import torch

# Load Model 
model_sd_path = "/home/aftab/workspace/Experiment-for-Trojan-Identification/sh/saved_models/clone/DCIv2_poison2ndCode_prate5_Nov2023/codet5p-770m-py_all_lr2_bs8_src400_trg400_pat2_e50/checkpoint-best-acc/pytorch_model.bin"
sd = torch.load(model_sd_path)


# Model Parameter Value Extraction

## Method 1
num_params=0
for key in sd.keys():
    num_params+=sd[key].numel()
print(num_params)

## Method 2
num_params=0
num_params = sum(p.numel() for p in sd.values()) 
print(num_params)


# Model Parameter Value Extraction

# Access params tensor for any key 
# sd['<LAYER_NAME>']

# You save the params in any way as per the needs of your application. Here's one way:
# with open('file.pt', 'wb') as f:
#    torch.save(sd['encoder.encoder.layer.11.attention.self.key.bias'], f)




