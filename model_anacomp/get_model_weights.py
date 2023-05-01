# This file is a standalone file that can be run directly.
import torch

# Path of model state dictionary
model_sd_path = "PATH_TO pytorch_model.bin FILE" 

# Load, and check all layer names in models
sd = torch.load(model_sd_path)
#print(sd.keys())

# Access params tensor for any key 
# sd['<LAYER_NAME>']

# You save the params in any way as per the needs of your application. Here's one way:
with open('file.pt', 'wb') as f:
    torch.save(sd['encoder.encoder.layer.11.attention.self.key.bias'], f)




