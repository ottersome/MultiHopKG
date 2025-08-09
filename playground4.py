import torch
from torch import nn

checkpoint = torch.load("./models/gtllm/flagship.pt")

print(f"CHeckpoint type is {type(checkpoint)}")
print("Checkpoint right now has keys ")
for k,v in checkpoint.items():
    try:
        if isinstance(v, torch.Tensor):
            str_val = f"Tensor of shape {v.shape}"
        elif isinstance(v, nn.Module):
            str_val = f"Module of type {type(v)}"
        elif isinstance(v, list):
            str_val = f"List of length {len(v)}. preview: {v[:5]}"
        elif isinstance(v, dict):
            str_val = f"Dict of length {len(v)} with keys {list(v.keys())[:5]}"
        else:
            str_val = str(v)
        print(f"- {k}: {str_val}")
    except Exception as e:
        exit()
