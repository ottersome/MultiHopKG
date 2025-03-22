import torch

# Import torch checkpoint 
checkpoint = torch.load("./models/protatE_FBWikiV4/checkpoint")
print("---------Checkpoint Keys-----------")
print(checkpoint.keys())
print("---------Model State Dict Keys-----------")
print(checkpoint["model_state_dict"].keys())
