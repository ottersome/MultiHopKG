import os
import  torch

embeddings_path = "./models/mquake/"

ckpnt = torch.load(os.path.join(embeddings_path, "checkpoint"))

# Now simply add gamma
ckpnt["gamma"] = 12

print(f"chekcpoint keys are {list(ckpnt.keys())}")

torch.save(ckpnt, os.path.join(embeddings_path, "checkpoint"))
