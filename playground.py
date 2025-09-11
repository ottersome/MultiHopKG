import pandas as pd

train_triples = pd.read_csv("./data/FB15K-237/train.triples", sep="\t", header=None)
dev_triples = pd.read_csv("./data/FB15K-237/dev.triples", sep="\t", header=None)
test_triples = pd.read_csv("./data/FB15K-237/test.triples", sep="\t", header=None)

print(f"Train_triples contains {train_triples.shape[0]} triples")
print(f"Dev_triples contains {dev_triples.shape[0]} triples")
print(f"Test_triples contains {test_triples.shape[0]} triples")

print(f"Totaling to {train_triples.shape[0] + dev_triples.shape[0] + test_triples.shape[0]} triples")

raw_kb = pd.read_csv("./data/FB15K-237/raw.kb", sep="\t", header=None)

print(f"Raw KB contains {raw_kb.shape[0]} triples")
