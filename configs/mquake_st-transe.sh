#!/usr/bin/env bash

# TransE substrate for the complete observed MQuAKE-ST KG. This is a
# transductive QA substrate, not a held-out link-prediction benchmark.
data_dir="data/mquake_st"
lp_train_triples_path="data/mquake_st/lp_train.triples"
model="transe"
group_examples_by_query="True"
entity_dim=100
relation_dim=100
# These are accepted by the common launcher but unused by TransE.
emb_2D_d1=10
emb_2D_d2=10
num_out_channels=64
kernel_size=2
num_rollouts=1
bucket_interval=10
num_epochs=1000
num_wait_epochs=30
batch_size=64
train_batch_size=64
dev_batch_size=16
learning_rate=0.005
grad_norm=0
emb_dropout_rate=0.1
feat_dropout_rate=0.2
hidden_dropout_rate=0.1
label_smoothing_epsilon=0.05
beam_size=100
num_negative_samples=20
checkpoint_keep_last=3
wandb_project="mquake_st_linkpred_transe"
