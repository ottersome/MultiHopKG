#!/usr/bin/env bash

# ConvE substrate for MQuAKE-ST multi-answer QA.
# Assumes scripts/preprocess_mquake_st_ma.py has already regenerated
# data/mquake_st/train.triples as the all-splits KG triple union.

data_dir="data/mquake_st"
train_triples_path="data/mquake_st/lp_train.triples"
model="conve"
group_examples_by_query="True"
entity_dim=100
relation_dim=100
emb_2D_d1=10
emb_2D_d2=10
num_rollouts=1
bucket_interval=10
# A generous cap plus validation-based convergence selection.  The LP graph is
# intentionally transductive; see README for the interpretation of dev MRR.
num_epochs=1000
num_wait_epochs=50
batch_size=64
train_batch_size=64
dev_batch_size=16
learning_rate=0.005
grad_norm=0
beam_size=100
kernel_size=2
num_out_channels=64
emb_dropout_rate=0.1
feat_dropout_rate=0.2
hidden_dropout_rate=0.1
label_smoothing_epsilon=0.05
checkpoint_keep_last=3

num_negative_samples=20
margin=0.5
wandb_project="mquake_st_ma_conve_substrate"
