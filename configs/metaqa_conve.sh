#!/usr/bin/env bash

data_dir="data/MetaQA"
model="conve"
add_reversed_training_edges="True"
group_examples_by_query="True"
entity_dim=100
relation_dim=100
emb_2D_d1=10
emb_2D_d2=10
num_rollouts=1
bucket_interval=10
num_epochs=300
num_wait_epochs=500
batch_size=64
train_batch_size=64
dev_batch_size=16
learning_rate=0.005
grad_norm=0
emb_dropout_rate=0.2
hidden_dropout_rate=0.2
beam_size=128
kernel_size=2
num_out_channels=64
emb_dropout_rate=0.1
feat_dropout_rate=0.2
hidden_dropout_rate=0.1
label_smoothing_epsilon=0.05
checkpoint_keep_last=3

num_negative_samples=20
margin=0.5
wandb_project="metaqa_st_linkpred_emb"
