#!/usr/bin/env bash

data_dir="./data/KinshipHintonLatest"
model="conve"
add_reversed_training_edges="True"
group_examples_by_query="True"
emb_2D_d1=3
emb_2D_d2=4
entity_dim=12
relation_dim=12
num_rollouts=1
bucket_interval=10
num_epochs=100
num_wait_epochs=500
batch_size=16
train_batch_size=8
dev_batch_size=64
learning_rate=0.010
num_out_channels=8
grad_norm=0
emb_dropout_rate=0.1
feat_dropout_rate=0.2
hidden_dropout_rate=0.1
kernel_size=2
label_smoothing_epsilon=0.05
beam_size=32

wandb_project="kinship_hinton_linkpred_emb"
num_negative_samples=20
margin=0.5
