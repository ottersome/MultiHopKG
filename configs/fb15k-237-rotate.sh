#!/usr/bin/env bash

data_dir="data/FB15K-237"
model="operational_rotate"

add_reversed_training_edges="True"
#
group_examples_by_query="False"
entity_dim=200
relation_dim=200
num_rollouts=1
bucket_interval=10
num_epochs=1000
num_wait_epochs=100
batch_size=512  
train_batch_size=512
dev_batch_size=64
learning_rate=0.0003
grad_norm=0
emb_dropout_rate=0.3
beam_size=128

num_negative_samples=100
margin=12
