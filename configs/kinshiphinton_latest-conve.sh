#!/usr/bin/env bash

data_dir="./data/KinshipHintonLatest"
model="conve"
add_reversed_training_edges="True"
group_examples_by_query="True"
emb_2D_d1=6
emb_2D_d2=6
entity_dim=36
relation_dim=36
num_rollouts=1
bucket_interval=10
num_epochs=100
num_wait_epochs=500
batch_size=12
train_batch_size=12
dev_batch_size=64
learning_rate=0.001
grad_norm=0
emb_dropout_rate=0.2
beam_size=128

num_negative_samples=20
margin=0.5
