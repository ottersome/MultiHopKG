#!/usr/bin/env bash

data_dir="data/mquake_salesforce_compatible"
model="transe"
add_reversed_training_edges="True"
group_examples_by_query="True"
entity_dim=500
relation_dim=500
num_rollouts=1
bucket_interval=10
num_epochs=1000
num_wait_epochs=500
batch_size=64
train_batch_size=128
dev_batch_size=128
learning_rate=0.003
grad_norm=5
emb_dropout_rate=0.3
beam_size=128
num_negative_samples=50
margin=10
