#!/usr/bin/env bash

data_dir="data/eduin_kinshiphinton/"
model="conve"
add_reversed_training_edges="True"
group_examples_by_query="True"
entity_dim=100
relation_dim=100
num_rollouts=1
bucket_interval=10
num_epochs=1000
num_wait_epochs=1000
batch_size=32
train_batch_size=32
dev_batch_size=64
learning_rate=0.003
grad_norm=0
emb_dropout_rate=0.2
beam_size=128

action_dropout_rate=0.95
action_dropout_anneal_interval=1000
action_dropout_rate=0.1
adam_beta1=0.9
adam_beta2=0.999
add_reverse_relations="True"
bandwith=300

num_negative_samples=20
margin=0.5
