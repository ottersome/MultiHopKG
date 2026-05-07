#!/usr/bin/env bash

# Plain graph-search config for the MQuAKE-ST multi-answer preprocessing flow.
# Use this after scripts/preprocess_mquake_st_ma.py has regenerated data/mquake_st.

data_dir="data/mquake_st"
model="point"
group_examples_by_query="False"
use_action_space_bucketing="True"
allow_direct_answer_edges="True"

# Declared for compatibility with the shared experiment launcher.
question_texts_paths="n/a"
QAData_path="n/a"
max_question_len=30

bandwidth=400
entity_dim=200
relation_dim=200
history_dim=200
history_num_layers=3
num_rollouts=20
num_rollout_steps=2
bucket_interval=10
num_epochs=1000
num_wait_epochs=400
num_peek_epochs=5
batch_size=64
train_batch_size=64
dev_batch_size=64
learning_rate=0.001
baseline="n/a"
grad_norm=5
emb_dropout_rate=0.3
ff_dropout_rate=0.1
action_dropout_rate=0.9
action_dropout_anneal_interval=1000
beta=0.05
relation_only="False"
beam_size=128
checkpoint_keep_last=3

wandb="True"
wandb_project="mquake_st_ma_point"

num_paths_per_entity=-1
margin=-1
