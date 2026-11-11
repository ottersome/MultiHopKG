#!/usr/bin/env bash

# BERT-conditioned reward shaping on MQuAKE-ST, initialized from TransE.
data_dir="data/mquake_st"
model="point.rs.transe"
group_examples_by_query="False"
use_action_space_bucketing="True"

use_question_encoder="True"
allow_direct_answer_edges="True"
bert_model_name="bert-base-uncased"
cached_qa_metadata_path="./.cache/mquake_st/mquake_clean.json"
raw_QAData_path="data/mquake_st/mquake_st-qa_nhop.csv"
recompute_qadata_cache="False"
max_question_len=200

bandwidth=200
relation_dim=100
entity_dim=100
history_dim=200
emb_2D_d1=10
emb_2D_d2=10
kernel_size=2
history_num_layers=3
num_rollouts=100
num_rollout_steps=4
num_out_channels=64
bucket_interval=10
num_epochs=500
num_wait_epochs=8
num_peek_epochs=2
checkpoint_keep_last=3
batch_size=128
train_batch_size=128
dev_batch_size=64
learning_rate=0.001
baseline="n/a"
grad_norm=0
emb_dropout_rate=0.1
ff_dropout_rate=0.2
action_dropout_rate=0.4
action_dropout_anneal_interval=1000
reward_shaping_threshold=0.1
beta=0.00
relation_only="False"
beam_size=100
mu=0.25
num_negative_samples=12

# Overridden by Make with the staged TransE substrate checkpoint.
transe_state_dict_path=""
num_paths_per_entity=-1
wandb_project="rs-nlp-transe-mquake-final_run"
seed=12
