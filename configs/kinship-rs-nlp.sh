#!/usr/bin/env bash

# BERT-conditioned Reward Shaping on MQuake

data_dir="data/eduin_kinshiphinton"
model="point.rs.conve"
group_examples_by_query="False"
use_action_space_bucketing="True"

# Language conditioning
use_question_encoder="True"
bert_model_name="bert-base-uncased"
cached_qa_metadata_path="./.cache/mquake/kinship_clean.json"
raw_QAData_path="data/eduin_kinshiphinton/kinship_hinton_qa_2hop.csv" # This ought to change a lot depending on what it is that you are testing.
recompute_qadata_cache="False"
max_question_len=200


bandwidth=400
relation_dim=100
entity_dim=100
history_dim=200
history_num_layers=3
num_rollouts=100
num_rollout_steps=2
bucket_interval=10
num_epochs=300
num_wait_epochs=100
num_peek_epochs=2
batch_size=128
train_batch_size=32
dev_batch_size=32
learning_rate=0.003
baseline="n/a"
grad_norm=0
emb_dropout_rate=0.3
ff_dropout_rate=0.1
action_dropout_rate=0.5
action_dropout_anneal_interval=1000
reward_shaping_threshold=0
beta=0.02
relation_only="False"
beam_size=128

# Reference placeholders for other FNs (unused here)
distmult_state_dict_path="model/FB15K-237-distmult-xavier-200-200-0.003-0.3-0.1/model_best.tar"
complex_state_dict_path="model/FB15K-237-complex-RV-xavier-200-200-0.003-0.3-0.1/model_best.tar"
conve_state_dict_path="model/mquake_standdata_full-graph/model_best.tar"

num_paths_per_entity=-1
margin=-1
