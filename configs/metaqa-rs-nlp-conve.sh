#!/usr/bin/env bash

# BERT-conditioned Reward Shaping on MQuake-ST

data_dir="data/MetaQA"
model="point.rs.conve"
group_examples_by_query="False"
use_action_space_bucketing="True"

# Language conditioning
use_question_encoder="True"
bert_model_name="bert-base-uncased"
cached_qa_metadata_path="./.cache/metaqa/metaqa_clean.json"
raw_QAData_path="data/MetaQA/metaqa_qa_nhop.csv" # This ought to change a lot depending on what it is that you are testing.
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
num_epochs=100
num_wait_epochs=100
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
beam_size=128
mu=0.25
checkpoint_keep_last=3

# Reference placeholders for other FNs (unused here)
distmult_state_dict_path="/home/ottersome/projects/llmbi/salesforce/model/MetaQA-conve-RV-xavier-100-100-0.005-64-2-0.1-0.1-0.2-0.05/checkpoint-57.tar"
complex_state_dict_path="/home/ottersome/projects/llmbi/salesforce/model/MetaQA-conve-RV-xavier-100-100-0.005-64-2-0.1-0.1-0.2-0.05/checkpoint-57.tar"
conve_state_dict_path="/home/ottersome/projects/llmbi/salesforce/model/MetaQA-conve-RV-xavier-100-100-0.005-64-2-0.1-0.1-0.2-0.05/checkpoint-57.tar"

num_paths_per_entity=-1
margin=-1

wandb_project="rs-nlp-conve-metaqa-final_run"
seed=12
