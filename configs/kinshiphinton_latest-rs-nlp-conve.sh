#!/usr/bin/env bash

data_dir="data/KinshipHintonLatest/"
model="point.rs.conve"
add_reversed_training_edges="True"
group_examples_by_query="True"
emb_2D_d1=6
emb_2D_d2=6
entity_dim=36
relation_dim=36
num_rollouts=1
bucket_interval=10
num_epochs=1000
num_wait_epochs=1000
batch_size=16
train_batch_size=16
dev_batch_size=64
learning_rate=0.003
grad_norm=0
emb_dropout_rate=0.2
beam_size=128

history_dim=64
history_num_layers=2
num_rollout_steps=2
num_peek_epochs=2
baseline="n/a"
ff_dropout_rate=0.1
reward_shaping_threshold=0
beta=0.02
num_paths_per_entity=-1

cached_qa_metadata_path="./.cache/kinshiphinton_latest/kinshiphinton_clean.json"
raw_QAData_path="data/_salesforce_compatible/mquake_qa_2hop.csv" # This ought to change a lot depending on what it is that you are testing.
bert_model_name="bert-base-uncased"
max_question_len=100

distmult_state_dict_path="model/kshinton-latest-distmult-xavier-200-200-0.003-0.3-0.1/model_best.tar"
complex_state_dict_path="model/kshinton-latest-complex-RV-xavier-200-200-0.003-0.3-0.1/model_best.tar"
conve_state_dict_path="model/kshipton-latest_standdata_full-graph/model_best.tar"

action_dropout_rate=0.95
action_dropout_anneal_interval=1000
action_dropout_rate=0.1
adam_beta1=0.9
adam_beta2=0.999
add_reverse_relations="True"
bandwidth=300

num_negative_samples=12
margin=0.5
