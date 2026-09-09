#!/usr/bin/env bash

data_dir="data/KinshipHintonLatest/"
model="point.rs.conve"
add_reversed_training_edges="True"
group_examples_by_query="True"
emb_2D_d1=3
emb_2D_d2=4
entity_dim=12
relation_dim=12
# num_rollouts generally for training
num_rollouts=32
bucket_interval=10
num_epochs=100
num_wait_epochs=500
batch_size=16
train_batch_size=16
dev_batch_size=64
learning_rate=0.0016
grad_norm=0
emb_dropout_rate=0.1
# beam_size generally for inferene
beam_size=100
num_out_channels=8
kernel_size=2
use_question_encoder="True"
allow_direct_answer_edges="True"

history_dim=64
history_num_layers=2
num_rollout_steps=3
num_peek_epochs=2
checkpoint_keep_last=3
baseline="n/a"
ff_dropout_rate=0.1
reward_shaping_threshold=0
beta=0.02
num_paths_per_entity=-1

cached_qa_metadata_path="./.cache/kinshiphinton_latest/kinshiphinton_clean.json"
raw_QAData_path="data/KinshipHintonLatest/kinship_qa_nhop.csv" # This ought to change a lot depending on what it is that you are testing.
bert_model_name="bert-base-uncased"
max_question_len=100
mu=1.0

# For now all we really have is conve
distmult_state_dict_path="${PWD}/model/KinshipHintonLatest-conve-RV-xavier-12-12-0.01-8-2-0.1-0.1-0.2-0.05/checkpoint-90.tar" # 不在用，但還要定義
complex_state_dict_path="${PWD}/model/KinshipHintonLatest-conve-RV-xavier-12-12-0.01-8-2-0.1-0.1-0.2-0.05/checkpoint-90.tar" # 不在用，但還要定義
conve_state_dict_path="${PWD}/model/KinshipHintonLatest-conve-RV-xavier-12-12-0.01-8-2-0.1-0.1-0.2-0.05/checkpoint-90.tar"
bandwidth=50

action_dropout_rate=0.3
action_dropout_anneal_interval=1000
action_dropout_rate=0.1
adam_beta1=0.9
adam_beta2=0.999
add_reverse_relations="True"
checkpoint_keep_best=3 # Just for clarity 

# Per hop testing: Generally leave empty if you dont want to do tis. 
train_hop=0 # Means that it takes all hops for training
# evaluate_per_hop="True" # By default we do not evaluate by per hop
# eval_hops= # IF we did you would fill that up 

num_negative_samples=12
margin=0.5
seed=12

wandb_project="rs-nlp-conve-kinship"
