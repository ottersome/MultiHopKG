#!/bin/bash

export PYTHONPATH=`pwd`
echo $PYTHONPATH

source $1
exp=$2
gpu=$3
ARGS=${@:4}

group_examples_by_query_flag=''
if [[ $group_examples_by_query = *"True"* ]]; then
    group_examples_by_query_flag="--group_examples_by_query"
fi
relation_only_flag=''
if [[ $relation_only = *"True"* ]]; then
    relation_only_flag="--relation_only"
fi
use_action_space_bucketing_flag=''
if [[ $use_action_space_bucketing = *"True"* ]]; then
    use_action_space_bucketing_flag='--use_action_space_bucketing'
fi
recompute_qadata_cache_flag=''
if [[ $recompute_qadata_cache = *"True"* ]]; then
    recompute_qadata_cache_flag="--recompute_qadata_cache"
fi

use_question_encoder_flag=''
if [[ $use_question_encoder = *"True"* ]]; then
    use_question_encoder_flag="--use_question_encoder"
fi

cmd="python3 -m src.experiments \
    --data_dir $data_dir \
    $exp \
    --model $model \
    --bandwidth $bandwidth \
    --entity_dim $entity_dim \
    --emb_2D_d1 $emb_2D_d1 \
    --emb_2D_d2 $emb_2D_d2 \
    --kernel_size $kernel_size \
    --relation_dim $relation_dim \
    --history_dim $history_dim \
    --history_num_layers $history_num_layers \
    --use_action_space_bucketing \
    --num_rollouts $num_rollouts \
    --num_rollout_steps $num_rollout_steps \
    --bucket_interval $bucket_interval \
    --num_epochs $num_epochs \
    --num_out_channels $num_out_channels \
    --num_wait_epochs $num_wait_epochs \
    --num_peek_epochs $num_peek_epochs \
    --batch_size $batch_size \
    --train_batch_size $train_batch_size \
    --dev_batch_size $dev_batch_size \
    --margin $margin \
    --learning_rate $learning_rate \
    --baseline $baseline \
    --grad_norm $grad_norm \
    --emb_dropout_rate $emb_dropout_rate \
    --ff_dropout_rate $ff_dropout_rate \
    --action_dropout_rate $action_dropout_rate \
    --action_dropout_anneal_interval $action_dropout_anneal_interval \
    --reward_shaping_threshold $reward_shaping_threshold \
    $relation_only_flag \
    --beta $beta \
    --beam_size $beam_size \
    --num_paths_per_entity $num_paths_per_entity \
    $group_examples_by_query_flag \
    $use_action_space_bucketing_flag \
    $use_question_encoder_flag \
    --cached_qa_metadata_path $cached_qa_metadata_path \
    --raw_QAData_path $raw_QAData_path \
    $recompute_qadata_cache_flag \
    --bert_model_name $bert_model_name \
    --max_question_len $max_question_len \
    --mu $mu \
    --distmult_state_dict_path $distmult_state_dict_path \
    --complex_state_dict_path $complex_state_dict_path \
    --conve_state_dict_path $conve_state_dict_path \
    --gpu $gpu \
    $ARGS"

echo "Executing $cmd"

$cmd
