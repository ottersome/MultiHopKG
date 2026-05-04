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

# Defaults for optional ConvE/fact-network args. Many older configs omit these
# and rely on argparse defaults; keep experiment.sh usable for those configs.
: ${hidden_dropout_rate:=0.3}
: ${feat_dropout_rate:=0.2}
: ${emb_2D_d1:=10}
: ${emb_2D_d2:=20}
: ${num_out_channels:=32}
: ${kernel_size:=3}
: ${bert_model_name:=bert-base-uncased}
: ${max_question_len:=64}
: ${cached_qa_metadata_path:=./.cache}
: ${raw_QAData_path:=./datasets/data_preprocessed/mquake/mquake_qa_2hop.csv}
: ${recompute_qadata_cache:=False}
: ${disable_checkpoint_saving:=False}
: ${reward_shaping_threshold:=0}
: ${mu:=1.0}
: ${distmult_state_dict_path:=}
: ${complex_state_dict_path:=}
: ${conve_state_dict_path:=}

# Language conditioning flags
use_question_encoder_flag=''
if [[ $use_question_encoder = *"True"* ]]; then
    use_question_encoder_flag='--use_question_encoder'
fi
recompute_qadata_cache_flag=''
if [[ $recompute_qadata_cache = *"True"* ]]; then
    recompute_qadata_cache_flag='--recompute_qadata_cache'
fi
disable_checkpoint_saving_flag=''
if [[ $disable_checkpoint_saving = *"True"* ]]; then
    disable_checkpoint_saving_flag='--disable_checkpoint_saving'
fi
distmult_state_dict_path_arg=''
if [[ -n "$distmult_state_dict_path" ]]; then
    distmult_state_dict_path_arg="--distmult_state_dict_path $distmult_state_dict_path"
fi
complex_state_dict_path_arg=''
if [[ -n "$complex_state_dict_path" ]]; then
    complex_state_dict_path_arg="--complex_state_dict_path $complex_state_dict_path"
fi
conve_state_dict_path_arg=''
if [[ -n "$conve_state_dict_path" ]]; then
    conve_state_dict_path_arg="--conve_state_dict_path $conve_state_dict_path"
fi

cmd="python3 -m src.experiments \
    --data_dir $data_dir \
    $exp \
    --model $model \
    --bandwidth $bandwidth \
    --entity_dim $entity_dim \
    --relation_dim $relation_dim \
    --history_dim $history_dim \
    --history_num_layers $history_num_layers \
    --num_rollouts $num_rollouts \
    --num_rollout_steps $num_rollout_steps \
    --bucket_interval $bucket_interval \
    --num_epochs $num_epochs \
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
    --hidden_dropout_rate $hidden_dropout_rate \
    --feat_dropout_rate $feat_dropout_rate \
    --emb_2D_d1 $emb_2D_d1 \
    --emb_2D_d2 $emb_2D_d2 \
    --num_out_channels $num_out_channels \
    --kernel_size $kernel_size \
    --ff_dropout_rate $ff_dropout_rate \
    --action_dropout_rate $action_dropout_rate \
    --action_dropout_anneal_interval $action_dropout_anneal_interval \
    $relation_only_flag \
    --beta $beta \
    --beam_size $beam_size \
    --num_paths_per_entity $num_paths_per_entity \
    $group_examples_by_query_flag \
    $use_action_space_bucketing_flag \
    $use_question_encoder_flag \
    $recompute_qadata_cache_flag \
    $disable_checkpoint_saving_flag \
    --bert_model_name $bert_model_name \
    --max_question_len $max_question_len \
    --cached_qa_metadata_path $cached_qa_metadata_path \
    --raw_QAData_path $raw_QAData_path \
    --reward_shaping_threshold $reward_shaping_threshold \
    --mu $mu \
    $distmult_state_dict_path_arg \
    $complex_state_dict_path_arg \
    $conve_state_dict_path_arg \
    --gpu $gpu \
    $ARGS"

# NOTE: This might be needed later as arguments to the command above
# --max_question_len $max_question_len \
# --question_texts_path "$question_texts_path" \
# --QAData_path $QAData_path \

echo "Executing $cmd"


$cmd
