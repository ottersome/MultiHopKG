#!/bin/bash

export PYTHONPATH=`pwd`
echo $PYTHONPATH

source $1
exp=$2
gpu=$3
ARGS=${@:4}

# `margin` is retained as a legacy argparse option, but is not used by the
# embedding models.  The launcher forwards it for compatibility, so ensure a
# config that intentionally omits it never expands to a bare `--margin`.
: "${margin:=0}"
: "${early_stopping_min_delta:=0.01}"

add_reversed_training_edges_flag=''
if [[ $add_reversed_training_edges = *"True"* ]]; then
    add_reversed_training_edges_flag="--add_reversed_training_edges"
fi
add_reverse_relations_flag=''
if [[ $add_reverse_relations = *"True"* ]]; then
    add_reverse_relations_flag="--add_reverse_relations"
fi
lp_train_triples_path_flag=''
if [[ -n "$lp_train_triples_path" ]]; then
    lp_train_triples_path_flag="--lp_train_triples_path $lp_train_triples_path"
fi
group_examples_by_query_flag=''
if [[ $group_examples_by_query = *"True"* ]]; then
    group_examples_by_query_flag="--group_examples_by_query"
fi
wandb_watch_gradients_flag=''
if [[ $wandb_watch_gradients = *"True"* ]]; then
    wandb_watch_gradients_flag="--wandb_watch_gradients"
fi

cmd="python3 -m src.experiments \
    --data_dir $data_dir \
    $lp_train_triples_path_flag \
    $exp \
    --model $model \
    --entity_dim $entity_dim \
    --relation_dim $relation_dim \
    --emb_2D_d1 $emb_2D_d1 \
    --emb_2D_d2 $emb_2D_d2 \
    --num_rollouts $num_rollouts \
    --bucket_interval $bucket_interval \
    --num_epochs $num_epochs \
    --num_wait_epochs $num_wait_epochs \
    --early_stopping_min_delta $early_stopping_min_delta \
    --batch_size $batch_size \
    --train_batch_size $train_batch_size \
    --dev_batch_size $dev_batch_size \
    --num_negative_samples $num_negative_samples \
    --margin $margin \
    --learning_rate $learning_rate \
    --grad_norm $grad_norm \
    --emb_dropout_rate $emb_dropout_rate \
    --num_out_channels $num_out_channels \
    --hidden_dropout_rate $hidden_dropout_rate \
    --feat_dropout_rate $feat_dropout_rate\
    --label_smoothing_epsilon $label_smoothing_epsilon \
    --kernel_size $kernel_size \
    --beam_size $beam_size \
    --wandb_project $wandb_project \
    $wandb_watch_gradients_flag \
    $group_examples_by_query_flag \
    $add_reverse_relations_flag \
    $add_reversed_training_edges_flag \
    --gpu $gpu \
    $ARGS"

echo "Executing $cmd"

$cmd
