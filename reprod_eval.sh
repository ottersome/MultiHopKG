#!/bin/sh

# How to run
# Make sure you place your downloaded model directory to ./output/final_evaluation
RC='\033[31m'
GC='\033[32m'
YC='\033[33m'
RC='\033[m'
MODEL_ROOT=$1

SPECIFIC_MODEL_DIRS="kinship metaqa_rl_binary mquake_ma_rl_binary mquake_sa_rl_binary"

# set -x
for model in $SPECIFIC_MODEL_DIRS; do
  printf "${YC}Running Evaluations for model: ${model}$RC"
  cur_path="$MODEL_ROOT/$model/"
  find "$cur_path" -type f -name "s*_model.tar" | while IFS= read -r file
  do 
    base=$(basename "$file")
    printf "${RC} base is $base $RC\n"
    number=$(printf "%s\n" $base | sed -n 's/^s\([0-9][0-9]*\)_model.tar$/\1/p')
    printf "Evaluating checkpoint with seed ${YC}${number}${RC}\n"
    set +x
    ./experiment-rs-nlp.sh configs/kinshiphinton_latest-rs-nlp-conve.sh --inference 0 \
        --checkpoint_path "${MODEL_ROOT}/${model}/${seed}_model.tar" \
        --rollout_eval_num_rollouts 128 \
        --rollout_eval_batch_size 16
    set -x
  done
done
exit 0
