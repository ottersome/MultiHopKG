#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD"

SWEEP_CONFIG="${1:-sweeps/conve_mquake_st_prelim.yaml}"

cat <<EOF
Create the sweep:
  wandb sweep --entity halcyon-solutions --project <project-from-yaml> "$SWEEP_CONFIG"

Run an agent after W&B prints the sweep id:
  wandb agent <entity>/<project>/<sweep_id>

For a quick local smoke test without creating a sweep, run:
  python -m src.experiments --train --wandb --wandb_mode offline \\
    --data_dir data/mquake_st --model conve \\
    --add_reversed_training_edges --group_examples_by_query \\
    --entity_dim 200 --relation_dim 200 --emb_2D_d1 10 --emb_2D_d2 20 \\
    --num_epochs 2 --num_wait_epochs 2 --num_peek_epochs 1 \\
    --batch_size 32 --train_batch_size 32 --dev_batch_size 16 \\
    --learning_rate 0.003 --emb_dropout_rate 0.2 \\
    --hidden_dropout_rate 0.3 --feat_dropout_rate 0.2 \\
    --num_out_channels 32 --kernel_size 3 --beam_size 128 --gpu 0

KinshipHintonLatest micro-dataset sweep:
  wandb sweep --entity halcyon-solutions --project salesforce-entity_training sweeps/conve_kinshiphinton_latest_prelim.yaml

Final MQuAKE-ST NLP reward-shaping sweep:
  wandb sweep --entity halcyon-solutions --project salesforce-rsnlp-conve-mquake-microsweep sweeps/rs_nlp_mquake_st_final.yaml

Final KinshipHintonLatest NLP reward-shaping sweep:
  wandb sweep --entity halcyon-solutions --project rs-nlp-conve-kinship sweeps/rs_nlp_kinshiphinton_latest_final.yaml
EOF
