# Multi-Hop Knowledge Graph Reasoning with Reward Shaping

This repository is a maintained research fork of [Multi-Hop Knowledge Graph
Reasoning with Reward Shaping](https://arxiv.org/abs/1808.10568) (Lin, Socher,
and Xiong, EMNLP 2018), focused on question-answering (QA) experiments and the
evaluation protocol used for the current work.

Only the QA workflows, including KinshipHinton and MQuAKE, are supported.
The legacy non-QA/link-prediction datasets and configurations remain in the
repository for historical reference, but are untested and not supported.

## Requirements

Training and evaluation require an NVIDIA GPU. The project uses Python 3.12 and
the locked dependencies managed by [uv](https://docs.astral.sh/):

```sh
uv python install 3.12
uv sync
```

Use `uv run` for Python entry points. Shell launchers use the active Python
environment, so either activate `.venv` first or invoke them through `uv run --`:

```sh
uv run python -m pytest
uv run -- ./experiment-rs-nlp.sh configs/kinshiphinton-rs-nlp-conve.sh --train 0
```

The bundled Dockerfile targets the original 2018 software stack and is not a
supported environment for this Python 3.12 version of the project.

## Command convention

All `experiment*.sh` launchers use this positional convention:

```text
./experiment-*.sh CONFIG ACTION GPU_ID [extra src.experiments options]
```

For example, `--train 0` means “train on GPU 0”; `0` is not a value for the
Boolean `--train` flag. Options after the GPU ID override config values, so
`--seed=44` is a valid reproducible override.

## Data preparation

Use the QA data specified by the configuration you are running. In particular,
the current KinshipHinton configuration expects:

```text
data/KinshipHinton/kinship_qa_nhop.csv
```

MQuAKE configurations similarly identify their required QA CSV with
`raw_QAData_path`. Cached QA metadata is created under `.cache/` as needed.
Review these paths, plus the pretrained ConvE checkpoint paths, in a copied
configuration before launching a run.

`data-release.tgz` belongs to the original non-QA release; it is not part of a
supported workflow in this version of the repository.

## Training

Configurations in `configs/` are the source of truth for model and data
settings. Copy one before changing hyperparameters so published configurations
remain reproducible.

### QA ConvE prerequisite

The reward-shaped QA configurations use a pretrained ConvE fact model. Train
or supply that checkpoint, then set `conve_state_dict_path` in the QA
configuration to its location:

```sh
uv run -- ./experiment-emb.sh configs/kinshiphinton-conve.sh --train 0
```

The supported ConvE configurations train on `lp_train.triples`, the deduplicated
union of `raw.kb`, `train.triples`, `dev.triples`, and `test.triples`.  ConvE is
used as a transductive KG substrate for the QA policy, so this is intentional:
it is not a held-out link-prediction benchmark.  Build or refresh these files
before standalone LP training:

```sh
make prepare-lp-data
```

For MetaQA and MQuAKE, LP uses a generous epoch ceiling and early-stops on the
existing dev-MRR signal as an optimization/convergence criterion. Because the
full KG includes those triples, do not report that MRR as a held-out LP result.
RL early stopping, in contrast, uses held-out QA development questions. Kinship
always runs its paper-specified 100 epochs.

All LP (including ConvE) and RL runs use `early_stopping_min_delta=0.001` by
default: an increase in the monitored development metric must be strictly
larger than that value to reset patience. The numerically best checkpoint is
still retained. Set
`early_stopping_min_delta=0.01` in a shell config (or pass
`--early_stopping_min_delta 0.01`) for a coarser tolerance; set it to `0` to
restore the previous strict-any-increase behavior.

W&B scalar logging is safe to enable with `WANDB=1`. Gradient/parameter
watching is off by default because it can consume substantial memory for the
question-conditioned BERT policy; opt in only for short diagnostics with
`wandb_watch_gradients="True"` in the relevant config.

### Question-conditioned reward shaping

For the current KinshipHintonLatest setup:

```sh
uv run -- ./experiment-rs-nlp.sh \
  configs/kinshiphinton-rs-nlp-conve.sh --train 0 --seed=44
```

### Reproduce the full experiment suite

From the repository root, the complete reviewer workflow is one command:

```sh
make reproduce GPU=0
```

It rebuilds the traversal graphs, prepares full-KG LP training data, trains one
ConvE substrate per KG (MQuAKE-SA and MQuAKE-MA share theirs), trains three RL
seeds (`12`, `42`, and `1337`) per QA setting, then evaluates each three-seed
family separately. Results and staged checkpoints go to
`model/final_evaluation/reviewer_3seed/`.

The pipeline also caches English Wikidata labels for MQuAKE evidence paths,
including readable inverse-relation names. An offline run can skip this optional
presentation step with `WIKIDATA_LABELS=0`; raw IDs remain in every export.

To use Weights & Biases, add `WANDB=1`. To use another output namespace or
seed list, use `RUN_NAME=my_run RL_SEEDS="12 42 1337"`.

```sh
make reproduce GPU=0 WANDB=1 RUN_NAME=my_run
```

`make train-all GPU=0` performs only preprocessing and training; `make evaluate
GPU=0` evaluates an already staged run.

For MQuAKE single-answer QA:

```sh
uv run -- ./experiment-rs-nlp.sh \
  configs/mquake_st-rs-nlp-conve.sh --train 0 --seed=12
```

The launcher passes extra options to `src.experiments`; useful examples include
`--num_epochs`, `--checkpoint_path`, `--train_hop`, and
`--evaluate_per_hop`. Training writes checkpoints under `model/` by default.

## Evaluation

### One checkpoint

Use the same launcher and configuration as training, replace `--train` with
`--inference`, and pass the checkpoint explicitly. Inference reports both dev
and test results.

```sh
uv run -- ./experiment-rs-nlp.sh \
  configs/kinshiphinton-rs-nlp-conve.sh --inference 0 \
  --checkpoint_path model/path/to/model_best.tar --seed=44
```

Add `--save_beam_search_paths` to save decoded beam-search paths. For
question-conditioned evaluation, `--evaluate_paraphrases` expands evaluation
rows over paraphrases; add `--filter_original_paraphrases` to exclude copies of
the original question. `--eval_hops 2,3` restricts QA evaluation to those hops.

To retain one detailed evaluation record per question, pass
`--evaluation_examples_output_path results/eval-{split}.jsonl`.  Each JSONL
record includes decoded question text, readable gold/predicted paths and answers, path/relation
metrics, cumulative negative log-probability, beam-path entropy, hop counts,
and Hit@1. Aggregate metrics also report the predicted-hop mean, its distribution,
and the predicted-hop mean for each gold-hop bucket. `{split}` is replaced with `dev` or `test`; omit it to put both
splits in one file (the test records are appended).

### Released checkpoint collection

Download the released `model/final_evaluation/` artifacts before running the
batch evaluator. It discovers checkpoint directories recursively; each needs
its `.sh` config alongside the `s*_model.tar` files.

```sh
uv run python batch_eval.py model/final_evaluation
```

By default, logs and per-checkpoint JSON metrics go below
`batch_eval_logs/<timestamp>/`; detailed dev/test JSONL evaluation records go
under its `examples/` subdirectory. This also includes the predicted-hop
statistics in the JSON metrics and summary tables. Select a GPU or persist
aggregate results when needed:

```sh
uv run python batch_eval.py --gpu 1 \
  --output_json results/evaluation.json --output_csv results/evaluation.csv \
  model/final_evaluation
```

Arguments after `--` are passed to every checkpoint evaluation:

```sh
uv run python batch_eval.py model/final_evaluation -- --evaluate_paraphrases
```

Evaluate several checkpoint collections together by listing their directories:

```sh
uv run python batch_eval.py \
  model/final_evaluation/kinshiphinton \
  model/final_evaluation/mquake_ma_rl_binary \
  model/final_evaluation/metaqa_rl_binary \
  model/final_evaluation/mquake_sa_rl_binary
```

Use `--evaluation_examples_dir results/examples` to choose where the detailed
JSONL files are written, or `--no_evaluation_examples` to skip them.

Use `--dry_run` to inspect generated commands before a large evaluation.

## Tests

```sh
uv run python -m pytest
```

## Citation

If you use the original MultiHopKG work, please cite:

```bibtex
@inproceedings{LinRX2018:MultiHopKG,
  author = {Xi Victoria Lin and Richard Socher and Caiming Xiong},
  title = {Multi-Hop Knowledge Graph Reasoning with Reward Shaping},
  booktitle = {Proceedings of the 2018 Conference on Empirical Methods in Natural
               Language Processing, {EMNLP} 2018},
  year = {2018}
}
```

## Project note

This is a modified version of the original project for the current evaluation
work. See `metrics.md` for the repository-specific metric and paraphrase
evaluation protocol.
