# Multi-Hop Knowledge Graph Reasoning with Reward Shaping

This repository is a maintained research fork of [Multi-Hop Knowledge Graph
Reasoning with Reward Shaping](https://arxiv.org/abs/1808.10568) (Lin, Socher,
and Xiong, EMNLP 2018), focused on question-answering (QA) experiments and the
evaluation protocol used for the current work.

Only the QA workflows, including KinshipHintonLatest and MQuAKE, are supported.
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
uv run -- ./experiment-rs-nlp.sh configs/kinshiphinton_latest-rs-nlp-conve.sh --train 0
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
the current KinshipHintonLatest configuration expects:

```text
data/KinshipHintonLatest/kinship_qa_nhop.csv
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
uv run -- ./experiment-emb.sh configs/kinshiphinton_latest-conve.sh --train 0
```

### Question-conditioned reward shaping

For the current KinshipHintonLatest setup:

```sh
uv run -- ./experiment-rs-nlp.sh \
  configs/kinshiphinton_latest-rs-nlp-conve.sh --train 0 --seed=44
```

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
  configs/kinshiphinton_latest-rs-nlp-conve.sh --inference 0 \
  --checkpoint_path model/path/to/model_best.tar --seed=44
```

Add `--save_beam_search_paths` to save decoded beam-search paths. For
question-conditioned evaluation, `--evaluate_paraphrases` expands evaluation
rows over paraphrases; add `--filter_original_paraphrases` to exclude copies of
the original question. `--eval_hops 2,3` restricts QA evaluation to those hops.

### Released checkpoint collection

Download the released `model/final_evaluation/` artifacts before running the
batch evaluator. It discovers checkpoint directories recursively; each needs
its `.sh` config alongside the `s*_model.tar` files.

```sh
uv run python batch_eval.py model/final_evaluation
```

By default, logs and per-checkpoint JSON metrics go below
`batch_eval_logs/<timestamp>/`. Select a GPU or persist aggregate results when
needed:

```sh
uv run python batch_eval.py --gpu 1 \
  --output_json results/evaluation.json --output_csv results/evaluation.csv \
  model/final_evaluation
```

Arguments after `--` are passed to every checkpoint evaluation:

```sh
uv run python batch_eval.py model/final_evaluation -- --evaluate_paraphrases
```

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
