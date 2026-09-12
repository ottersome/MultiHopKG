# ⚠️⚠️⚠️ Instructions ⚠️⚠️⚠️

## Python environment (uv)

This repository pins Python **3.12** in [`.python-version`](.python-version).
Install [uv](https://docs.astral.sh/uv/getting-started/installation/) first, then
create the project environment and install the locked dependencies from the
repository root:

```sh
uv python install 3.12
uv sync
```

`uv sync` creates `.venv/` automatically. Run commands inside that environment
with `uv run`; no manual activation is required:

```sh
uv run python -m unittest tests/test_faithfulness_metrics.py
uv run python batch_eval.py model/final_evaluation
```

The configured PyTorch packages use the CUDA 13.0 wheel index. Ensure the host
has a compatible NVIDIA driver before running GPU evaluation.

Before you run evaluation make sure to get the files from the kaggle. 
After you download it the `model/` directory should have the following structure:

```txt
model/final_evaluation
├── kinship
│   ├── kinshiphinton_latest-rs-nlp-conve.sh
│   ├── parameters.json
│   ├── s42_model.tar
│   ├── s43_model.tar
│   └── s44_model.tar
├── lp_embeddings_conve
│   ├── kinship_conve-best.tar
│   ├── metaqa_conve-epoch42.tar
│   ├── metaqa_encoded_conve-best.tar
│   ├── mquake_st_conve-best.tar
│   └── NO_EVAL
├── metaqa_rl_binary
│   ├── metaqa_encoded-rs-nlp-conve.sh
│   ├── parameters.json
│   ├── s12_model.tar
│   ├── s232_model.tar
│   └── s42_model.tar
├── mquake_ma_rl_binary
│   ├── mquake_st_ma-rs-nlp-conve.sh
│   ├── parameters.json
│   ├── s12_model.tar
│   ├── s29_model.tar
│   └── s44_model.tar
├── mquake_sa_rl_binary
│   ├── mquake_st-rs-nlp-conve.sh
│   ├── paramaters.json
│   ├── s00_model.tar
│   ├── s01_model.tar
│   └── s02_model.tar
└── README.md
```

Evaluation can be run through 

```sh
uv run python batch_eval.py model/final_evaluation
```

This will write the evaluations to `<repository_root>/<date>-<time>/<model>_s<seed>.metrics.log`


Make sure to run with `--evaluate_paraphrases` after *end-of-options delimeter*. If you want to evaluate paraphrases
i.e.:

```sh
uv run python batch_eval.py model/final_evaluation -- --evaluate_paraphrases
```

# Disclaimer. 

This is a modified version of "Multi-Hop Knowledge Graph Reasoning with Reward Shaping" with the purpose of fitting neurips 2026 evaluations.
Original work is by Xi Victoria Lin, Richard Socher and Caiming Xiong. [Multi-Hop Knowledge Graph Reasoning with Reward Shaping](https://arxiv.org/abs/1808.10568). EMNLP 2018.
