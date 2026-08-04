# ⚠️⚠️⚠️ NEURIPS Instructions ⚠️⚠️⚠️

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

To install the requirements you may run:

```sh
pip install requirements.txt
```

Evaluation can be run through 

```sh
python batch_eval.py model/final_evaluation
```

This will write the evaluations to `<repository_root>/<date>-<time>/<model>_s<seed>.metrics.log`


Make sure to run with `--evaluate_paraphrases` after *end-of-options delimeter*. If you want to evaluate paraphrases
i.e.:

```sh
python batch_eval.py model/final_evaluation -- --evaluate_paraphrases
```

# Disclaimer. 

This is a modified version of "Multi-Hop Knowledge Graph Reasoning with Reward Shaping" with the purpose of fitting neurips 2026 evaluations.
Original work is by Xi Victoria Lin, Richard Socher and Caiming Xiong. [Multi-Hop Knowledge Graph Reasoning with Reward Shaping](https://arxiv.org/abs/1808.10568). EMNLP 2018.
