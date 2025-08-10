# Multi-Hop Knowledge Graph Reasoning with Reward Shaping

Based off the work by [salesforce](https://github.com/salesforce/MultiHopKG) but heavily edited.


 Prep

## Poetry (Optional)

Not necessary but you might find it convenient to install poetry:

```sh
sudo apt install poetry
```

Then install all packages here

```sh
poetry install --no-root
```

Then enter the environment

```sh
poetry shell
```

## Data

Make sure you've got your hands on `data-release.tgz` and, while in the repo root, decompress it with:

```sh
tar -xvf data-release.tgz
```

# Running it

## (Old) Masked Language Modeling Training

```sh
python mlm_training.py
```

## Knowledge Graph Embedding Training

This serves as a pseudo-preparation step. Further modules in our training require for there to be pretrained graph embeddings. 
We use the script `./kge_train.py` for that.
A sample of how that looks like with the mquake.yaml dataset:

```sh
python kge_train.py  --saved_config_path=configs/embedding_training/mquake.yaml --cuda
```

## Pretraining Round

Our Graph-LLM Transformer requires a pretraining round before being plugged into the final algorithm.
This can be run via 

```sh
python pretraining.py --preferred_config=./configs/pretraining/transE_mquake_dim500.yaml
```


## Train and Evaluate KGE Model
If you want to **train** you can find a best config in `configs/sun_best_config.sh`, just copy the one you need from there and use it in the command line.

**Autoencoder** feature can be enabled, see `configs/sun_best_config.sh`. Fully tested only on **pRotatE FB15k**.
Doesn't work if `-dr` or `--double_relation_embedding` is `True`.

Model will be saved to `models/`

To **evaluate** the model:
- Use `tests/conftest.py` if you want to run unit tests or validate specific components of the model. This script is designed for testing purposes and ensures that individual parts of the system are functioning correctly.
- Use `kge_train.py` if you want to perform a full evaluation of the trained model. This script is intended for end-to-end testing and generating evaluation metrics.

If you choose to use `kge_train.py`, you can run the following command:
`CUDA_VISIBLE_DEVICES=[DEVICE NUMBER] python -u kge_train.py --do_test --cuda -init models/[MODEL NAME] --save_path [PATH TO SAVE RESULTS] [--autoencoder_flag] [--autoencoder_hidden_dim VALUE] [--autoencoder_lambda VALUE]`
