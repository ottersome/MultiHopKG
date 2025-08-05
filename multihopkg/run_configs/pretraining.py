"""
Mostly to keep clean the main file from evergrowing amount of parameters
"""
import argparse
import os

from multihopkg.run_configs.common import overload_parse_defaults_with_yaml


def get_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()

    # --- BEWARE: If this set the parameters below will be overwritten with it ---
    # example config file: ./configs/pretraining/pretraining_alpha.yaml
    ap.add_argument("--preferred_config", type=str)

    # -------------------- General Parameters --------------------
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force_recompute_cache", action="store_true", help="Will wipe old cache and recompute it.")
    ap.add_argument("--debug", "-d", action="store_true", help="Debugpy activation")
    ap.add_argument("--device", type=str, default="cuda", help="What device will be used for training.")
    ap.add_argument("--verbose", "-v",action="store_true", help="Whether to run more verbosely")

    # -------------------- Training Parameters --------------------
    ap.add_argument("--epochs", "-e", type=int, default=10, help="How many epochs to use")
    ap.add_argument("--batch_size", "-b", type=int, default=64, help="Batch size")
    ap.add_argument("--val_every_n_batches", type=int, default=50, help="How many batches to run validation on")

    # -------------------- General Data Parameters --------------------
    ap.add_argument("--path_dataraw", type=str, default="./data/mquake/MQuAKE-CF.json")
    ap.add_argument("--path_cache_dir", type=str, default="./.cache/mquake/")
    ap.add_argument("--tvt_split", type=list, default=[0.8, 0.1, 0.1], help="Train-Valid-Test Splits")
    ap.add_argument("--path_entities_dict", type=str, default="./data/mquake/entities.dict", help="Entities Dictionary")
    ap.add_argument("--path_relations_dict", type=str, default="./data/mquake/relations.dict", help="Relations Dictionary")

    # -------------------- Language Modeling Parameters --------------------
    ap.add_argument("--question_tokenizer_name", type=str, default="facebook/bart-base")
    ap.add_argument("--answer_tokenizer_name", type=str, default="facebook/bart-base")
    ap.add_argument("--hunch_answer_model", type=str, default="facebook/bart-base")
    ap.add_argument("--lr", type=float, default=1e-3)

    args = ap.parse_args()
    # Some sanity checks/helps
    assert sum(args.tvt_split) == 1.0, f"Train Val Test split ({args.tvt_split}) does not add to one"
    os.makedirs(args.path_cache_dir, exist_ok = True)

    # Do the argument overloading
    if args.preferred_config:
        args = overload_parse_defaults_with_yaml(args.preferred_config, args)

    return args
