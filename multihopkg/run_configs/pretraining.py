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

    # -------------------- General Data Parameters --------------------
    ap.add_argument("--path_dataraw", type=str, default="./data/FBWikiV4/MQuAKE-CF.json")
    ap.add_argument("--path_cache_dir", type=str, default="./.cache/")
    ap.add_argument("--tvt_split", type=list, default=[0.8, 0.1, 0.1], help="Train-Valid-Test Splits")
    ap.add_argument("--path_entities_dict", type=str, default="./data/FBWikiV4/entities.dict", help="Entities Dictionary")
    ap.add_argument("--path_relations_dict", type=str, default="./data/FBWikiV4/relations.dict", help="Relations Dictionary")

    # -------------------- Embeddings Configs --------------------
    ap.add_argument("--embedding_model_type", type=str, default="pRotatE")
    ap.add_argument("--path_entities_embeddings", type=str, default="./models/protatE_FBWikiV4/entity_embedding.npy", help="Path to Entity Embeddings")
    ap.add_argument("--path_relations_embeddings", type=str, default="./models/protatE_FBWikiV4/relation_embedding.npy", help="Path Relations Embeddings")

    # -------------------- Language Modeling Parameters --------------------
    ap.add_argument("--question_tokenizer_name", type=str, default="bert-base-uncased")
    ap.add_argument("--answer_tokenizer_name", type=str, default="facebook/bart-base")

    args = ap.parse_args()
    # Some sanity checks/helps
    assert sum(args.tvt_split) == 1.0, f"Train Val Test split ({args.tvt_split}) does not add to one"
    os.makedirs(args.path_cache_dir, exist_ok = True)

    # Do the argument overloading
    if args.preferred_config:
        args = overload_parse_defaults_with_yaml(args.preferred_config, args)

    return args
