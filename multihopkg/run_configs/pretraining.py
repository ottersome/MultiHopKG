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
    ap.add_argument("--num_warmup_steps", "-w", type=int, default=300, help="Amont of gradient steps to warmup before engaging in the next step of scheduler.")

    # -------------------- Logging Parameters --------------------
    ap.add_argument("-W", "--wandb_on", action="store_true", help="Enable Weights & Biases experiment tracking")
    ap.add_argument("--wandb_project", type=str, default="gtllm_pretraining",help="wandb: Project name to group runs")
    ap.add_argument("--wr_name", type=str, help="wandb: Unique name for this run")
    ap.add_argument("--wr_notes", type=str, help="wandb: Additional notes for this run")

    # -------------------- General Data Parameters --------------------
    ap.add_argument("--path_mquake_data", type=str, default="./data/mquake/")
    ap.add_argument("--path_cache_dir", type=str, default="./.cache/mquake/")
    ap.add_argument("--tvt_split", type=list, default=[0.8, 0.1, 0.1], help="Train-Valid-Test Splits")
    ap.add_argument("--path_graph_emb_data", type=str, default="./models/graph_embeddings/transE_mquake_dim500", help="Path to Entity Embeddings")
    ap.add_argument("--outPath_save_model", type=str, default="./models/gtllm/")

    # -------------------- Language Modeling Parameters --------------------
    # Generally speaking these two are the same.
    ap.add_argument("--hunchbart_base_llm_tokenizer", type=str, default="facebook/bart-base")
    ap.add_argument("--hunchbart_base_llm_model", type=str, default="facebook/bart-base")
    ap.add_argument("--baseline_lr", type=float, default=1e-3)
    ap.add_argument("--minimum_lr", type=float, default=1e-6)

    args = ap.parse_args()
    # Some sanity checks/helps
    assert sum(args.tvt_split) == 1.0, f"Train Val Test split ({args.tvt_split}) does not add to one"
    os.makedirs(args.path_cache_dir, exist_ok = True)

    # Do the argument overloading
    if args.preferred_config:
        args = overload_parse_defaults_with_yaml(args.preferred_config, args)

    return args
