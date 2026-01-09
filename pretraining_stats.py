import ast
import os
from collections import Counter
from typing import Dict, List, Sequence, Tuple

import matplotlib

# Force a non-interactive backend so the script works in headless environments.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
import torch
from transformers.models.bert import BertModel, BertTokenizer

from multihopkg import data_utils
from multihopkg.data_utils import load_native_index
from multihopkg.logging import setup_logger
from multihopkg.run_configs.pretraining import get_args
from multihopkg.utils.data_structures import DataPartitions
from multihopkg.utils.setup import set_seeds

logger = setup_logger("__PRETRAINING_STATS__")


def _normalize_path(path_value: object) -> List[int]:
    """
    Ensure we always operate with a Python list of ints alternating entity-relation-entity.
    """
    if isinstance(path_value, list):
        path = path_value
    elif isinstance(path_value, np.ndarray):
        path = path_value.tolist()
    elif isinstance(path_value, str):
        try:
            parsed = ast.literal_eval(path_value)
            path = parsed if isinstance(parsed, list) else []
        except (ValueError, SyntaxError):
            path = []
    elif isinstance(path_value, Sequence):
        path = list(path_value)  # type: ignore[arg-type]
    else:
        path = []

    path = [int(x) for x in path if isinstance(x, (int, np.integer))]
    if len(path) < 3:
        return []
    return path


def _collect_counts(df: pd.DataFrame) -> Tuple[Counter, Counter]:
    entities = Counter()
    relations = Counter()

    if "triples_ints" not in df.columns:
        logger.warning("Dataframe does not contain 'triples_ints'. Skipping counts.")
        return entities, relations

    for raw_path in df["triples_ints"]:
        path = _normalize_path(raw_path)
        if not path:
            continue
        entities.update(path[::2])
        relations.update(path[1::2])
    return entities, relations


def _compute_coverage(
    train_counts: Counter, target_counts: Counter
) -> Dict[str, float]:
    total_mentions = sum(target_counts.values())
    mentioned_seen = sum(
        freq for item_id, freq in target_counts.items() if train_counts.get(item_id, 0) > 0
    )
    unique_total = len(target_counts)
    unique_seen = sum(1 for item_id in target_counts if item_id in train_counts)
    coverage = {
        "unique_seen": unique_seen,
        "unique_total": unique_total,
        "unique_seen_ratio": (unique_seen / unique_total) if unique_total else 0.0,
        "mention_seen": mentioned_seen,
        "mention_total": total_mentions,
        "mention_seen_ratio": (mentioned_seen / total_mentions) if total_mentions else 0.0,
    }
    return coverage


def _build_topk_dataframe(
    train_counts: Counter,
    target_counts: Counter,
    id_to_label: Dict[int, str],
    top_k: int,
) -> pd.DataFrame:
    records = []
    for item_id, target_freq in target_counts.most_common():
        label = id_to_label.get(item_id, f"<missing:{item_id}>")
        records.append(
            {
                "id": item_id,
                "name": label,
                "target_freq": target_freq,
                "train_freq": train_counts.get(item_id, 0),
            }
        )
    if not records:
        return pd.DataFrame(columns=["id", "name", "target_freq", "train_freq"])
    df = pd.DataFrame.from_records(records)
    df = df.sort_values(by="target_freq", ascending=False)
    return df.head(top_k)


def _plot_frequency_overlap(
    entity_df: pd.DataFrame,
    relation_df: pd.DataFrame,
    split_name: str,
    output_path: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    datasets = [
        (entity_df, "Entities", axes[0]),
        (relation_df, "Relations", axes[1]),
    ]
    bar_width = 0.4

    for df, title, ax in datasets:
        ax.set_title(f"{title} overlap ({split_name} vs train)")
        if df.empty:
            ax.text(
                0.5,
                0.5,
                "No data available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.axis("off")
            continue
        idx = np.arange(len(df))
        ax.bar(
            idx - bar_width / 2,
            df["target_freq"],
            bar_width,
            label=f"{split_name} freq",
            color="#4c72b0",
        )
        ax.bar(
            idx + bar_width / 2,
            df["train_freq"],
            bar_width,
            label="train freq",
            color="#55a868",
        )
        ax.set_xticks(idx)
        ax.set_xticklabels(df["name"], rotation=45, ha="right", fontsize=9)
        ax.set_ylabel("Count")
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.3)

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _select_split(partitions: DataPartitions, split_name: str) -> pd.DataFrame:
    if split_name == "dev":
        return partitions.validation
    if split_name == "test":
        return partitions.test
    raise ValueError(f"Unsupported split '{split_name}'. Use 'dev' or 'test'.")


def _prepare_device(device_name: str) -> torch.device:
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        logger.warning(
            "CUDA device '%s' requested but not available. Falling back to CPU.",
            device_name,
        )
        return torch.device("cpu")
    return torch.device(device_name)


def load_datasets(args) -> Tuple[DataPartitions, Dict[int, str], Dict[int, str]]:
    path_entities_dict = os.path.join(args.path_general_data, "entities.dict")
    path_relations_dict = os.path.join(args.path_mquake_data, "relations.dict")
    id2ent, ent2id = load_native_index(path_entities_dict)
    id2rel, rel2id = load_native_index(path_relations_dict)

    device = _prepare_device(args.device)
    bert_model = BertModel.from_pretrained(args.bert_base_llm_model).to(device)
    bert_tokenizer = BertTokenizer.from_pretrained(args.bert_base_llm_tokenizer)

    train_df, dev_df, test_df, _ = data_utils.load_qa_data(
        cached_metadata_path=os.path.join(args.path_cache_dir, "mquake.json"),
        raw_QAData_path=os.path.join(args.path_mquake_data, "mquake_qna_ds.csv"),
        question_tokenizer_name=args.hunchbart_base_llm_tokenizer,
        answer_tokenizer_name=args.hunchbart_base_llm_tokenizer,
        entity2id=ent2id,
        relation2id=rel2id,
        logger=logger,
        bert_tokenizer=bert_tokenizer,
        bert_model=bert_model,
        force_recompute=args.force_recompute_cache,
        supervised=False,
    )

    partitions = DataPartitions(train_df, dev_df, test_df)

    return partitions, id2ent, id2rel


def _count_paths_with_unseen_items(
    df: pd.DataFrame,
    seen_entities: set,
    seen_relations: set,
) -> Tuple[int, int]:
    if "triples_ints" not in df.columns:
        return 0, 0

    unseen_paths = 0
    total_paths = 0
    for raw_path in df["triples_ints"]:
        path = _normalize_path(raw_path)
        if not path:
            continue
        total_paths += 1
        entities = path[::2]
        relations = path[1::2]
        has_unseen_entity = any(ent not in seen_entities for ent in entities)
        has_unseen_relation = any(rel not in seen_relations for rel in relations)
        if has_unseen_entity or has_unseen_relation:
            unseen_paths += 1
    return unseen_paths, total_paths


def main():
    args = get_args()
    set_seeds(args.seed)
    target_split = "dev"
    stats_top_k = 20
    stats_output_dir = "stats"
    stats_output_prefix = f"{target_split}_stats"

    partitions, id2ent, id2rel = load_datasets(args)

    train_entities, train_relations = _collect_counts(partitions.train)
    seen_entity_ids = set(train_entities.keys())
    seen_relation_ids = set(train_relations.keys())
    target_df = _select_split(partitions, target_split)
    target_entities, target_relations = _collect_counts(target_df)
    unseen_paths, total_paths = _count_paths_with_unseen_items(
        target_df, seen_entity_ids, seen_relation_ids
    )

    entity_coverage = _compute_coverage(train_entities, target_entities)
    relation_coverage = _compute_coverage(train_relations, target_relations)

    logger.info(
        "Entity coverage for %s - unique %.1f%% (%d/%d), mentions %.1f%% (%d/%d)",
        target_split,
        entity_coverage["unique_seen_ratio"] * 100,
        entity_coverage["unique_seen"],
        entity_coverage["unique_total"],
        entity_coverage["mention_seen_ratio"] * 100,
        entity_coverage["mention_seen"],
        entity_coverage["mention_total"],
    )
    logger.info(
        "Relation coverage for %s - unique %.1f%% (%d/%d), mentions %.1f%% (%d/%d)",
        target_split,
        relation_coverage["unique_seen_ratio"] * 100,
        relation_coverage["unique_seen"],
        relation_coverage["unique_total"],
        relation_coverage["mention_seen_ratio"] * 100,
        relation_coverage["mention_seen"],
        relation_coverage["mention_total"],
    )
    if total_paths:
        unseen_ratio = unseen_paths / total_paths
        logger.info(
            "%d of %d %s triplets (%.1f%%) include unseen entities or relations",
            unseen_paths,
            total_paths,
            target_split,
            unseen_ratio * 100,
        )
    else:
        logger.info(
            "No valid paths found while checking for unseen entities/relations in the %s split",
            target_split,
        )

    top_k = stats_top_k
    entity_df = _build_topk_dataframe(train_entities, target_entities, id2ent, top_k)
    relation_df = _build_topk_dataframe(train_relations, target_relations, id2rel, top_k)

    output_dir = stats_output_dir
    os.makedirs(output_dir, exist_ok=True)
    output_name = f"{stats_output_prefix}_{target_split}.png"
    output_path = os.path.join(output_dir, output_name)
    _plot_frequency_overlap(entity_df, relation_df, target_split, output_path)
    logger.info("Saved coverage plot to %s", output_path)


if __name__ == "__main__":
    main()
