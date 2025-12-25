#!/usr/bin/env python3
"""
Small experiment harness for teacher-forcing RL work.

Runs targeted probes against ``calculate_llm_reward_supasoft`` such as
perfect-path vs random-path reward gaps.
"""

import argparse
import logging
import os
from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
import torch
from rich import traceback
from transformers import AutoModel, AutoTokenizer

import multihopkg.data_utils as data_utils
from multihopkg.exogenous.sun_models import KGEModel, get_embeddings_from_indices
from multihopkg.logging import setup_logger
from multihopkg.models_language.classical import HunchBart
from multihopkg.rl.graph_search.pn import ReinforcedUnsupervisedEnv
from multihopkg.run_configs import rl_alpha
from multihopkg.run_configs.common import overload_parse_defaults_with_yaml
from multihopkg.utils.data_structures import DataPartitions
from multihopkg.utils.setup import set_seeds
from rl_trainint_teacherforce import (
    BART_PADDING_VALUE,
    PATH_PADDING_VALUE,
    calculate_llm_reward_supasoft,
    get_ground_truth_paths,
)

traceback.install()

DEFAULT_NUM_SAMPLES = 8
DEFAULT_NUM_RANDOM_TRIALS = 1


@dataclass
class ExperimentArtifacts:
    hunch_llm: HunchBart
    env: ReinforcedUnsupervisedEnv
    data_partitions: DataPartitions
    tokenizer: AutoTokenizer
    bert_emb_dim: int
    max_path_len: int
    device: torch.device
    num_random_trials: int


def parse_args() -> argparse.Namespace:
    """Load base RL config plus lightweight experiment defaults."""

    args = rl_alpha.get_args()
    args = overload_parse_defaults_with_yaml(args.preferred_config, args)

    # Experiment-only knobs (env vars to avoid touching the shared parser)
    args.exp_num_samples = int(os.getenv("EXP_NUM_SAMPLES", DEFAULT_NUM_SAMPLES))
    args.exp_num_random_trials = int(
        os.getenv("EXP_NUM_RANDOM_TRIALS", DEFAULT_NUM_RANDOM_TRIALS)
    )
    args.exp_split = os.getenv("EXP_SPLIT", "validation")

    return args


def build_logger() -> logging.Logger:
    logger = setup_logger("__RL_EXP__")
    return logger


def load_artifacts(args: argparse.Namespace, logger: logging.Logger) -> ExperimentArtifacts:
    """Load models, data, and environment mirrors of the training stack."""

    set_seeds(args.seed)
    device = torch.device(args.device)

    if device.type == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available; falling back to CPU.")
        device = torch.device("cpu")

    pretrained_gtllm_metadata = torch.load(
        args.pretrained_gtllm_path, weights_only=False
    )

    gtllm_hunch_base_model = pretrained_gtllm_metadata["hunchbart_base_llm_model"]
    gtllm_graph_embedding_dim = pretrained_gtllm_metadata["hunchbart_hidden_dim"]
    gtllm_tokenizer_name = pretrained_gtllm_metadata["hunchbart_base_llm_tokenizer"]
    gtllm_tokenizer = AutoTokenizer.from_pretrained(gtllm_tokenizer_name)

    hunch_llm = HunchBart.from_pretrained(
        hunchbart_base_llm_model_name=gtllm_hunch_base_model,
        state_dict=pretrained_gtllm_metadata["gtllm_state_dict"],
        graph_embedding_dim=gtllm_graph_embedding_dim,
        tokenizer=gtllm_tokenizer,
    ).to(device)
    hunch_llm.eval()

    ge_params = pretrained_gtllm_metadata["embedding_training_metaparam"]
    ge_geom = ge_params["model"]
    ge_gamma = ge_params["gamma"]
    ge_save_path = pretrained_gtllm_metadata["path_graph_emb_data"]

    entity_embeddings = np.load(os.path.join(ge_save_path, "entity_embedding.npy"))
    relation_embeddings = np.load(os.path.join(ge_save_path, "relation_embedding.npy"))
    checkpoint = torch.load(os.path.join(ge_save_path, "checkpoint"))

    kge_model = KGEModel.from_pretrained(
        model_name=ge_geom,
        entity_embedding=entity_embeddings,
        relation_embedding=relation_embeddings,
        gamma=ge_gamma,
        state_dict=checkpoint["model_state_dict"],
    ).to(device)
    kge_model.recalculate_entity_centroid()

    train_df, dev_df, test_df, metadata = data_utils.load_cached_pretraining_data(
        args.pretraining_metadata_cache_path
    )
    bert_emb_dim = metadata["ques_cols"][1] - metadata["ques_cols"][0]
    data_partitions = DataPartitions(train_df, dev_df, test_df)

    question_embedding_module = AutoModel.from_pretrained(
        gtllm_hunch_base_model
    ).to(device)
    question_embedding_module.eval()
    env = ReinforcedUnsupervisedEnv(
        bert_question_embedding_module=question_embedding_module,
        knowledge_graph=kge_model,
        nav_start_emb_type=args.nav_start_emb_type,
        reached_destination_threshold=args.reached_destination_threshold,
    )

    max_path_len = args.max_env_steps * 2 + 1

    logger.info(
        "Artifacts ready | KG model: %s | HunchBart: %s | samples per run: %s",
        ge_geom,
        gtllm_hunch_base_model,
        args.exp_num_samples,
    )

    return ExperimentArtifacts(
        hunch_llm=hunch_llm,
        env=env,
        data_partitions=data_partitions,
        tokenizer=gtllm_tokenizer,
        bert_emb_dim=bert_emb_dim,
        max_path_len=max_path_len,
        device=device,
        num_random_trials=args.exp_num_random_trials,
    )


def _select_split(
    data_partitions: DataPartitions, split: str
) -> Tuple[pd.DataFrame, str]:
    split = split.lower()
    if split in ("validation", "dev", "val"):
        return data_partitions.validation, "validation"
    if split == "train":
        return data_partitions.train, "train"
    if split == "test":
        return data_partitions.test, "test"
    raise ValueError(f"Unsupported split '{split}'")


def prepare_question_tokens(
    batch: pd.DataFrame, tokenizer: AutoTokenizer, device: torch.device
) -> torch.Tensor:
    bos = tokenizer.bos_token_id
    pad = tokenizer.pad_token_id
    questions = [
        torch.tensor(q + [bos], dtype=torch.long, device=device)
        for q in batch["enc_questions"].tolist()
    ]
    padded = torch.nn.utils.rnn.pad_sequence(
        questions, batch_first=True, padding_value=pad
    )
    return padded


def extract_answer_embeddings(
    batch: pd.DataFrame, bert_dim: int, device: torch.device
) -> torch.Tensor:
    ans_cols = batch.iloc[:, 3 : 3 + bert_dim].values
    return torch.as_tensor(ans_cols, dtype=torch.float32, device=device)


def _start_entity_ids(batch: pd.DataFrame, device: torch.device) -> torch.Tensor:
    start_ids = [path[0] for path in batch["triples_ints"].tolist()]
    return torch.tensor(start_ids, dtype=torch.long, device=device)


def build_random_paths(
    *,
    batch_size: int,
    max_path_len: int,
    state_dim: int,
    step_counts: torch.Tensor,
    knowledge_graph: KGEModel,
    start_entity_ids: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Construct random paths with matching hop counts."""

    ent_embed = knowledge_graph.entity_embedding
    rel_embed = knowledge_graph.relation_embedding

    num_entities = ent_embed.shape[0]
    num_relations = rel_embed.shape[0]

    random_paths = torch.full(
        (batch_size, max_path_len, state_dim),
        PATH_PADDING_VALUE,
        device=device,
        dtype=ent_embed.dtype,
    )
    random_paths[:, 0, :] = get_embeddings_from_indices(ent_embed, start_entity_ids)

    for row_idx in range(batch_size):
        steps = int(step_counts[row_idx].item())
        if steps <= 0:
            continue

        rel_ids = torch.randint(
            0, num_relations, (steps,), device=device, dtype=torch.long
        )
        ent_ids = torch.randint(
            0, num_entities, (steps,), device=device, dtype=torch.long
        )
        random_paths[row_idx, 1 : 2 * steps + 1 : 2, :] = get_embeddings_from_indices(
            rel_embed, rel_ids
        )
        random_paths[row_idx, 2 : 2 * steps + 2 : 2, :] = get_embeddings_from_indices(
            ent_embed, ent_ids
        )

    return random_paths


def build_corrupted_paths(
    *,
    gt_paths: torch.Tensor,
    step_counts: torch.Tensor,
    knowledge_graph: KGEModel,
) -> torch.Tensor:
    """Corrupt a single hop (relation + following entity) per path."""

    corrupted = gt_paths.clone()
    rel_embed = knowledge_graph.relation_embedding
    ent_embed = knowledge_graph.entity_embedding
    num_relations = rel_embed.shape[0]
    num_entities = ent_embed.shape[0]

    for row_idx in range(gt_paths.shape[0]):
        steps = int(step_counts[row_idx].item())
        if steps <= 0:
            continue
        hop = torch.randint(0, steps, ()).item()
        rel_idx = torch.randint(
            0, num_relations, (), device=gt_paths.device, dtype=torch.long
        )
        ent_idx = torch.randint(
            0, num_entities, (), device=gt_paths.device, dtype=torch.long
        )
        corrupted[row_idx, 2 * hop + 1, :] = get_embeddings_from_indices(
            rel_embed, rel_idx
        )
        corrupted[row_idx, 2 * hop + 2, :] = get_embeddings_from_indices(
            ent_embed, ent_idx
        )

    return corrupted


def build_truncated_paths(gt_paths: torch.Tensor) -> torch.Tensor:
    truncated = torch.full_like(gt_paths, PATH_PADDING_VALUE)
    truncated[:, 0, :] = gt_paths[:, 0, :]
    return truncated


def reward_stats(values: torch.Tensor) -> Dict[str, float]:
    if values.numel() == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": values.mean().item(),
        "std": values.std().item() if values.numel() > 1 else 0.0,
        "min": values.min().item(),
        "max": values.max().item(),
    }


def run_reward_probes(
    artifacts: ExperimentArtifacts,
    batch: pd.DataFrame,
    logger: logging.Logger,
) -> Dict[str, torch.Tensor]:
    device = artifacts.device
    question_tokens = prepare_question_tokens(batch, artifacts.tokenizer, device)
    answer_embeddings = extract_answer_embeddings(
        batch, artifacts.bert_emb_dim, device
    )

    gt_paths, step_counts = get_ground_truth_paths(
        mini_batch=batch,
        env=artifacts.env,
        max_path_len=artifacts.max_path_len,
        device=device,
    )
    state_dim = gt_paths.shape[-1]
    start_entity_ids = _start_entity_ids(batch, device)

    rewards: Dict[str, torch.Tensor] = {}

    rewards["perfect"], _ = calculate_llm_reward_supasoft(
        artifacts.hunch_llm,
        gt_paths,
        answer_embeddings,
        question_tokens,
        artifacts.tokenizer.pad_token_id,
    )

    random_trials = []
    for _ in range(max(1, artifacts.num_random_trials)):
        random_paths = build_random_paths(
            batch_size=len(batch),
            max_path_len=artifacts.max_path_len,
            state_dim=state_dim,
            step_counts=step_counts,
            knowledge_graph=artifacts.env.knowledge_graph,
            start_entity_ids=start_entity_ids,
            device=device,
        )
        trial_reward, _ = calculate_llm_reward_supasoft(
            artifacts.hunch_llm,
            random_paths,
            answer_embeddings,
            question_tokens,
            artifacts.tokenizer.pad_token_id,
        )
        random_trials.append(trial_reward)
    rewards["random"] = (
        torch.stack(random_trials, dim=0).mean(dim=0) if len(random_trials) > 1 else random_trials[0]
    )

    corrupted_paths = build_corrupted_paths(
        gt_paths=gt_paths,
        step_counts=step_counts,
        knowledge_graph=artifacts.env.knowledge_graph,
    )
    rewards["single_hop_corrupted"], _ = calculate_llm_reward_supasoft(
        artifacts.hunch_llm,
        corrupted_paths,
        answer_embeddings,
        question_tokens,
        artifacts.tokenizer.pad_token_id,
    )

    truncated_paths = build_truncated_paths(gt_paths)
    rewards["start_only"], _ = calculate_llm_reward_supasoft(
        artifacts.hunch_llm,
        truncated_paths,
        answer_embeddings,
        question_tokens,
        artifacts.tokenizer.pad_token_id,
    )

    rewards["perfect_minus_random"] = rewards["perfect"] - rewards["random"]
    rewards["perfect_minus_corrupted"] = (
        rewards["perfect"] - rewards["single_hop_corrupted"]
    )
    rewards["perfect_minus_start_only"] = rewards["perfect"] - rewards["start_only"]

    logger.info(
        "Computed rewards | perfect mean %.4f | random mean %.4f",
        rewards["perfect"].mean().item(),
        rewards["random"].mean().item(),
    )

    return rewards


def log_reward_table(
    rewards: Dict[str, torch.Tensor],
    sample_ids: Iterable[int],
    logger: logging.Logger,
    limit: int = 5,
) -> None:
    """Log a small per-sample table for quick inspection."""

    ids = list(sample_ids)
    rows = []
    for idx in range(min(limit, len(ids))):
        rows.append(
            {
                "id": ids[idx],
                "perfect": rewards["perfect"][idx].item(),
                "random": rewards["random"][idx].item(),
                "corrupted": rewards["single_hop_corrupted"][idx].item(),
                "start_only": rewards["start_only"][idx].item(),
            }
        )
    if not rows:
        return
    df = pd.DataFrame(rows)
    logger.info("Sample reward slices:\n%s", df.to_string(index=False))


def main() -> None:
    args = parse_args()
    logger = build_logger()
    artifacts = load_artifacts(args, logger)

    dataset, split = _select_split(artifacts.data_partitions, args.exp_split)
    sample_size = min(args.exp_num_samples, len(dataset))
    batch = dataset.sample(n=sample_size, random_state=args.seed)
    sample_ids = batch.index.tolist()
    batch = batch.reset_index(drop=True)

    logger.info(
        "Running reward probes on %s split with %d samples (pad=%s, bart_pad=%s)",
        split,
        sample_size,
        PATH_PADDING_VALUE,
        BART_PADDING_VALUE,
    )

    with torch.no_grad():
        rewards = run_reward_probes(artifacts, batch, logger)

    summary = {
        name: reward_stats(tensor)
        for name, tensor in rewards.items()
        if tensor is not None
    }

    for name, stats in summary.items():
        logger.info(
            "%s -> mean: %.4f | std: %.4f | min: %.4f | max: %.4f",
            name,
            stats["mean"],
            stats["std"],
            stats["min"],
            stats["max"],
        )

    log_reward_table(rewards, sample_ids, logger)


if __name__ == "__main__":
    main()
