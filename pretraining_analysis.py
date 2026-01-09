#!/usr/bin/env python3
"""
Analysis harness to isolate why supasoft rewards fail to separate paths.

Runs targeted probes:
- Path sensitivity variance across random paths.
- Question shuffle impact.
- Train vs eval prompt mismatch.
- Contrastive gaps (perfect vs corrupted/permuted/random).
- Start-entity leakage (first-hop randomization).
"""

import argparse
import logging
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

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

DEFAULT_NUM_SAMPLES = 16
DEFAULT_NUM_RANDOM_TRIALS = 8
DEFAULT_NUM_PATH_SAMPLES = 8


@dataclass
class AnalysisArtifacts:
    hunch_llm: HunchBart
    env: ReinforcedUnsupervisedEnv
    data_partitions: DataPartitions
    tokenizer: AutoTokenizer
    bert_emb_dim: int
    max_path_len: int
    device: torch.device
    num_random_trials: int
    num_path_samples: int


def parse_args() -> argparse.Namespace:
    args = rl_alpha.get_args()
    args = overload_parse_defaults_with_yaml(args.preferred_config, args)

    args.exp_num_samples = int(os.getenv("EXP_NUM_SAMPLES", DEFAULT_NUM_SAMPLES))
    args.exp_num_random_trials = int(
        os.getenv("EXP_NUM_RANDOM_TRIALS", DEFAULT_NUM_RANDOM_TRIALS)
    )
    args.exp_num_path_samples = int(
        os.getenv("EXP_NUM_PATH_SAMPLES", DEFAULT_NUM_PATH_SAMPLES)
    )
    args.exp_split = os.getenv("EXP_SPLIT", "validation")

    return args


def build_logger() -> logging.Logger:
    return setup_logger("__PRETRAINING_ANALYSIS__")


def load_artifacts(args: argparse.Namespace, logger: logging.Logger) -> AnalysisArtifacts:
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

    return AnalysisArtifacts(
        hunch_llm=hunch_llm,
        env=env,
        data_partitions=data_partitions,
        tokenizer=gtllm_tokenizer,
        bert_emb_dim=bert_emb_dim,
        max_path_len=max_path_len,
        device=device,
        num_random_trials=args.exp_num_random_trials,
        num_path_samples=args.exp_num_path_samples,
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
    if bos is None or pad is None:
        raise ValueError("Tokenizer must define bos_token_id and pad_token_id")
    questions = [
        torch.tensor(q + [bos], dtype=torch.long, device=device)
        for q in batch["enc_questions"].tolist()
    ]
    padded = torch.nn.utils.rnn.pad_sequence(
        questions, batch_first=True, padding_value=pad
    )
    return padded


def prepare_full_qna_tokens(
    batch: pd.DataFrame, tokenizer: AutoTokenizer, device: torch.device
) -> torch.Tensor:
    bos = tokenizer.bos_token_id
    pad = tokenizer.pad_token_id
    sep = tokenizer.sep_token_id
    if bos is None or pad is None or sep is None:
        raise ValueError("Tokenizer must define bos/pad/sep token ids")
    qna = [
        torch.tensor(q + [bos] + a + [sep], dtype=torch.long, device=device)
        for q, a in zip(batch["enc_questions"].tolist(), batch["enc_answer"].tolist())
    ]
    padded = torch.nn.utils.rnn.pad_sequence(
        qna, batch_first=True, padding_value=pad
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
    """
    Takes the start entity and then just fills other relations and entities in the same number of steps as ground truth path at random.
    """
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


def build_first_hop_random_paths(
    *,
    gt_paths: torch.Tensor,
    step_counts: torch.Tensor,
    knowledge_graph: KGEModel,
) -> torch.Tensor:
    randomized = gt_paths.clone()
    rel_embed = knowledge_graph.relation_embedding
    ent_embed = knowledge_graph.entity_embedding
    num_relations = rel_embed.shape[0]
    num_entities = ent_embed.shape[0]

    for row_idx in range(gt_paths.shape[0]):
        steps = int(step_counts[row_idx].item())
        if steps <= 0:
            continue
        rel_idx = torch.randint(
            0, num_relations, (), device=gt_paths.device, dtype=torch.long
        )
        ent_idx = torch.randint(
            0, num_entities, (), device=gt_paths.device, dtype=torch.long
        )
        randomized[row_idx, 1, :] = get_embeddings_from_indices(rel_embed, rel_idx)
        randomized[row_idx, 2, :] = get_embeddings_from_indices(ent_embed, ent_idx)

    return randomized


def _reward_with_embeddings(
    hunch_llm: HunchBart,
    paths: torch.Tensor,
    answer_embeddings: torch.Tensor,
    question_tokens: torch.Tensor,
    pad_token_id: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return calculate_llm_reward_supasoft(
        hunch_llm,
        paths,
        answer_embeddings,
        question_tokens,
        pad_token_id,
    )


def summarize_delta(name: str, deltas: torch.Tensor, logger: logging.Logger) -> None:
    logger.info(
        "%s | mean: %.4f | std: %.4f | min: %.4f | max: %.4f | pos_rate: %.2f",
        name,
        deltas.mean().item(),
        deltas.std().item() if deltas.numel() > 1 else 0.0,
        deltas.min().item(),
        deltas.max().item(),
        (deltas > 0).float().mean().item(),
    )


def path_variance_test(
    artifacts: AnalysisArtifacts,
    batch: pd.DataFrame,
    question_tokens: torch.Tensor,
    answer_embeddings: torch.Tensor,
    logger: logging.Logger,
) -> None:
    gt_paths, step_counts = get_ground_truth_paths(
        mini_batch=batch,
        env=artifacts.env,
        max_path_len=artifacts.max_path_len,
        device=artifacts.device,
    )
    state_dim = gt_paths.shape[-1]
    start_entity_ids = _start_entity_ids(batch, artifacts.device)

    reward_trials: List[torch.Tensor] = []
    embedding_trials: List[torch.Tensor] = []
    for _ in range(max(1, artifacts.num_path_samples)):
        random_paths = build_random_paths(
            batch_size=len(batch),
            max_path_len=artifacts.max_path_len,
            state_dim=state_dim,
            step_counts=step_counts,
            knowledge_graph=artifacts.env.knowledge_graph,
            start_entity_ids=start_entity_ids,
            device=artifacts.device,
        )
        reward, embeds = _reward_with_embeddings(
            artifacts.hunch_llm,
            random_paths,
            answer_embeddings,
            question_tokens,
            artifacts.tokenizer.pad_token_id,
        )
        reward_trials.append(reward)
        embedding_trials.append(embeds)

    reward_stack = torch.stack(reward_trials, dim=0)  # [T, B]
    embed_stack = torch.stack(embedding_trials, dim=0)  # [T, B, D]

    reward_var = reward_stack.var(dim=0).mean().item()
    embed_var = embed_stack.var(dim=0).mean(dim=-1).mean().item()

    logger.info(
        "Path sensitivity variance | reward_var_mean: %.6f | embed_var_mean: %.6f",
        reward_var,
        embed_var,
    )


def question_shuffle_test(
    artifacts: AnalysisArtifacts,
    batch: pd.DataFrame,
    question_tokens: torch.Tensor,
    answer_embeddings: torch.Tensor,
    logger: logging.Logger,
) -> None:
    gt_paths, _ = get_ground_truth_paths(
        mini_batch=batch,
        env=artifacts.env,
        max_path_len=artifacts.max_path_len,
        device=artifacts.device,
    )

    base_reward, _ = _reward_with_embeddings(
        artifacts.hunch_llm,
        gt_paths,
        answer_embeddings,
        question_tokens,
        artifacts.tokenizer.pad_token_id,
    )

    perm = torch.randperm(question_tokens.shape[0], device=artifacts.device)
    shuffled_questions = question_tokens[perm]
    shuffled_reward, _ = _reward_with_embeddings(
        artifacts.hunch_llm,
        gt_paths,
        answer_embeddings,
        shuffled_questions,
        artifacts.tokenizer.pad_token_id,
    )

    diff = (base_reward - shuffled_reward).abs()
    corr = torch.corrcoef(torch.stack([base_reward, shuffled_reward]))[0, 1].item()

    logger.info(
        "Question shuffle | abs_delta_mean: %.4f | abs_delta_std: %.4f | corr: %.4f",
        diff.mean().item(),
        diff.std().item() if diff.numel() > 1 else 0.0,
        corr,
    )


def train_eval_mismatch_test(
    artifacts: AnalysisArtifacts,
    batch: pd.DataFrame,
    question_tokens: torch.Tensor,
    answer_embeddings: torch.Tensor,
    logger: logging.Logger,
) -> None:
    full_qna_tokens = prepare_full_qna_tokens(
        batch, artifacts.tokenizer, artifacts.device
    )

    gt_paths, step_counts = get_ground_truth_paths(
        mini_batch=batch,
        env=artifacts.env,
        max_path_len=artifacts.max_path_len,
        device=artifacts.device,
    )
    state_dim = gt_paths.shape[-1]
    start_entity_ids = _start_entity_ids(batch, artifacts.device)
    random_paths = build_random_paths(
        batch_size=len(batch),
        max_path_len=artifacts.max_path_len,
        state_dim=state_dim,
        step_counts=step_counts,
        knowledge_graph=artifacts.env.knowledge_graph,
        start_entity_ids=start_entity_ids,
        device=artifacts.device,
    )

    perfect_q, _ = _reward_with_embeddings(
        artifacts.hunch_llm,
        gt_paths,
        answer_embeddings,
        question_tokens,
        artifacts.tokenizer.pad_token_id,
    )
    perfect_full, _ = _reward_with_embeddings(
        artifacts.hunch_llm,
        gt_paths,
        answer_embeddings,
        full_qna_tokens,
        artifacts.tokenizer.pad_token_id,
    )
    random_q, _ = _reward_with_embeddings(
        artifacts.hunch_llm,
        random_paths,
        answer_embeddings,
        question_tokens,
        artifacts.tokenizer.pad_token_id,
    )
    random_full, _ = _reward_with_embeddings(
        artifacts.hunch_llm,
        random_paths,
        answer_embeddings,
        full_qna_tokens,
        artifacts.tokenizer.pad_token_id,
    )

    sep_q = (perfect_q - random_q).mean().item()
    sep_full = (perfect_full - random_full).mean().item()
    diff = (perfect_q - perfect_full).abs().mean().item()

    logger.info(
        "Train-eval mismatch | sep_question_only: %.4f | sep_full_qna: %.4f | perfect_abs_delta: %.4f",
        sep_q,
        sep_full,
        diff,
    )


def contrastive_sensitivity_test(
    artifacts: AnalysisArtifacts,
    batch: pd.DataFrame,
    question_tokens: torch.Tensor,
    answer_embeddings: torch.Tensor,
    logger: logging.Logger,
) -> None:
    gt_paths, step_counts = get_ground_truth_paths(
        mini_batch=batch,
        env=artifacts.env,
        max_path_len=artifacts.max_path_len,
        device=artifacts.device,
    )
    state_dim = gt_paths.shape[-1]
    start_entity_ids = _start_entity_ids(batch, artifacts.device)

    random_paths = build_random_paths(
        batch_size=len(batch),
        max_path_len=artifacts.max_path_len,
        state_dim=state_dim,
        step_counts=step_counts,
        knowledge_graph=artifacts.env.knowledge_graph,
        start_entity_ids=start_entity_ids,
        device=artifacts.device,
    )
    corrupted_paths = build_corrupted_paths(
        gt_paths=gt_paths,
        step_counts=step_counts,
        knowledge_graph=artifacts.env.knowledge_graph,
    )
    permuted_paths = torch.roll(gt_paths, shifts=1, dims=0)
    start_only_paths = build_truncated_paths(gt_paths)

    perfect, _ = _reward_with_embeddings(
        artifacts.hunch_llm,
        gt_paths,
        answer_embeddings,
        question_tokens,
        artifacts.tokenizer.pad_token_id,
    )
    random_reward, _ = _reward_with_embeddings(
        artifacts.hunch_llm,
        random_paths,
        answer_embeddings,
        question_tokens,
        artifacts.tokenizer.pad_token_id,
    )
    corrupted, _ = _reward_with_embeddings(
        artifacts.hunch_llm,
        corrupted_paths,
        answer_embeddings,
        question_tokens,
        artifacts.tokenizer.pad_token_id,
    )
    permuted, _ = _reward_with_embeddings(
        artifacts.hunch_llm,
        permuted_paths,
        answer_embeddings,
        question_tokens,
        artifacts.tokenizer.pad_token_id,
    )
    start_only, _ = _reward_with_embeddings(
        artifacts.hunch_llm,
        start_only_paths,
        answer_embeddings,
        question_tokens,
        artifacts.tokenizer.pad_token_id,
    )

    summarize_delta("Perfect - Random", perfect - random_reward, logger)
    summarize_delta("Perfect - Corrupted", perfect - corrupted, logger)
    summarize_delta("Perfect - Permuted", perfect - permuted, logger)
    summarize_delta("Perfect - StartOnly", perfect - start_only, logger)


def start_entity_leakage_test(
    artifacts: AnalysisArtifacts,
    batch: pd.DataFrame,
    question_tokens: torch.Tensor,
    answer_embeddings: torch.Tensor,
    logger: logging.Logger,
) -> None:
    gt_paths, step_counts = get_ground_truth_paths(
        mini_batch=batch,
        env=artifacts.env,
        max_path_len=artifacts.max_path_len,
        device=artifacts.device,
    )
    state_dim = gt_paths.shape[-1]
    start_entity_ids = _start_entity_ids(batch, artifacts.device)

    random_paths = build_random_paths(
        batch_size=len(batch),
        max_path_len=artifacts.max_path_len,
        state_dim=state_dim,
        step_counts=step_counts,
        knowledge_graph=artifacts.env.knowledge_graph,
        start_entity_ids=start_entity_ids,
        device=artifacts.device,
    )
    first_hop_random = build_first_hop_random_paths(
        gt_paths=gt_paths,
        step_counts=step_counts,
        knowledge_graph=artifacts.env.knowledge_graph,
    )

    perfect, _ = _reward_with_embeddings(
        artifacts.hunch_llm,
        gt_paths,
        answer_embeddings,
        question_tokens,
        artifacts.tokenizer.pad_token_id,
    )
    full_random, _ = _reward_with_embeddings(
        artifacts.hunch_llm,
        random_paths,
        answer_embeddings,
        question_tokens,
        artifacts.tokenizer.pad_token_id,
    )
    first_hop, _ = _reward_with_embeddings(
        artifacts.hunch_llm,
        first_hop_random,
        answer_embeddings,
        question_tokens,
        artifacts.tokenizer.pad_token_id,
    )

    summarize_delta("Perfect - FirstHopRandom", perfect - first_hop, logger)
    summarize_delta("Perfect - FullRandom", perfect - full_random, logger)


def path_mask_test(
    artifacts: AnalysisArtifacts,
    batch: pd.DataFrame,
    question_tokens: torch.Tensor,
    answer_embeddings: torch.Tensor,
    logger: logging.Logger,
) -> None:
    gt_paths, _ = get_ground_truth_paths(
        mini_batch=batch,
        env=artifacts.env,
        max_path_len=artifacts.max_path_len,
        device=artifacts.device,
    )
    masked_paths = torch.full_like(gt_paths, BART_PADDING_VALUE)

    perfect, _ = _reward_with_embeddings(
        artifacts.hunch_llm,
        gt_paths,
        answer_embeddings,
        question_tokens,
        artifacts.tokenizer.pad_token_id,
    )
    masked, _ = _reward_with_embeddings(
        artifacts.hunch_llm,
        masked_paths,
        answer_embeddings,
        question_tokens,
        artifacts.tokenizer.pad_token_id,
    )

    summarize_delta("Perfect - MaskedPath", perfect - masked, logger)


def answer_shuffle_test(
    artifacts: AnalysisArtifacts,
    batch: pd.DataFrame,
    question_tokens: torch.Tensor,
    answer_embeddings: torch.Tensor,
    logger: logging.Logger,
) -> None:
    gt_paths, _ = get_ground_truth_paths(
        mini_batch=batch,
        env=artifacts.env,
        max_path_len=artifacts.max_path_len,
        device=artifacts.device,
    )
    perm = torch.randperm(answer_embeddings.shape[0], device=artifacts.device)
    shuffled_answers = answer_embeddings[perm]

    base_reward, _ = _reward_with_embeddings(
        artifacts.hunch_llm,
        gt_paths,
        answer_embeddings,
        question_tokens,
        artifacts.tokenizer.pad_token_id,
    )
    shuffled_reward, _ = _reward_with_embeddings(
        artifacts.hunch_llm,
        gt_paths,
        shuffled_answers,
        question_tokens,
        artifacts.tokenizer.pad_token_id,
    )

    diff = (base_reward - shuffled_reward).abs()
    corr = torch.corrcoef(torch.stack([base_reward, shuffled_reward]))[0, 1].item()

    logger.info(
        "Answer shuffle | abs_delta_mean: %.4f | abs_delta_std: %.4f | corr: %.4f",
        diff.mean().item(),
        diff.std().item() if diff.numel() > 1 else 0.0,
        corr,
    )


def margin_probe(
    artifacts: AnalysisArtifacts,
    batch: pd.DataFrame,
    question_tokens: torch.Tensor,
    answer_embeddings: torch.Tensor,
    logger: logging.Logger,
    margin: float = 0.05,
) -> None:
    gt_paths, step_counts = get_ground_truth_paths(
        mini_batch=batch,
        env=artifacts.env,
        max_path_len=artifacts.max_path_len,
        device=artifacts.device,
    )
    state_dim = gt_paths.shape[-1]
    start_entity_ids = _start_entity_ids(batch, artifacts.device)

    random_paths = build_random_paths(
        batch_size=len(batch),
        max_path_len=artifacts.max_path_len,
        state_dim=state_dim,
        step_counts=step_counts,
        knowledge_graph=artifacts.env.knowledge_graph,
        start_entity_ids=start_entity_ids,
        device=artifacts.device,
    )
    corrupted_paths = build_corrupted_paths(
        gt_paths=gt_paths,
        step_counts=step_counts,
        knowledge_graph=artifacts.env.knowledge_graph,
    )

    perfect, _ = _reward_with_embeddings(
        artifacts.hunch_llm,
        gt_paths,
        answer_embeddings,
        question_tokens,
        artifacts.tokenizer.pad_token_id,
    )
    random_reward, _ = _reward_with_embeddings(
        artifacts.hunch_llm,
        random_paths,
        answer_embeddings,
        question_tokens,
        artifacts.tokenizer.pad_token_id,
    )
    corrupted, _ = _reward_with_embeddings(
        artifacts.hunch_llm,
        corrupted_paths,
        answer_embeddings,
        question_tokens,
        artifacts.tokenizer.pad_token_id,
    )

    delta_random = perfect - random_reward
    delta_corrupted = perfect - corrupted
    frac_random = (delta_random > margin).float().mean().item()
    frac_corrupted = (delta_corrupted > margin).float().mean().item()

    logger.info(
        "Margin probe (margin=%.3f) | perfect>random: %.2f | perfect>corrupted: %.2f",
        margin,
        frac_random,
        frac_corrupted,
    )


def main() -> None:
    args = parse_args()
    logger = build_logger()
    artifacts = load_artifacts(args, logger)

    dataset, split = _select_split(artifacts.data_partitions, args.exp_split)
    sample_size = min(args.exp_num_samples, len(dataset))
    batch = dataset.sample(n=sample_size, random_state=args.seed)
    batch = batch.reset_index(drop=True)

    logger.info(
        "Running analysis on %s split with %d samples (pad=%s, bart_pad=%s)",
        split,
        sample_size,
        PATH_PADDING_VALUE,
        BART_PADDING_VALUE,
    )

    question_tokens = prepare_question_tokens(
        batch, artifacts.tokenizer, artifacts.device
    )
    answer_embeddings = extract_answer_embeddings(
        batch, artifacts.bert_emb_dim, artifacts.device
    )

    with torch.no_grad():
        path_variance_test(artifacts, batch, question_tokens, answer_embeddings, logger)
        question_shuffle_test(artifacts, batch, question_tokens, answer_embeddings, logger)
        train_eval_mismatch_test(artifacts, batch, question_tokens, answer_embeddings, logger)
        contrastive_sensitivity_test(artifacts, batch, question_tokens, answer_embeddings, logger)
        start_entity_leakage_test(artifacts, batch, question_tokens, answer_embeddings, logger)
        path_mask_test(artifacts, batch, question_tokens, answer_embeddings, logger)
        answer_shuffle_test(artifacts, batch, question_tokens, answer_embeddings, logger)
        margin_probe(artifacts, batch, question_tokens, answer_embeddings, logger)


if __name__ == "__main__":
    main()
