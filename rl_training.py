#!/usr/bin/env python3

"""
Copyright (c) 2018, salesforce.com, inc.
For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

Experiment Portal.
"""

import argparse
import ast
import copy
import io
import json
import logging
from ntpath import exists
import os
import pickle
import random
import sys
import time
import math
from collections import Counter, defaultdict
from typing import Any, DefaultDict, Dict, List, Optional, Sequence, Tuple

import debugpy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from transformers.models.bart.modeling_bart import BART_GENERATION_EXAMPLE
from multihopkg.datasets import GraphEmbeddingDataset
from multihopkg.utils.data_structures import DataPartitions
from multihopkg.utils.ops import ensure_list_of_ints
import torch
import torch.nn.functional as F
from PIL import Image
from rich import traceback

# PCA
from sklearn.decomposition import PCA
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import (
    AutoModel,
    AutoTokenizer,
    BartConfig,
    BertModel,
    PreTrainedTokenizer,
)

import multihopkg.data_utils as data_utils
import multihopkg.utils_debug.distribution_tracker as dist_tracker
import wandb
from multihopkg.environments import Observation
from multihopkg.exogenous.sun_models import KGEModel, get_embeddings_from_indices
from multihopkg.logging import setup_logger
from multihopkg.logs import torch_module_logging
from multihopkg.models_language.classical import HunchBart, collate_token_ids_batch
from multihopkg.rl.graph_search.cpg import ContinuousPolicyGradient
from multihopkg.rl.graph_search.sac import CriticQ, CriticV, GraphCriticQ, GraphCriticV
from multihopkg.rl.graph_search.pn import ITLGraphEnvironment, ReinforcedUnsupervisedEnv
from multihopkg.rl.utils import QuestionReplayBuffer
from multihopkg.run_configs import rl_alpha
from multihopkg.run_configs.common import overload_parse_defaults_with_yaml
from multihopkg.utils.convenience import tensor_normalization
from multihopkg.utils.setup import set_seeds
from multihopkg.utils.wandb import histogram_all_modules
from multihopkg.utils_debug.dump_evals import dump_evaluation_metrics
from multihopkg.vector_search import ANN_IndexMan, ANN_IndexMan_pRotatE

traceback.install()
wandb_run = None

PATH_PADDING_VALUE = 1
BART_PADDING_VALUE = 1 # TODO:  Need to remove hardcoding on this later


def initialize_model_directory(args, random_seed=None):
    # add model parameter info to model directory
    # TODO: We might2ant our implementation of something like this later
    raise NotImplementedError


def initial_setup() -> (
    Tuple[argparse.Namespace, logging.Logger]
):
    global logger
    args = rl_alpha.get_args()
    args = overload_parse_defaults_with_yaml(args.preferred_config, args)

    set_seeds(args.seed)
    logger = setup_logger("__MLM__")

    assert isinstance(args, argparse.Namespace)

    return args, logger



def batch_loop_dev(
    env: ITLGraphEnvironment,
    mini_batch: pd.DataFrame,  # Perhaps change this ?
    nav_agent: ContinuousPolicyGradient,
    hunch_llm: nn.Module,
    steps_in_episode: int,
    pad_token_id: int,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Executes a batch loop for the development set to compute additional evaluation metrics.
    This function is similar to `batch_loop` but focuses on collecting metrics for debugging
    and analysis during development.

    During the batch loop:
    - The navigation agent (`nav_agent`) interacts with the environment (`env`) to take actions inside `rollout`.
    - Rewards are computed from both the language model (`hunch_llm`) and the knowledge graph environment (KGE).
    - Evaluation metrics are collected for analysis.

    This function is found within `evaluate_training` calls upon `rollout`.

    Args:
        env (ITLGraphEnvironment):
            The knowledge graph environment that provides observations, rewards, and state transitions.
        mini_batch (pd.DataFrame):
            A batch of data containing questions, answers, relevant entities, and relations.
        nav_agent (ContinuousPolicyGradient):
            The policy network responsible for deciding actions based on the current state.
        hunch_llm (nn.Module):
            A language model used to compute rewards based on how well the agent's state aligns with the expected answers.
        steps_in_episode (int):
            The number of steps to execute in each episode.
        pad_token_id (int):
            The token ID used for padding sequences in the answer IDs.

    Returns:
        - `pg_loss` (torch.Tensor):
            The policy gradient loss computed for the batch.
        - `eval_extras` (Dict[str, Any]):
            A dictionary containing additional evaluation metrics collected during the batch loop.


    Notes:
        - This function is specifically designed for development and debugging purposes.
        - Rewards are normalized for stability before being used to compute the policy gradient loss.
    """

    ########################################
    # Start the batch loop with zero grad
    ########################################
    nav_agent.zero_grad()
    device = next(nav_agent.parameters()).device

    # Deconstruct the batch
    questions = mini_batch["Question"].tolist()
    answers = mini_batch["Answer"].tolist()
    query_ent = mini_batch["Query-Entity"].tolist()
    query_rel = mini_batch["Query-Relation"].tolist()
    answer_id = mini_batch["Answer-Entity"].tolist()
    # question_embeddings = env.get_llm_embeddings(questions, device)
    if env.use_kge_question_embedding:
        question_embeddings = env.get_kge_question_embedding(
            query_ent, query_rel, device
        )  # Shape: (batch, 2*embedding_dim)
    else:
        question_embeddings = env.get_llm_embeddings(questions, device)

    answer_ids_padded_tensor = (
        collate_token_ids_batch(answers, pad_token_id).to(torch.int32).to(device)
    )
    pad_mask = answer_ids_padded_tensor.ne(pad_token_id)


    raise NotImplementedError("Removed Rollout here, Ought to be replaced for something else now")
    #TODO: Remove. We dont do rollouts on SAC training
    # log_probs, entropies, llm_rewards, kg_rewards, eval_extras = rollout(
    #     steps_in_episode,
    #     nav_agent,
    #     hunch_llm,
    #     env,
    #     question_embeddings,
    #     answer_ids_padded_tensor,
    #     query_ent=query_ent,
    #     query_rel=query_rel,
    #     answer_id=answer_id,
    #     dev_mode=True,
    # )

    ########################################
    # Calculate Reinforce Objective
    ########################################
    "LLM Rewards"

    llm_rewards_t = (torch.stack(llm_rewards)).permute(1, 0, 2)

    assert not torch.isnan(
        llm_rewards_t
    ).any(), "NaN detected in the llm rewards (batch_loop_dev). Aborting training."

    # Get only masked, then mean
    llm_rewards_t_unpacked = []
    for i, reward_batch_element in enumerate(llm_rewards_t):
        mask_for_element = pad_mask[i][1:].unsqueeze(0).repeat(steps_in_episode, 1)
        filtered_rewards = reward_batch_element[mask_for_element].reshape(
            steps_in_episode, -1
        )
        mean_reward = torch.mean(filtered_rewards, dim=-1)
        llm_rewards_t_unpacked.append(mean_reward)
    llm_rewards_t = torch.stack(llm_rewards_t_unpacked)

    log_probs_t = torch.stack(log_probs).T
    entropies_t = torch.stack(entropies).T
    num_steps = log_probs_t.shape[-1]

    assert not torch.isnan(
        log_probs_t
    ).any(), "NaN detected in the log probs (batch_loop_dev). Aborting training."

    # TODO: Check if this is not bad.
    llm_rewards_t = llm_rewards_t.expand_as(
        log_probs_t
    )  # TOREM: This is a hack to make the shapes match
    # -------------------------------------------------------------------------
    "Knowledge Graph Environment Rewards"

    kg_rewards_t = (torch.stack(kg_rewards)).permute(
        1, 0, 2
    )  # Correcting to Shape: (batch_size, num_steps, reward_type)
    kg_rewards_t = kg_rewards_t.squeeze(2)  # Shape: (batch_size, num_steps)

    assert not torch.isnan(
        kg_rewards_t
    ).any(), "NaN detected in the kg rewards (batch_loop_dev). Aborting training."

    # -------------------------------------------------------------------------
    "Discount and Merging of Rewards"

    # TODO: Check if a weight is needed for combining the rewards
    gamma = nav_agent.gamma
    discounted_rewards = torch.zeros_like(llm_rewards_t).to(
        device
    )  # Shape: (batch_size, num_steps)
    G = torch.zeros_like(llm_rewards_t[:, 0]).to(
        device
    )  # Shape: (batch_size, num_steps)

    for t in reversed(range(kg_rewards_t.size(1))):
        G = (llm_rewards_t[:, t] + kg_rewards_t[:, t]) + gamma * G
        discounted_rewards[:, t] = G

    # discounted_rewards[:,-1] = llm_rewards_t[:,-1] + kg_rewards_t[:,-1]
    # for t in reversed(range(num_steps - 1)):
    #     discounted_rewards[:,t] += gamma * (llm_rewards_t[:,t + 1] + kg_rewards_t[:,t + 1])

    # Sample-wise normalization of the rewards for stability
    # discounted_rewards = (discounted_rewards - discounted_rewards.mean(axis=-1)[:, torch.newaxis]) / (discounted_rewards.std(axis=-1)[:, torch.newaxis] + 1e-8)

    # --------------------------------------------------------------------------
    "Loss Calculation"

    pg_loss = (
        -(discounted_rewards * log_probs_t) - nav_agent.beta * entropies_t
    )  # Have to negate it into order to do gradient ascent

    logger.warning(f"We just left dev rollout")

    return pg_loss, eval_extras


def batch_loop(
    env: ITLGraphEnvironment,
    mini_batch: pd.DataFrame,  # Perhaps change this ?
    nav_agent: ContinuousPolicyGradient,
    hunch_llm: nn.Module,
    steps_in_episode: int,
    bos_token_id: int,
    eos_token_id: int,
    pad_token_id: int,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Executes a batch loop for training the navigation agent and language model.
    This function performs reinforcement learning (RL) rollouts for a batch of data
    and computes the policy gradient loss for training.

    During the batch loop:
    - The navigation agent (`nav_agent`) interacts with the environment (`env`) to take actions inside `rollout`.
    - Rewards are computed from both the language model (`hunch_llm`) and the knowledge graph environment (KGE).
    - The policy gradient loss is calculated based on the rewards and log probabilities of actions.

    This function is found within `train_multihopkg` and calls upon `rollout`.

    Args:
        env (ITLGraphEnvironment):
            The knowledge graph environment that provides observations, rewards, and state transitions.
        mini_batch (pd.DataFrame):
            A batch of data containing questions, answers, relevant entities, and relations.
        nav_agent (ContinuousPolicyGradient):
            The policy network responsible for deciding actions based on the current state.
        hunch_llm (nn.Module):
            A language model used to compute rewards based on how well the agent's state aligns with the expected answers.
        steps_in_episode (int):
            The number of steps to execute in each episode.
        bos_token_id (int):
            The token ID representing the beginning of a sequence in the answer IDs.
        eos_token_id (int):
            The token ID representing the end of a sequence in the answer IDs.
        pad_token_id (int):
            The token ID used for padding sequences in the answer IDs.

    Returns:
        - `pg_loss` (torch.Tensor):
            The policy gradient loss computed for the batch.
        - `eval_extras` (Dict[str, Any]):
            A dictionary containing additional evaluation metrics collected during the batch loop.

    Notes:
        - Rewards are normalized for stability before being used to compute the policy gradient loss.
        - This function is designed for training and does not collect as many metrics as `batch_loop_dev`.
    """

    ########################################
    # Start the batch loop with zero grad
    ########################################
    nav_agent.zero_grad()
    device = next(nav_agent.parameters()).device

    print(f"Columns here are {mini_batch.columns}")
    # Deconstruct the batch
    enc_questions = mini_batch["enc_questions"].tolist()
    enc_answers = mini_batch["enc_answer"].tolist()
    path = mini_batch["triples_ints"].tolist()
    enc_questions = mini_batch["enc_questions"].tolist()

    # Lets not use question embeddings
    # if env.use_kge_question_embedding:
    #     question_embeddings = env.get_kge_question_embedding(
    #         query_ent, query_rel, device
    #     )  # Shape: (batch, 2*embedding_dim)
    # else:
    #     question_embeddings = env.get_llm_embeddings(questions, device)

    # NOTE: Question will have to be formatted into the decoder for now
    # That how we pretrained gtllm anyways 

    # answer_ids_padded_tensor = (
    #     collate_token_ids_batch(answers, pad_token_id).to(torch.int32).to(device)
    # )
    # TODO: Come back to this and figure if this is necessary
    pad_mask = answer_ids_padded_tensor.ne(pad_token_id)

    raise NotImplementedError(
        "Have not implemented normal bootstrapped approach. Just deleted rollout"
    )
    # log_probs, entropies, llm_rewards, kg_rewards, eval_extras = rollout(
    #     steps_in_episode,
    #     nav_agent,
    #     hunch_llm,
    #     env,
    #     question_embeddings,
    #     answer_ids_padded_tensor,
    #     query_ent=query_ent,
    #     query_rel=query_rel,
    #     answer_id=answer_id,
    # )

    ########################################
    # Calculate Reinforce Objective
    ########################################
    logger.debug("About to calculate rewards")
    # -------------------------------------------------------------------------
    "LLM Rewards"

    llm_rewards_t = (torch.stack(llm_rewards)).permute(1, 0, 2)

    # Get only masked, then mean
    llm_rewards_t_unpacked = []
    for i, reward_batch_element in enumerate(llm_rewards_t):
        mask_for_element = pad_mask[i][1:].unsqueeze(0).repeat(steps_in_episode, 1)
        filtered_rewards = reward_batch_element[mask_for_element].reshape(
            steps_in_episode, -1
        )
        mean_reward = torch.mean(filtered_rewards, dim=-1)
        llm_rewards_t_unpacked.append(mean_reward)
    llm_rewards_t = torch.stack(llm_rewards_t_unpacked)

    log_probs_t = torch.stack(log_probs).T
    entropies_t = torch.stack(entropies).T
    num_steps = log_probs_t.shape[-1]

    # TODO: Check if this is not bad.
    llm_rewards_t = llm_rewards_t.expand_as(
        log_probs_t
    )  # TOREM: This is a hack to make the shapes match
    # -------------------------------------------------------------------------
    "Knowledge Graph Environment Rewards"

    kg_rewards_t = (torch.stack(kg_rewards)).permute(
        1, 0, 2
    )  # Correcting to Shape: (batch_size, num_steps, reward_type)
    kg_rewards_t = kg_rewards_t.squeeze(2)  # Shape: (batch_size, num_steps)

    # -------------------------------------------------------------------------
    "Discount and Merging of Rewards"

    # TODO: Check if a weight is needed for combining the rewards
    gamma = nav_agent.gamma
    discounted_rewards = torch.zeros_like(llm_rewards_t).to(
        device
    )  # Shape: (batch_size, num_steps)
    G = torch.zeros_like(llm_rewards_t[:, 0]).to(
        device
    )  # Shape: (batch_size, num_steps)

    for t in reversed(range(kg_rewards_t.size(1))):
        G = (llm_rewards_t[:, t] + kg_rewards_t[:, t]) + gamma * G
        discounted_rewards[:, t] = G

    # discounted_rewards[:,-1] = llm_rewards_t[:,-1] + kg_rewards_t[:,-1]
    # for t in reversed(range(num_steps - 1)):
    #     discounted_rewards[:,t] += gamma * (llm_rewards_t[:,t + 1] + kg_rewards_t[:,t + 1])

    # Sample-wise normalization of the rewards for stability
    # discounted_rewards = (discounted_rewards - discounted_rewards.mean(axis=-1)[:, torch.newaxis]) / (discounted_rewards.std(axis=-1)[:, torch.newaxis] + 1e-8)

    # --------------------------------------------------------------------------
    "Loss Calculation"

    pg_loss = (
        -(discounted_rewards * log_probs_t) - nav_agent.beta * entropies_t
    )  # Have to negate it into order to do gradient ascent

    return pg_loss, eval_extras


def evaluate_training(
    env: ITLGraphEnvironment,
    dev_df: pd.DataFrame,
    nav_agent: ContinuousPolicyGradient,
    hunch_llm: nn.Module,
    steps_in_episode: int,
    batch_size_dev: int,
    batch_count: int,
    verbose: bool,
    visualize: bool,
    writer: SummaryWriter,
    question_tokenizer: PreTrainedTokenizer,
    answer_tokenizer: PreTrainedTokenizer,
    wandb_on: bool,
    iteration: int,
    timestamp: str,
):
    """
    Evaluates the performance of the navigation agent and language model on the development set.
    This function computes evaluation metrics, logs results, and optionally visualizes the evaluation process.

    This function is found within `train_multihopkg` and is called periodically during training.
    This function calls upon `batch_loop_dev` and `dump_evaluation_metrics`.

    Args:
        env (ITLGraphEnvironment):
            The knowledge graph environment that provides observations, rewards, and state transitions.
        dev_df (pd.DataFrame):
            The development dataset containing questions, answers, relevant entities, and relations.
        nav_agent (ContinuousPolicyGradient):
            The policy network responsible for deciding actions based on the current state.
        hunch_llm (nn.Module):
            A language model used to compute rewards based on how well the agent's state aligns with the expected answers.
        steps_in_episode (int):
            The number of steps to execute in each episode.
        batch_size_dev (int):
            The batch size for the development set.
        batch_count (int):
            The current batch count during training.
        verbose (bool):
            If `True`, additional information is logged for debugging purposes.
        visualize (bool):
            If `True`, visualizations of the evaluation process are generated.
        writer (SummaryWriter):
            A TensorBoard writer for logging metrics and visualizations.
        question_tokenizer (PreTrainedTokenizer):
            The tokenizer used for processing questions.
        answer_tokenizer (PreTrainedTokenizer):
            The tokenizer used for processing answers.
        wandb_on (bool):
            If `True`, logs metrics to Weights & Biases (wandb).
        num_update_steps (int):
            Maximum number of gradient updates to run across the full training loop.
        iteration (int):
            The current iteration number, used for logging and tracking progress.
        answer_id (List[int], optional):
            A list of IDs corresponding to the correct answer entities. Defaults to `None`.

    Returns:
        None

    Notes:
        - This function evaluates only the last batch of the development set.
        - Metrics are logged to TensorBoard and optionally to wandb.
        - The function ensures that the environment and models are in evaluation mode during the process.
    """
    num_batches = len(dev_df) // batch_size_dev
    nav_agent.eval()
    hunch_llm.eval()

    env.eval()
    # env.question_embedding_module.eval()
    assert (
        not env.question_embedding_module.training
    ), "The question embedding module must not be in training mode"

    batch_cumulative_metrics = {
        "dev/batch_count": [batch_count],
        "dev/pg_loss": [],
    }  # For storing results from all batches

    current_evaluations = (
        {}
    )  # For storing results from last batch. Otherwise too much info

    with torch.no_grad():

        # We will only evaluate on the last batch
        batch_id = num_batches - 1

        mini_batch = dev_df[batch_id * batch_size_dev : (batch_id + 1) * batch_size_dev]

        if not isinstance(  # TODO: Remove this assertion once it is never ever met again
            mini_batch, pd.DataFrame
        ):  # For the lsp to give me a break
            raise RuntimeError(
                f"The mini batch is not a pd.DataFrame, but a {type(mini_batch)}. Please check the data loading code."
            )

        current_evaluations["reference_questions"] = mini_batch["Question"]
        current_evaluations["true_answer"] = mini_batch["Answer"]
        current_evaluations["query_entity"] = mini_batch["Query-Entity"]
        current_evaluations["query_relation"] = mini_batch["Query-Relation"]
        current_evaluations["true_answer_id"] = mini_batch["Answer-Entity"]

        # Get the Metrics
        bos_token_id = answer_tokenizer.bos_token_id
        eos_token_id = answer_tokenizer.eos_token_id
        pad_token_id = answer_tokenizer.pad_token_id
        if bos_token_id is None or eos_token_id is None or pad_token_id is None:
            raise ValueError(
                "Assumptions Wrong. The answer_tokenizer must have a bos_token_id, eos_token_id and pad_token_id"
            )

        # pg_loss, eval_extras = batch_loop_dev(
        #     env,
        #     mini_batch,
        #     nav_agent,
        #     hunch_llm,
        #     steps_in_episode,
        #     pad_token_id,
        # )

        "Extract all the variables from eval_extras"
        for k, v in eval_extras.items():
            current_evaluations[k] = v

        # Accumlate the metrics
        current_evaluations["pg_loss"] = pg_loss.detach().cpu()
        batch_cumulative_metrics["dev/pg_loss"].append(pg_loss.mean().item())

        ########################################
        # Take `current_evaluations` as
        # a sample of batches and dump its results
        ########################################
        if verbose and logger:
            graph_annotation = []
            if env.entity2title:
                for i0 in range(len(env.graph_annotation)):
                    if env.graph_annotation[i0] in env.entity2title.keys():
                        graph_annotation.append(
                            env.entity2title[env.graph_annotation[i0]]
                        )
                    else:
                        graph_annotation.append("")

            # eval_extras has variables that we need
            just_dump_it_here = f"./logs/mlm_{env.knowledge_graph.model_name.lower()}_{timestamp}_evaluation_dumps.log"

            answer_id = current_evaluations["true_answer_id"].tolist()

            answer_kge_tensor = get_embeddings_from_indices(
                env.knowledge_graph.entity_embedding,
                torch.tensor(answer_id, dtype=torch.int),
            ).unsqueeze(
                1
            )  # Shape: (batch, 1, embedding_dim)

            logger.warning(f"About to go into dump_evaluation_metrics")
            dump_evaluation_metrics(
                path_to_log=just_dump_it_here,
                evaluation_metrics_dictionary=current_evaluations,
                vector_entity_searcher=env.ann_index_manager_ent,
                vector_rel_searcher=env.ann_index_manager_rel,
                question_tokenizer=question_tokenizer,
                answer_tokenizer=answer_tokenizer,
                answer_kge_tensor=answer_kge_tensor,
                id2entity=env.id2entity,
                id2relations=env.id2relation,
                entity2title=env.entity2title,
                relation2title=env.relation2title,
                kg_model_name=env.knowledge_graph.model_name,
                kg_ent_distance_func=env.knowledge_graph.absolute_difference,
                kg_rel_denormalize_func=env.knowledge_graph.denormalize_relation,
                kg_rel_wrap_func=env.knowledge_graph.wrap_relation,
                iteration=iteration,
                writer=writer,
                wandb_on=wandb_on,
                logger=logger,
                llm_answered_enabled=True,
            )
            logger.warning(f"We just left dump_evaluation_metrics")

            logger.warning(f"Cleaning up the dev dictionaries")

            current_evaluations.clear()
            eval_extras.clear()

            if not mini_batch._is_view:  # if a copy was created, delete after usage
                del mini_batch

    ########################################
    # Average out all metrics across batches
    # The dump to wandb
    ########################################
    """
    for k, v in batch_cumulative_metrics.items():
        metric_to_report = 0
        if isinstance(v[0],torch.Tensor):
            metric_to_report = torch.stack(v).mean()
        elif isinstance(v[0], int) or isinstance(v[0], float):
            metric_to_report = v[0]
        else:
            raise ValueError(f"The metric to report is not a tensor or int but rather {type(v[0])}")

        if wandb_run is not None:
            wandb.log({k: metric_to_report})
        logger.debug(f"Metric '{k}' has value {metric_to_report}")


    nav_agent.train()
    hunch_llm.train()
    env.train()
    dev_mode = False
    logger.info("Done with Evaluation")
    """
    # TODO: Implement this


class QuestionCoverageSampler:
    """Sample question ids uniformly while tracking coverage cycles."""

    def __init__(self, question_ids: Sequence[int]) -> None:
        if not question_ids:
            raise ValueError("QuestionCoverageSampler requires at least one question id")
        self._question_ids = list(question_ids)
        self._rng = random.Random()
        self._perm: List[int] = []
        self._cursor: int = 0
        self.cycles_completed: int = 0
        self._needs_reshuffle: bool = False
        self._reshuffle()

    def _reshuffle(self) -> None:
        self._perm = self._question_ids.copy()
        self._rng.shuffle(self._perm)
        self._cursor = 0
        self._needs_reshuffle = False

    def sample(self, batch_size: int) -> Tuple[List[int], int]:
        """Return ``batch_size`` question ids and coverage cycles closed in this draw."""

        if batch_size <= 0:
            return [], 0

        len_ids = len(self._question_ids)
        sampled: List[int] = []
        completed_cycles = 0

        if self._needs_reshuffle:
            self._reshuffle()

        while len(sampled) < batch_size:
            remaining = len_ids - self._cursor
            take = min(batch_size - len(sampled), remaining)
            sampled.extend(self._perm[self._cursor : self._cursor + take])
            self._cursor += take

            if self._cursor == len_ids:
                completed_cycles += 1
                self.cycles_completed += 1
                if len(sampled) < batch_size:
                    self._reshuffle()
                else:
                    self._needs_reshuffle = True

        return sampled, completed_cycles

    def get_progress(self) -> float:
        """Fraction of unique questions touched in the current coverage sweep."""

        return self._cursor / len(self._question_ids)



@torch.no_grad()
def prepopulate_replay_buffer(
    *,
    env: ReinforcedUnsupervisedEnv,
    actor: ContinuousPolicyGradient,
    hunch_llm: nn.Module,
    replay_buffer: QuestionReplayBuffer,
    train_df: pd.DataFrame,
    num_simulations_per_question: int, # Simulations ~= Transitions  
    max_env_steps: int,
    pad_token_id: int,
) -> QuestionReplayBuffer:

    actor.eval()
    hunch_llm.eval()
    BATCH_SIZE=64 # TODO: Parameterize later
    gpu_device = next(actor.parameters()).device
    cpu_device = replay_buffer.cur_states.device
    # Lets Initiate sub buffers for each question

    for i in tqdm(range(0, len(train_df), BATCH_SIZE), f"Populating the replay buffer"):
        mini_batch = train_df.iloc[i : i + BATCH_SIZE]
        _inner_batch_size = len(mini_batch)

        # Get questions ready for batch Bart processing
        questions = [torch.Tensor(ques).to(torch.long) for ques in train_df.loc[mini_batch.index, "enc_questions"]]
        padded_questions_tokens = torch.nn.utils.rnn.pad_sequence(
            questions, batch_first=True, padding_value=pad_token_id
        ).to(gpu_device)
        padded_questions_tokens = padded_questions_tokens.view(-1,1,padded_questions_tokens.shape[-1])
        padded_questions_tokens = padded_questions_tokens.repeat(1,num_simulations_per_question,1).squeeze(1)

        # Answers Graph Embeddings
        paths = mini_batch.loc[:, "triples_ints"]
        answer_graphemb_idxs = torch.LongTensor([path[-1] for path in paths]).to(gpu_device) # TODO: see if we can remove device
        answer_graphemb_idxs = answer_graphemb_idxs.unsqueeze(1).repeat(1,num_simulations_per_question).view(-1).to(gpu_device)

        bert_emb_dim = replay_buffer.get_question_bert_emb_dim()

        # Question Bert Embedding
        bert_quest_emb = torch.Tensor(mini_batch.iloc[:, 3 + bert_emb_dim:].values.tolist())
        bert_quest_emb = (
            bert_quest_emb.view(-1, 1, bert_quest_emb.shape[-1])
            .repeat(1, num_simulations_per_question, 1)
            .view(-1,bert_emb_dim)
        ).to(gpu_device)


        # Answer Bert Heuristic
        answer_bert_heuristics = torch.Tensor(mini_batch.iloc[:, 3: 3 + bert_emb_dim].values.tolist())
        answer_bert_heuristics = (
            answer_bert_heuristics.view(-1, 1, answer_bert_heuristics.shape[-1])
            .repeat(1, num_simulations_per_question, 1)
            .squeeze(1)
        ).to(gpu_device)

        #.... reset
        init_states = env.reset(bert_quest_emb)
        # init_states = init_states.detach().to(cpu_device)
        padded_path = torch.full([init_states.shape[0], max_env_steps*2 + 1, init_states.shape[1]], PATH_PADDING_VALUE, dtype=torch.float)
        padded_path[:,0,:] = init_states

        #.... Action

        # Initially we use random actions
        action = torch.empty(init_states.shape).uniform_(-1.0, 1.0).to(gpu_device)
        log_prob = None
        entropy = None

        # ... Environment step
        observation = ReinforcedUnsupervisedEnv.RUE_Observation(
            state=init_states,
            answer_id=answer_graphemb_idxs
        )
        
        # TODO: Either get LLM reward inside of this function or remove this one.
        next_state, extrinsic_reward, done, = env.step(observation, action)

        # Calculate Main Reward (Yeah, outside the environment for now)
        bert_emb_dim = answer_bert_heuristics.shape[-1]
        next_state_path = padded_path.clone()
        next_state_path[:,1,:] = action
        next_state_path[:,2,:] = next_state
        llm_reward, _ = calculate_llm_reward_supasoft(
            hunch_llm,
            next_state_path,
            answer_bert_heuristics.view(-1, bert_emb_dim),
            padded_questions_tokens.view(-1, padded_questions_tokens.shape[-1]),
            pad_token_id,
        )
        combined_reward = llm_reward #+ extrinsic_reward.squeeze(-1)  #NOTE: at some point we might be interested in using this extrinsic reward.

        # Obviously this is only a one step thing:
        step_counter = torch.zeros_like(llm_reward, dtype=torch.long)
        ########################################
        # Fill out Replay Buffer 
        ########################################
        action_dim = action.shape[-1]
        state_dim = init_states.shape[-1]
        question_n_exp_idxs = torch.Tensor(mini_batch.index).to(torch.long).unsqueeze(1).repeat(1, num_simulations_per_question).view(-1)
        replay_buffer.add_transitions(
            questions_ids=question_n_exp_idxs,
            quest_bert_emb=bert_quest_emb,
            cur_states=init_states.view(-1, state_dim), # TODO: we might want to remove this since we already have path_states
            actions=action.view(-1, action_dim).detach().to(cpu_device),
            rewards=combined_reward.detach().to(cpu_device),
            path_states = padded_path.view(_inner_batch_size * num_simulations_per_question, -1, padded_path.shape[-1]),
            next_states=next_state.view(-1, state_dim).detach().to(cpu_device),
            dones=torch.zeros((_inner_batch_size * num_simulations_per_question), dtype=torch.bool),
            log_probs = torch.ones_like(llm_reward, dtype=torch.float), # TODO: make sure we handle this place holder value properly later
            entropies = torch.full_like(llm_reward, -1.0),
            step_counter = step_counter,
        )
        del bert_quest_emb, answer_graphemb_idxs, answer_bert_heuristics, init_states, next_state, llm_reward, combined_reward, action, padded_path
        torch.cuda.empty_cache()
    actor.train()
    # hunch_llm does not really need to be trained here.

    return replay_buffer


@torch.no_grad()
def hydrate_replay_buffer(
    num_hydration_samples: int,
    env: ReinforcedUnsupervisedEnv,
    actor: ContinuousPolicyGradient,
    hunch_llm: nn.Module,
    replay_buffer: QuestionReplayBuffer,
    train_df: pd.DataFrame,
    traindf_question_ids: Sequence[int],
    pad_token_id: int,
) -> int:
    """Generate new transitions by extending oldest trajectories in replay."""

    actor.eval()
    hunch_llm.eval()

    device = next(actor.parameters()).device
    cpu_device = replay_buffer.cur_states.device
    bert_embed_dim = replay_buffer.get_question_bert_emb_dim()
    max_path_len = replay_buffer.path_states.shape[2]
    state_dim = replay_buffer.path_states.shape[-1]
    max_path_len =  replay_buffer.get_max_path_len()
    max_steps = (max_path_len-1)//2
    amount_ques = len(traindf_question_ids)
    seq_positions = torch.arange(max_path_len, device=device).unsqueeze(0)

    sample_size = min(len(traindf_question_ids), num_hydration_samples)

    # Get Samples
    question_counts = Counter(random.choices(traindf_question_ids, k=sample_size))
    (
        sampled_qidx,
        experiences_qids,
        bert_quest,
        path_states,
        actions,
        next_states,
        step_counter,
        done_flags,
    ) = replay_buffer.get_oldest_experiences(question_counts, device)

    mini_batch = train_df.loc[traindf_question_ids]

    questions_tokens = [torch.Tensor(ques).to(torch.long) for ques in train_df.loc[mini_batch.index, "enc_questions"]]
    padded_questions_tokens = torch.nn.utils.rnn.pad_sequence(
        questions_tokens, batch_first=True, padding_value=pad_token_id
    ).to(device)
    # NOTE: Double check on this indexing
    answer_bert_embs = (
        torch.Tensor(
            train_df.iloc[mini_batch.index, 3 : 3 + bert_embed_dim].values.tolist()
        )
        .to(torch.long)
        .to(device)
    )

    answer_ids = [path[-1] for path in mini_batch["triples_ints"].tolist()]
    answer_ids_tensor = torch.tensor(answer_ids, dtype=torch.long, device=device)

    if done_flags.any():
        reset_states = env.reset(bert_quest[done_flags])
        new_paths = torch.full(
            (reset_states.shape[0], max_path_len, state_dim),
            PATH_PADDING_VALUE,
            device=device,
            dtype=path_states.dtype,
        )
        new_paths[:, 0, :] = reset_states
        path_states[done_flags] = new_paths
        step_counter[done_flags] = 0
    else:
        notdone_flags = ~done_flags
        steps = step_counter[notdone_flags]
        path_states[notdone_flags, ((steps+1)*2) + 1] = actions[notdone_flags]
        path_states[notdone_flags, ((steps+1)*2) + 2] = next_states[notdone_flags]
        # CHeck if its done
        step_counter[notdone_flags]  += 1
        done_flags[notdone_flags] = steps == max_steps

    valid_mask = torch.zeros((amount_ques, max_path_len), dtype=torch.bool)
    for row_idx in range(step_counter.shape[0]):
        valid_mask[row_idx, :step_counter[row_idx] + 1] = True
    graph_state_mask = valid_mask.unsqueeze(1).unsqueeze(2).to(device)
    valid_counts = valid_mask.sum(dim=-1)

    actions, log_probs, entropy, _, _ = actor(
        path_states,
        graph_state_mask=graph_state_mask,
        context_quest_bert_emb=bert_quest,
    )

    row_idx = torch.arange(amount_ques, device=device)
    current_states = path_states[row_idx, valid_counts, :]

    observation = ReinforcedUnsupervisedEnv.RUE_Observation(
        state=current_states,
        answer_id=answer_ids_tensor,
    )

    # Take a Step
    next_states, extrinsic_reward, done = env.step(observation, actions)

    # Calculate Reward
    # TODO: Confirm this works well
    next_state_path = path_states.clone()
    next_state_path[row_idx, (step_counter + 1) * 2 + 1, :] = next_states
    next_state_path[row_idx, (step_counter + 1) * 2 + 2, :] = next_states
    llm_reward, _ = calculate_llm_reward_supasoft(
        hunch_llm,
        #NOTE : Check on this unsqueeze
        next_states.unsqueeze(1),
        answer_bert_embs,
        padded_questions_tokens,
        pad_token_id,
    )

    combined_reward = llm_reward.squeeze()# + extrinsic_reward.squeeze()

    # path_states_updated = path_states.clone()
    # action_indices = 2 * current_steps + 1
    # state_indices = action_indices + 1
    #
    # within_bounds = state_indices < max_path_len
    # if not within_bounds.all():
    #     overflow_mask = ~within_bounds
    #     action_indices = torch.clamp(action_indices, max=max_path_len - 2)
    #     state_indices = action_indices + 1
    #     done = done.clone()
    #     done[overflow_mask] = True
    #
    # path_states_updated[row_idx, action_indices, :] = actions
    # path_states_updated[row_idx, state_indices, :] = next_states

    replay_buffer.add_transitions(
        questions_ids=sampled_qidx.to(cpu_device),
        quest_bert_emb=bert_quest.detach().to(cpu_device),
        cur_states=current_states.detach().to(cpu_device),
        actions=actions.detach().to(cpu_device),
        rewards=combined_reward.detach().to(cpu_device),
        next_states=next_states.detach().to(cpu_device),
        dones=done.squeeze(-1).detach().to(torch.bool).cpu(),
        path_states=path_states.detach().cpu(),
        log_probs=log_probs.detach().cpu(),
        entropies=entropy.detach().cpu(),
        step_counter=step_counter.detach().cpu(),
    )

    actor.train()
    hunch_llm.train()

    return path_states.shape[0]



@torch.no_grad()
def evaluate_seq2seq_outputs(
    env: ReinforcedUnsupervisedEnv,
    nav_agent: ContinuousPolicyGradient,
    hunch_llm: nn.Module,
    dataset: pd.DataFrame,
    question_tokenizer: PreTrainedTokenizer,
    answer_tokenizer: PreTrainedTokenizer,
    batch_size: int,
    bert_dim: int,
    max_env_steps: int,
    bart_pad_token_id: int,
    global_step: int,
    prefix: str,
    writer: SummaryWriter,
    num_samples_to_log: int = 5,
) -> Dict[str, float]:
    """Run policy evaluation on a dataset and log seq2seq decoder outputs."""

    if dataset is None or len(dataset) == 0:
        return {}

    device = next(nav_agent.parameters()).device
    state_dim = nav_agent.observation_dim
    max_path_len = max_env_steps * 2 + 1
    max_transitions = max_env_steps

    assert isinstance(bert_dim, int)
    nav_mode = nav_agent.training
    llm_mode = hunch_llm.training
    nav_agent.eval()
    hunch_llm.eval()
    # env.eval()

    total_nll = 0.0
    total_tokens = 0
    total_token_correct = 0
    total_sequences = 0
    exact_match_count = 0
    success_count = 0
    total_steps = 0.0
    total_generated_len = 0.0
    mse_alignment_sum = 0.0
    mse_alignment_count = 0

    sample_logs: List[str] = []

    seq_positions = torch.arange(max_path_len, device=device).unsqueeze(0)

    total_batches = math.ceil(len(dataset) / batch_size)

    for batch_idx in range(total_batches):
        # Get the Minibatch
        start = batch_idx * batch_size
        end = min((start + batch_size), len(dataset))
        mini_batch = dataset.iloc[
            start : end
        ]
        _batch_size = end - start

        # Get individual columns and do post processing
        question_tokens_list = [q for q in mini_batch["enc_questions"].tolist()]
        answer_tokens_list = [ans for ans in mini_batch["enc_answer"].tolist()]
        # TODO: Confirm that question_token_list has a `2` at the end (separator token)
        question_tokens_list_tensor = [torch.tensor(q, dtype=torch.long) for q in mini_batch["enc_questions"].tolist()]
        padded_questions = torch.nn.utils.rnn.pad_sequence(question_tokens_list_tensor, batch_first=True, padding_value=bart_pad_token_id).to(device)
        bart_questions_mask = padded_questions != bart_pad_token_id
        answer_tokens_list_tensor = [torch.tensor(ans, dtype=torch.long) for ans in mini_batch["enc_answer"].tolist()]
        # padded_answers = torch.nn.utils.rnn.pad_sequence(answer_tokens_list_tensor, batch_first=True, padding_value=bart_pad_token_id).to(device)
        # max_answer_len = max(len(ans) for ans in answer_tokens_list_tensor)

        # TODO: Check if we actually have the answer tokens
        # labels = padded_answers.clone()
        # labels[labels == pad_token_id] = -100
        bert_quest = torch.tensor(mini_batch.iloc[:,3 + bert_dim:].values.tolist(), dtype=torch.float32, device=device)
        bert_ans = torch.tensor(mini_batch.iloc[:,3:3 + bert_dim].values.tolist(), dtype=torch.float32, device=device)

        paths = mini_batch["triples_ints"].tolist()
        answer_entity_ids = torch.tensor([path[-1] for path in paths], dtype=torch.long, device=device)
        init_states = env.reset(bert_quest)
        path_trace = torch.full(
            (_batch_size, max_path_len, state_dim),
            PATH_PADDING_VALUE,
            device=device,
            dtype=init_states.dtype,
        )
        path_trace[:, 0, :] = init_states

        step_counter = torch.zeros(_batch_size, dtype=torch.long, device=device)
        done_mask = torch.zeros(_batch_size, dtype=torch.bool, device=device)
        # success_flags = torch.zeros(_batch_size, dtype=torch.bool, device=device)

        step_active = torch.Tensor([])

        for i in range(max_transitions):
            active_idx = (~done_mask).nonzero(as_tuple=False).squeeze(-1)
            if active_idx.numel() == 0:
                break

            path_active = path_trace[active_idx]
            step_active = step_counter[active_idx]
            mask_active = seq_positions <= (2 * step_active).unsqueeze(1)

            actions, _, _, _, _ = nav_agent(
                path_active,
                graph_state_mask=mask_active,
                context_quest_bert_emb=bert_quest[active_idx],
            )

            row_idx = torch.arange(active_idx.size(0), device=device)
            # TODO: confirm this looks okay
            current_states = path_active[row_idx, 2 * step_active, :]

            observation = ReinforcedUnsupervisedEnv.RUE_Observation(
                state=current_states,
                answer_id=answer_entity_ids[active_idx],
            )

            # Agent Take Step
            next_states, extrinsic_reward, done = env.step(observation, actions)

            path_trace[active_idx, 2 * step_active + 1, :] = actions
            path_trace[active_idx, 2 * step_active + 2, :] = next_states

            # success_flags[active_idx] |= done.squeeze(-1).bool()

            step_counter[active_idx] = step_active + 1

            # TOREM: This should not be necessary
            # done_mask[active_idx] = done.squeeze(-1).bool() | (
            #     step_counter[active_idx] >= max_transitions
            # )

        # final_indices = torch.clamp(2 * step_counter, max=max_path_len - 1) # Tstep_activeOREM: I really dont see this being necessary for now
        idxs_out_of_range = [step_counter[i] > max_transitions for i in range(len(step_counter))]
        assert not any(idxs_out_of_range), "There should be no id out of range" # TOREM: After debugging for long enough
        # final_states = path_trace[torch.arange(_batch_size, device=device), step_counter, : ].unsqueeze(1)
        # encoder_attention_mask = torch.ones((_batch_size, final_states.shape[1]), dtype=torch.long, device=device)
        encoder_attention_mask = torch.zeros((_batch_size, max_path_len), dtype=torch.bool, device=device)
        for i in range(_batch_size):
            encoder_attention_mask[i, :step_active[i]] = True

        translated_embeddings = hunch_llm.embedding_translator(path_trace) # type: ignore
        ########################################
        # TODO: Logit Loss Calculation
        ########################################
        qna_tokens, answer_mask = GraphEmbeddingDataset._merge_questions_and_answers(
            question_tokens_list, answer_tokens_list, bart_pad_token_id
        )
        qna_tokens_tensor = [ torch.tensor(qna_token, dtype=torch.long, device=device) for qna_token in qna_tokens ]
        padded_qna_tokens = torch.nn.utils.rnn.pad_sequence(
            qna_tokens_tensor, batch_first=True, padding_value=bart_pad_token_id
        )
        decoder_attention_mask = torch.ones_like(padded_qna_tokens, dtype=torch.long, device=device)
        decoder_attention_mask[qna_tokens == bart_pad_token_id] = 0
        bart_outputs = hunch_llm.bart( # type:ignore
            inputs_embeds=translated_embeddings,
            attention_mask=encoder_attention_mask,
            decoder_input_ids=padded_qna_tokens,
            decoder_attention_mask=decoder_attention_mask,
            # labels=padded_answers,
            output_hidden_states=True,
        )
        logits = bart_outputs.logits
        #
        anslogits_pred_list = []
        shapes_for_reconstruction = []
        debug_meep = []
        for i, logit in enumerate(logits):
            ans_pred_ids = torch.nonzero(torch.tensor(answer_mask[i])).squeeze() - 1
            selected_logits = logit[ans_pred_ids]
            if len(selected_logits.shape) == 1:
                selected_logits = selected_logits.unsqueeze(0)
            anslogits_pred_list.append(selected_logits)
            debug_meep.append(selected_logits)
            shapes_for_reconstruction.append(selected_logits.shape[0])

        # We then concatenate it into a single 0-dim because it will simply go to cross entropy
        anslogits_pred_list = torch.cat(anslogits_pred_list, dim=0)

        # shift_logits = logits[:, :-1, :].contiguous()
        # shift_labels = padded_answers[:, 1:].contiguous()
        flattened_answer = torch.cat(answer_tokens_list_tensor).to(device)
        nll = F.cross_entropy(
            anslogits_pred_list,
            flattened_answer,
            reduction="sum",
            ignore_index=bart_pad_token_id,
        )
        total_nll += nll.item()
        total_tokens += flattened_answer.numel()
        #
        predictions = anslogits_pred_list.argmax(dim=-1)
        total_token_correct += ((predictions == flattened_answer)).sum().item()

        # For Human Visualization Later
        offset = 0
        legible_tokens = []
        for size in shapes_for_reconstruction:
            legible_tokens.append(
                predictions[offset : offset + size].tolist()
            )
            offset += size

        ########################################
        # Actual NLP Generation (Using Beam-search)
        ########################################
        # TODO: We have to to change decoder_start_token_id=answer_tokenizer.bos_token_id
        # To include the question at the beginning
        # We need to look for the logic in pretraining
        # Prep questions
        generated_ids = hunch_llm.bart.generate( # type: ignore
            decoder_input_ids=padded_questions,
            inputs_embeds=translated_embeddings,
            attention_mask=encoder_attention_mask,
            decoder_attention_mask=bart_questions_mask,
            # decoder_start_token_id=answer_tokenizer.bos_token_id,
            # eos_token_id=answer_tokenizer.eos_token_id,
            # pad_token_id=pad_token_id,
            max_length=100,
            num_beams=3,
        )

        pred_texts = answer_tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )
        ref_texts = answer_tokenizer.batch_decode(answer_tokens_list, skip_special_tokens=True)

        total_sequences += _batch_size
        # success_count += success_flags.sum().item()
        total_steps += step_counter.float().mean().item()
        total_generated_len += (
            generated_ids.ne(bart_pad_token_id).sum(dim=1).float().mean().item()
        )

        # For supasoft reward
        last_decoder_hidden_state = bart_outputs.decoder_hidden_states[-1]

        # 2. Pooling to get a single vector.
        # TODO: Figure out the attention_mask
        pooled = (last_decoder_hidden_state * decoder_attention_mask.unsqueeze(-1)).sum(1) / encoder_attention_mask.sum(1, keepdim=True)
        pooled = hunch_llm.activation(hunch_llm.pooler(pooled)) # type: ignore
        bert_alignment_inference = hunch_llm.projection(pooled)  # type: ignore

        # TODO: implement
        reward_supasoft, _ = calculate_llm_reward_supasoft(
            hunch_llm,
            path_trace,
            bert_ans,
            padded_questions,
            bart_pad_token_id,
        )
        # mse_alignment_sum += (-reward_supasoft).sum().item()
        # mse_alignment_count += reward_supasoft.numel()

        for pred, ref, q_tokens in zip(
            pred_texts,
            ref_texts,
            question_tokens_list_tensor,
        ):
            if len(sample_logs) >= num_samples_to_log:
                break
            question_text = question_tokenizer.decode(
                q_tokens.tolist(), skip_special_tokens=True
            )
            sample_logs.append(
                f"Q: {question_text} | Pred: {pred.strip()} | Ref: {ref.strip()}"
            )

        exact_match_count += sum(
            1 for pred, ref in zip(pred_texts, ref_texts) if pred.strip() == ref.strip()
        )

    # TODO: reimplement
    metrics: Dict[str, float] = {}
    if total_tokens > 0:
        avg_ce = total_nll / total_tokens
        metrics[f"{prefix}/cross_entropy"] = avg_ce
        metrics[f"{prefix}/perplexity"] = math.exp(avg_ce)
        metrics[f"{prefix}/token_accuracy"] = total_token_correct / total_tokens
    if total_sequences > 0:
        metrics[f"{prefix}/success_rate"] = success_count / total_sequences
        metrics[f"{prefix}/exact_match"] = exact_match_count / total_sequences
        metrics[f"{prefix}/avg_step_count"] = total_steps / max(total_sequences, 1)
        metrics[f"{prefix}/avg_generated_length"] = (
            total_generated_len / max(total_sequences, 1)
        )
    if mse_alignment_count > 0:
        metrics[f"{prefix}/bert_alignment_mse"] = (
            mse_alignment_sum / mse_alignment_count
        )

    for metric_name, metric_value in metrics.items():
        writer.add_scalar(metric_name, metric_value, global_step)
    # if wandb_on:
    #     wandb.log(metrics, step=global_step)
    if logger and sample_logs:
        logger.info("=== Seq2Seq Evaluation Samples (%s) ===", prefix)
        for line in sample_logs:
            logger.info(line)

    nav_agent.train()
    hunch_llm.train()

    return metrics






def train_multihopkg(
    epochs: int, 
    batch_size: int,
    batch_size_dev: int,
    nav_agent: ContinuousPolicyGradient,
    hunch_llm: nn.Module,
    learning_rate: float,
    env: ReinforcedUnsupervisedEnv,
    data_partitions: DataPartitions,
    replay_buffer: QuestionReplayBuffer,
    bart_pad_token_id: int,
    mbatches_b4_eval: int,
    question_tokenizer: PreTrainedTokenizer,
    answer_tokenizer: PreTrainedTokenizer,
    num_batches_till_eval: int,
    num_simulations_per_ques: int,
    wandb_on: bool,
    num_update_steps: int,
    num_hydration_samples: int,
    critic_q1: GraphCriticQ,
    critic_q2: GraphCriticQ,
    value_net: GraphCriticV,
):
    if answer_tokenizer.pad_token_id is None:
        raise ValueError(
            "Answer tokenizer must expose a pad token id before replay can be constructed"
        )
    pad_token_id = (
        question_tokenizer.pad_token_id
        if question_tokenizer.pad_token_id is not None
        else answer_tokenizer.pad_token_id
    )

    ########################################
    # Common Tools and Optimizer Setup
    ########################################
    device = next(nav_agent.parameters()).device
    action_shape = (nav_agent.mu_layer.out_features,)

    # For moving average stabilization
    target_value_net: GraphCriticV = copy.deepcopy(value_net)

    critic_optimizer = torch.optim.Adam(
        list(critic_q1.parameters()) + list(critic_q2.parameters()),
        lr=learning_rate,
    )
    value_optimizer = torch.optim.Adam(value_net.parameters(), lr=learning_rate)
    policy_optimizer = torch.optim.Adam(nav_agent.parameters(), lr=learning_rate)

    log_alpha = torch.tensor(
        [math.log(0.2)], device=device, dtype=torch.float32, requires_grad=True
    )
    alpha_optimizer = torch.optim.Adam([log_alpha], lr=learning_rate)
    target_entropy = -float(action_shape[0])
    tau = 0.005
    bert_dim = replay_buffer.get_question_bert_emb_dim()
    gamma = nav_agent.gamma
    hydration_interval = (
        max(1, num_update_steps // 4)
        if num_update_steps is not None and num_update_steps > 0
        else 1
    )
    hydration_rollouts = max(1, num_simulations_per_ques)
    updates_since_hydration = 0
    train_df = data_partitions.train
    question_ids = train_df.index.values.tolist()
    assert isinstance(question_ids, List)
    eval_interval_updates = max(1, num_batches_till_eval)
    last_eval_updates = 0

    ########################################
    # Helper Functions Setup
    ########################################
    def soft_update(source: nn.Module, target: nn.Module, tau: float) -> None:
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.mul_(1.0 - tau)
            target_param.data.add_(tau * param.data)

    def sac_update_step(
        bert_quest_emb: torch.Tensor,
        states_path: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        step_counter: torch.Tensor,
        dones: torch.Tensor,
    ) -> Dict[str, float]:
        nonlocal log_alpha

        alpha = log_alpha.exp()

        ########################################
        # Prepare States and corresp. Masks
        ########################################
        # For cur_state, action, next_state
        next_states_path = states_path.clone()
        next_states_path[:, 2*step_counter + 1, : ] = actions
        next_states_path[:, 2*step_counter + 2, : ] = next_states
        
        # For cur_state, action
        qa_state = states_path.clone()
        qa_state[:, 2*step_counter + 1, : ] = actions

        # Masking
        graph_state_mask = torch.zeros((states_path.shape[0], states_path.shape[1]), dtype=torch.bool).to(device)
        graph_nextState_mask = torch.zeros((states_path.shape[0], states_path.shape[1]), dtype=torch.bool).to(device)
        for i in range(graph_state_mask.shape[0]):
            graph_state_mask[i, : 2 * step_counter[i] + 3] = True
            graph_nextState_mask[i, : 2 * step_counter[i] + 2] = True
        graph_state_mask = graph_state_mask.unsqueeze(1).unsqueeze(2)
        graph_nextState_mask = graph_nextState_mask.unsqueeze(1).unsqueeze(2)

        ########################################
        # Forward Propagation
        ########################################
        with torch.no_grad():
            target_values = target_value_net(
                next_states_path, graph_state_mask, bert_quest_emb
            )
            q_target = rewards + (1.0 - dones.to(torch.float32)) * gamma * target_values.squeeze()

        q1_pred = critic_q1(qa_state, graph_nextState_mask, bert_quest_emb).squeeze()
        q2_pred = critic_q2(qa_state, graph_nextState_mask, bert_quest_emb).squeeze()
        critic_loss = F.mse_loss(q1_pred, q_target) + F.mse_loss(q2_pred, q_target)

        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()

        policy_actions, log_probs, entropy, _, _ = nav_agent(
            states_path,
            graph_state_mask=graph_nextState_mask,
            context_quest_bert_emb=bert_quest_emb,
        )
        policy_state = states_path.clone()
        policy_state[:, 2 * step_counter + 1, :] = policy_actions

        q1_pi = critic_q1(policy_state, graph_nextState_mask, bert_quest_emb)
        q2_pi = critic_q2(policy_state, graph_nextState_mask, bert_quest_emb)
        min_q_pi = torch.min(q1_pi, q2_pi)

        value_target = (min_q_pi - alpha * log_probs.unsqueeze(-1)).detach()
        value_pred = value_net(states_path, graph_nextState_mask, bert_quest_emb)
        value_loss = F.mse_loss(value_pred, value_target)

        value_optimizer.zero_grad()
        value_loss.backward()
        value_optimizer.step()

        policy_loss = (alpha * log_probs.unsqueeze(-1) - min_q_pi).mean()
        policy_optimizer.zero_grad()
        policy_loss.backward()
        policy_optimizer.step()

        alpha_loss = -(log_alpha * (log_probs.detach() + target_entropy)).mean()
        alpha_optimizer.zero_grad()
        alpha_loss.backward()
        alpha_optimizer.step()

        soft_update(value_net, target_value_net, tau)

        rewards_float = rewards.float()
        dones_float = dones.float()
        q_target_mean = q_target.mean().item()
        q_target_std = q_target.float().std(unbiased=False).item()
        q1_mean = q1_pred.mean().item()
        q2_mean = q2_pred.mean().item()
        value_pred_mean = value_pred.mean().item()
        log_prob_mean = log_probs.mean().item()
        log_prob_std = log_probs.float().std(unbiased=False).item()
        mask_token_counts = graph_nextState_mask.squeeze(1).squeeze(1).sum(dim=-1).float()

        return {
            "critic_loss": critic_loss.item(),
            "value_loss": value_loss.item(),
            "policy_loss": policy_loss.item(),
            "alpha_loss": alpha_loss.item(),
            "alpha": alpha.item(),
            "entropy": entropy.mean().item(),
            "log_prob_mean": log_prob_mean,
            "log_prob_std": log_prob_std,
            "reward_mean": rewards_float.mean().item(),
            "reward_std": rewards_float.std(unbiased=False).item(),
            "success_rate": dones_float.mean().item(),
            "q_target_mean": q_target_mean,
            "q_target_std": q_target_std,
            "q1_mean": q1_mean,
            "q2_mean": q2_mean,
            "value_pred_mean": value_pred_mean,
            "step_counter_mean": step_counter.float().mean().item(),
            "mask_tokens_mean": mask_token_counts.mean().item(),
        }

    coverage_sampler = QuestionCoverageSampler(question_ids)

    num_updates_limit = max(1, num_update_steps)
    total_gradient_updates = 0

    local_time = time.localtime()
    timestamp = time.strftime("%m%d%Y_%H%M%S", local_time)
    writer = SummaryWriter(
        log_dir=f"runs/rl/{env.knowledge_graph.model_name.lower()}/{timestamp}/"
    )
    writer.add_scalar("train_config/hydration_interval", hydration_interval, 0)
    writer.add_scalar("train_config/hydration_rollouts", hydration_rollouts, 0)
    if wandb_on:
        wandb.log(
            {
                "train_config/hydration_interval": hydration_interval,
                "train_config/hydration_rollouts": hydration_rollouts,
            },
            step=0,
        )

    collections_per_epoch = math.ceil(len(question_ids) / batch_size)

    for epoch_id in tqdm(range(epochs), desc="Epoch"):
        nav_agent.train()
        hunch_llm.train()
        critic_q1.train()
        critic_q2.train()
        value_net.train()
        target_value_net.eval()

        tqdm_bar = tqdm(range(collections_per_epoch), desc="Collection", leave=False)
        for _ in tqdm_bar:
            ########################################
            # Training/Updates
            ########################################
            # while total_gradient_updates < num_updates_limit:
            sampled_ids, _ = coverage_sampler.sample(batch_size)
            question_counts = {
                int(qid): num_simulations_per_ques for qid in sampled_ids
            }

            (
                _,
                bert_quest_emb,
                actions,
                rewards,
                next_states,
                dones,
                path_states,
                _,
                _,
                step_counter,
            ) = replay_buffer.sample_transitions(question_counts)

            update_metrics = sac_update_step(
                bert_quest_emb.to(device),
                path_states.to(device),
                actions.to(device),
                rewards.to(device),
                next_states.to(device),
                step_counter.to(device),
                dones.to(device),
            )
            update_metrics["coverage_progress"] = coverage_sampler.get_progress()

            if wandb_on:
                wandb.log({f"train/{k}": v for k, v in update_metrics.items()})
            for metric_name, metric_value in update_metrics.items():
                writer.add_scalar(
                    f"train/{metric_name}", metric_value, total_gradient_updates
                )
            total_gradient_updates += 1
            updates_since_hydration += 1

            if updates_since_hydration >= hydration_interval:
                added = hydrate_replay_buffer(
                    num_hydration_samples=num_hydration_samples,
                    env=env,
                    actor=nav_agent,
                    hunch_llm=hunch_llm,
                    replay_buffer=replay_buffer,
                    train_df=train_df,
                    traindf_question_ids=question_ids,
                    pad_token_id=pad_token_id,
                )
                if added:
                    if wandb_on:
                        wandb.log({"train/rehydrated_transitions": added})
                    writer.add_scalar(
                        "train/rehydrated_transitions",
                        added,
                        total_gradient_updates,
                    )
                updates_since_hydration = 0

            if total_gradient_updates >= num_updates_limit:
                break

            if total_gradient_updates >= num_updates_limit:
                break

        logger.info(
            "Epoch %d completed | replay_size=%d | gradient_updates=%d | coverage_cycles=%d",
            epoch_id,
            len(replay_buffer),
            total_gradient_updates,
            coverage_sampler.cycles_completed,
        )

        epoch_progress = coverage_sampler.get_progress()
        if wandb_on:
            wandb.log(
                {
                    "train/coverage_cycles": coverage_sampler.cycles_completed,
                    "train/coverage_progress_epoch": epoch_progress,
                }
            )
        writer.add_scalar(
            "train/coverage_cycles", coverage_sampler.cycles_completed, epoch_id
        )
        writer.add_scalar(
            "train/coverage_progress_epoch", epoch_progress, epoch_id
        )

        if total_gradient_updates - last_eval_updates >= eval_interval_updates:
            evaluate_seq2seq_outputs(
                env=env,
                nav_agent=nav_agent,
                hunch_llm=hunch_llm,
                dataset=data_partitions.validation,
                question_tokenizer=question_tokenizer,
                answer_tokenizer=answer_tokenizer,
                batch_size=batch_size_dev,
                bert_dim=bert_dim,
                max_env_steps=replay_buffer.max_env_steps,
                bart_pad_token_id=bart_pad_token_id,
                global_step=total_gradient_updates,
                prefix="dev",
                writer=writer,
                num_samples_to_log=1,
            )

            # evaluate_seq2seq_outputs(
            #     env=env,
            #     nav_agent=nav_agent,
            #     hunch_llm=hunch_llm,
            #     dataset=data_partitions.test,
            #     question_tokenizer=question_tokenizer,
            #     answer_tokenizer=answer_tokenizer,
            #     batch_size=batch_size_dev,
            #     max_env_steps=replay_buffer.max_env_steps,
            #     pad_token_id=pad_token_id,
            #     writer=writer,
            #     wandb_on=wandb_on,
            #     global_step=total_gradient_updates,
            #     logger=globals().get("logger"),
            #     prefix="test",
            #     num_rollouts_per_question=1,
            #     num_samples_to_log=3,
            # )

            last_eval_updates = total_gradient_updates

        if total_gradient_updates >= num_updates_limit:
            logger.info("Reached gradient update budget; stopping training loop early.")
            break


# TODO: Move function to a separate file
def calculate_llm_reward_autoregressive(
    hunch_llm: nn.Module,
    obtained_state: torch.Tensor,
    answers_ids: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Will take the answers and give an idea of how close we were.
    This will of course require us to have a language model that will start giving us the  answer.
    """
    batch_size = answers_ids.size(0)
    seq_max_len = answers_ids.size(1)
    hidden_dim = obtained_state.shape[-1]

    # From the obtained_state we will try to find an answer
    conditioning_labels = answers_ids[:, :-1].contiguous().to(dtype=torch.int64)
    teacher_forcing_labels = answers_ids[:, 1:].contiguous().to(dtype=torch.int64)

    answers_inf_softmax = hunch_llm(
        graph_embeddings=obtained_state, decoder_input_ids=conditioning_labels
    )

    _, logits = answers_inf_softmax.loss, answers_inf_softmax.logits

    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")

    # TODO: Ensure teacher-forcing is useful/necessary
    loss = loss_fn(logits.view(-1, logits.shape[-1]), teacher_forcing_labels.view(-1))

    # TODO: Perhaps Stabilize the loss. Normalize it or SMTH like that
    reward = -loss  # We expect this reward function to be concave rather than convex.

    # Reshape the reward to the batch size
    reward = reward.view(batch_size, -1)

    # # Get indices of the max value of the final output
    # answers_inf_ids = torch.argmax(logits, dim=-1)

    return reward, logits


@torch.no_grad()
def calculate_llm_reward_supasoft(
    hunch_llm: nn.Module,
    obtained_state: torch.Tensor,
    answer_embedding: torch.Tensor,
    question_tokens: torch.Tensor,
    pad_token_id: int,
) -> Tuple[torch.Tensor, torch.Tensor]:

    dec_attention_mask = question_tokens != pad_token_id
    # TODO: Inspect this when you get the whole thing running to discard it as problem ( in case you are having problems)
    paths_attention_mask = ~(obtained_state == BART_PADDING_VALUE).all(dim=-1)
    _, bert_ans_embeddings = hunch_llm(
        graph_embeddings=obtained_state,
        encoder_attention_mask=paths_attention_mask,
        decoder_input_ids=question_tokens,
        decoder_attention_mask=dec_attention_mask,
    )

    mse_loss = F.mse_loss(
        bert_ans_embeddings,
        answer_embedding,
        reduction="none",
    ).mean(dim=-1)
    reward = -mse_loss
    return reward, bert_ans_embeddings


def main():
    # By default we run the config
    # Process data will determine by itself if there is any data to process
    args, logger = initial_setup()
    global wandb_run

    if args.debug:
        logger.info("\033[1;33m Waiting for debugger to attach...\033[0m")
        debugpy.listen(("0.0.0.0", 42023))
        debugpy.wait_for_client()
        # USe debugpy to listen

    # TODO: Muybe ? (They use it themselves)
    # initialize_model_directory(args, args.seed)
    if args.wandb:
        logger.info(
            f"🪄 Initializing Weights and Biases. Under project name {args.wandb_project_name} and run name {args.wr_name}"
        )
        wandb_run = wandb.init(
            project=args.wandb_project_name,
            name=args.wr_name,
            config=vars(args),
            notes=args.wr_notes,
        )

    ########################################
    # Load Pretrained Graph To Language 
    ########################################
    pretrained_gtllm_metadata = torch.load(args.pretrained_gtllm_path, weights_only=False)

    logger.info("Loaded Pretrained Model Metadata, the data preview 如下所示: ")
    # for k,v in pretrained_gtllm_metadata.items():
        # Check if V is an item that can be converted to a string
        # logger.info(f"The key {k} has value {v}")

    logger.info(f"The keys inside of pretrained_model_metadata are {pretrained_gtllm_metadata.keys()}")

    gtllm_hunch_base_model = pretrained_gtllm_metadata["hunchbart_base_llm_model"]
    gtllm_graph_embedding_dim = pretrained_gtllm_metadata["hunchbart_hidden_dim"]
    gtllm_tokenizer_name = pretrained_gtllm_metadata["hunchbart_base_llm_tokenizer"]
    gtllm_tokenizer = AutoTokenizer.from_pretrained(gtllm_tokenizer_name)

    qna_data_path = pretrained_gtllm_metadata["path_mquake_data"]

    hunch_llm = HunchBart.from_pretrained(
        hunchbart_base_llm_model_name=gtllm_hunch_base_model,
        state_dict=pretrained_gtllm_metadata["gtllm_state_dict"],
        graph_embedding_dim=gtllm_graph_embedding_dim,
    ).to(args.device)

    # Prepare all the paremters used to train the graph embedding model
    ge_params = pretrained_gtllm_metadata["embedding_training_metaparam"] # Graph embedding pretrained model parameters
    ge_geom = ge_params["model"]
    ge_gamma = ge_params["gamma"]
    ge_save_path = pretrained_gtllm_metadata["path_graph_emb_data"]


    logger.info(
        "Loaded HunchBart from pretrained model with the following parameters:"
        f"\n\t- Hunchbart Base Model: {gtllm_hunch_base_model}"
        f"\n\t- Hunchbart Base Tokenizer: {gtllm_tokenizer_name}"
        f"\n\t- Graph Embedding Dimension: {gtllm_graph_embedding_dim}"
    )

    # Count the number of loaded parameters
    num_param = sum(p.numel() for p in hunch_llm.parameters() if p.requires_grad)
    logger.info(f"Loaded {num_param} parameters")

    ########################################
    # Set the KG Environment
    ########################################
    # Agent needs a Knowledge graph as well as the environment
    logger.info(":: Setting up the knowledge graph")

    entity_embeddings = np.load(os.path.join(ge_save_path, "entity_embedding.npy"))
    relation_embeddings = np.load(os.path.join(ge_save_path, "relation_embedding.npy"))
    checkpoint = torch.load(os.path.join(ge_save_path, "checkpoint"))
    dim_entity = entity_embeddings.shape[1]
    dim_relation = relation_embeddings.shape[1]

    # Load KGE Model
    kge_model = KGEModel.from_pretrained(
        model_name=ge_geom,
        entity_embedding=entity_embeddings,
        relation_embedding=relation_embeddings,
        gamma=ge_gamma,
        state_dict=checkpoint["model_state_dict"],
    ).to(args.device)
    # For good use:
    kge_model.recalculate_entity_centroid() # Just so we can keep it in the same device

    logger.info(f"Loaded KGE Model with the following parameters:"
                f"\n\t- Graph Model Geometry: {ge_geom}"
                f"\n\t- Graph Model Gamma: {ge_gamma}"
                f"\n\t- Entity Embedding Dimension: {dim_entity}"
                f"\n\t- Relation Embedding Dimension: {dim_relation}"
                f"\n\t- Entity Embedding Shape: {entity_embeddings.shape}"
                f"\n\t- Relation Embedding Shape: {relation_embeddings.shape}"
                f"\n\t- Checkpoint Shape: {checkpoint['model_state_dict'].keys()}"
    )

    # Information computed by knowldege graph for future dependency injection
    dim_entity = kge_model.get_entity_dim()
    dim_relation = kge_model.get_relation_dim()

    ########################################
    # Get the data
    ########################################
    logger.info(":: Setting up the data")

    # Load the KGE Dictionaries
    id2ent, ent2id, id2rel, rel2id = data_utils.load_dictionaries(qna_data_path)

    ########################################
    # Load the QA Dataset
    ########################################
    # At this point we assume that `pretraining` has run so a cache is waiting for us
    # You should not recalcualte the cache anyways, it is very dependent on how pretraining determined the cache
    # Talk to @ottersome if you need more info
    train_df, dev_df, test_df, metadata = data_utils.load_cached_pretraining_data(args.pretraining_metadata_cache_path)
    bert_emb_dim = metadata["ques_cols"][1] - metadata["ques_cols"][0]

    data_partitions = DataPartitions(
        train_df, dev_df, test_df
    )
    ########################################
    # Load Bert Embeddings
    ########################################
    # bert_model = AutoModel.from_pretrained(args.bert_model)
    # train_ques_list = train_df["enc_questions"].tolist()
    # quest_bert_dim = bert_model.config.hidden_size
    # embedded_questions = [
    #     bert_model(torch.LongTensor(enc_ques))
    #     for enc_ques in train_ques_list
    # ]

    # Get the Module for Approximate Nearest Neighbor Search
    ########################################
    # Setup the ann index.
    # Will be needed for obtaining observations.
    ########################################

    logger.info(":: Setting up the ANN Index")

    ########################################
    # Setup the Vector Searchers
    ########################################
    # TODO: Reincorporate it for extrinsic reward.
    # if ge_geom == "pRotatE":  # for rotational kge models
    #     ann_index_manager_ent = ANN_IndexMan_pRotatE(
    #         kge_model.get_all_entity_embeddings_wo_dropout(),
    #         embedding_range=kge_model.embedding_range.item(),
    #     )
    #     ann_index_manager_rel = ANN_IndexMan_pRotatE(
    #         kge_model.get_all_relations_embeddings_wo_dropout(),
    #         embedding_range=kge_model.embedding_range.item(),
    #     )
    # else:  # for non-rotational kge models
    #     ann_index_manager_ent = ANN_IndexMan(
    #         kge_model.get_all_entity_embeddings_wo_dropout(),
    #         exact_computation=True,
    #         nlist=100,
    #     )
    #     ann_index_manager_rel = ANN_IndexMan(
    #         kge_model.get_all_relations_embeddings_wo_dropout(),
    #         exact_computation=True,
    #         nlist=100,
    #     )

    # Setup the pretrained language model
    logger.info(":: Setting up the pretrained language model")
    config = BartConfig.from_pretrained("facebook/bart-base")
    # Access the hidden size (hidden dimension)
    # TODO: Remove the hardcode. Perhaps
    embedding_hidden_size = config.d_model
    embedding_vocab_size = config.vocab_size
    if args.verbose:
        print(
            f"The hidden dimension of the embedding layer is {embedding_hidden_size} and its vocab size is {embedding_vocab_size}"
        )

    # Setup the entity embedding module
    question_embedding_module = AutoModel.from_pretrained(
        gtllm_hunch_base_model
    ).to(args.device)

    # # Freeze the Question Embedding Module
    # for param in question_embedding_module.parameters():
    #     param.requires_grad = False

    env = ReinforcedUnsupervisedEnv(
        bert_question_embedding_module=question_embedding_module,
        knowledge_graph=kge_model,
        nav_start_emb_type=args.nav_start_emb_type,
        reached_destination_threshold=args.reached_destination_threshold,
    )

    meep = torch.rand(10, 100).to(args.device)
    init_states = env.reset(meep)

    # TODO: Reorganize the parameters lol
    logger.info(":: Setting up the navigation agent")
    assert dim_entity == dim_relation, "Entity and action dimensions must be the same"
    dim_observation = dim_entity
    nav_agent = ContinuousPolicyGradient(
        beta=args.beta,
        gamma=args.rl_gamma,
        dim_action=dim_relation,
        enc_ff_dim=args.enc_ff_dim,
        dim_observation=dim_observation,
        max_path_length=args.max_env_steps * 2 + 1, # We have to account for actions too
        encoder_num_layers=args.num_enc_layers,
        encoder_num_heads=args.num_enc_heads,
        encoder_dropout=args.enc_dropout,
        ques_emb_dim=bert_emb_dim,
        log_std_min=args.log_std_min,
        log_std_max=args.log_std_max,
    ).to(args.device)


    if os.path.exists(args.replay_buffer_cache_path) and not args.force_replaybuffer_recompute:
        logger.info(f"Found replay buffer cache. Will be using it now: {args.replay_buffer_cache_path}")
        with open(args.replay_buffer_cache_path, 'rb') as f: 
            replay_buffer = pickle.load(f)
            assert isinstance(replay_buffer, QuestionReplayBuffer)
            # TODO: Ensure it matches the args like experiences_per_question and so on
    else:
        logger.info(f"Did not find replay buffer cache. Will be creating it now: {args.replay_buffer_cache_path}")
        os.makedirs(os.path.dirname(args.replay_buffer_cache_path), exist_ok=True)
        replay_buffer = QuestionReplayBuffer(
            num_questions = len(train_df), # TODO: check this is correct
            state_shape=dim_entity,
            action_shape=dim_relation,
            bert_emb_dim=bert_emb_dim,
            experiences_per_question=args.experiences_per_question,
            max_env_steps=args.max_env_steps,
            question_ids=train_df.index.tolist(),
        )
        replay_buffer = prepopulate_replay_buffer(
            env=env,
            actor=nav_agent,
            hunch_llm=hunch_llm,
            replay_buffer=replay_buffer,
            train_df=train_df,
            num_simulations_per_question=args.experiences_per_question,
            max_env_steps=args.max_env_steps,
            pad_token_id=gtllm_tokenizer.pad_token_id, # type: ignore
        )
        with open(args.replay_buffer_cache_path,'wb') as f:
            pickle.dump(replay_buffer, f)
        logger.info(f"Replay buffer cache written to: {args.replay_buffer_cache_path}")

    # ======================================
    # Visualizaing nav_agent models using Netron
    # Save a model into .onnx format
    # torch_input = torch.randn(12, 768)
    # onnx_program = torch.onnx.dynamo_export(nav_agent, torch_input)
    # onnx_program.save("models/images/nav_agent.onnx")
    # ======================================

    # TODO: Add checkpoint support
    # See args.start_epoch

    # TODO: Make it take check for a checkpoint and decide what start_epoch
    # if args.checkpoint_path is not None:
    #     # TODO: Add it here to load the checkpoint separetely
    #     nav_agent.load_checkpoint(args.checkpoint_path)

    ######## ######## ########
    # Train:
    ######## ######## ########
    start_epoch = 0
    logger.info(":: Training the model")

    if args.visualize:
        args.verbose = True

    critic_q1 = GraphCriticQ(
        graph_obs_dim=dim_observation,
        action_dim=dim_relation,
        encoder_num_layers=args.num_enc_layers,
        encoder_num_heads=args.num_enc_heads,
        enc_ff_dim=args.enc_ff_dim,
        enc_dropout=args.enc_dropout,
        dim_hidden=args.critic_q_hiddim,
        max_seq_length=args.max_env_steps * 2 + 1,
        ques_emb_dim=bert_emb_dim,
    ).to(args.device)
    critic_q2 = GraphCriticQ(
        graph_obs_dim=dim_observation,
        action_dim=dim_relation,
        encoder_num_layers=args.num_enc_layers,
        encoder_num_heads=args.num_enc_heads,
        enc_ff_dim=args.enc_ff_dim,
        enc_dropout=args.enc_dropout,
        dim_hidden=args.critic_q_hiddim,
        max_seq_length=args.max_env_steps * 2 + 1,
        ques_emb_dim=bert_emb_dim,
    ).to(args.device)
    value_net = GraphCriticV(
        obs_dim=dim_observation,
        encoder_num_layers=args.num_enc_layers,
        encoder_num_heads=args.num_enc_heads,
        enc_ff_dim=args.enc_ff_dim,
        enc_dropout=args.enc_dropout,
        dim_hidden=args.critic_v_hiddim,
        max_seq_length=args.max_env_steps * 2 + 1,
        ques_emb_dim=bert_emb_dim,
    ).to(args.device)

    # # DEBUG: Again, remove this after
    # evaluate_seq2seq_outputs(
    #     env=env,
    #     nav_agent=nav_agent,
    #     hunch_llm=hunch_llm,
    #     dataset=data_partitions.validation,
    #     question_tokenizer=gtllm_tokenizer,
    #     answer_tokenizer=gtllm_tokenizer,
    #     batch_size=args.batch_size_dev,
    #     max_env_steps=replay_buffer.max_env_steps,
    #     bert_dim=replay_buffer.get_question_bert_emb_dim(),
    #     bart_pad_token_id=PATH_PADDING_VALUE,
    #     global_step=1,
    #     prefix="dev",
    # )
    train_multihopkg(
        epochs=args.epochs,
        batch_size=args.batch_size,
        batch_size_dev=args.batch_size_dev,
        nav_agent=nav_agent,
        hunch_llm=hunch_llm,
        learning_rate=args.learning_rate,
        env=env,
        data_partitions=data_partitions,
        replay_buffer=replay_buffer,
        bart_pad_token_id=BART_PADDING_VALUE,
        mbatches_b4_eval=args.batches_b4_eval,
        question_tokenizer=gtllm_tokenizer,
        answer_tokenizer=gtllm_tokenizer,
        num_batches_till_eval=args.num_batches_till_eval,
        num_simulations_per_ques=args.experiences_per_question,
        wandb_on=args.wandb,
        num_update_steps=args.num_update_steps,
        num_hydration_samples=args.num_hydration_samples,
        critic_q1=critic_q1,
        critic_q2=critic_q2,
        value_net=value_net,
    )
    logger.info("Done with everything. Exiting...")

    # TODO: Evaluation of the model
    # metrics = inference(lf)


if __name__ == "__main__":
    main()
