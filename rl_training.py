#!/usr/bin/env python3

"""
Copyright (c) 2018, salesforce.com, inc.
For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

Experiment Portal.
"""

import argparse
import ast
import io
import json
import logging
import os
import sys
import time
import math
from collections import Counter, defaultdict
from typing import Any, DefaultDict, Dict, List, Optional, Tuple

import debugpy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
from multihopkg.rl.graph_search.sac import CriticQ, CriticV
from multihopkg.rl.graph_search.pn import ITLGraphEnvironment, ReinforcedUnsupervisedEnv
from multihopkg.rl.utils import QuestionReplayBuffer, Transition
from multihopkg.run_configs import rl_alpha
from multihopkg.run_configs.common import overload_parse_defaults_with_yaml
from multihopkg.utils.convenience import tensor_normalization
from multihopkg.utils.setup import set_seeds
from multihopkg.utils.wandb import histogram_all_modules
from multihopkg.utils_debug.dump_evals import dump_evaluation_metrics
from multihopkg.vector_search import ANN_IndexMan, ANN_IndexMan_pRotatE

traceback.install()
wandb_run = None


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


def prep_questions(questions: List[torch.Tensor], model: BertModel):
    embedded_questions = model(questions)
    return embedded_questions


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

    This function is found within `train_multihokg` and calls upon `rollout`.

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
    dev_df: pd.DataFrame,
    mbatches_b4_eval: int,
    verbose: bool,
    visualize: bool,
    question_tokenizer: PreTrainedTokenizer,
    answer_tokenizer: PreTrainedTokenizer,
    track_gradients: bool,
    num_batches_till_eval: int,
    num_simulations_per_ques: int,
    wandb_on: bool,
    num_update_steps: int,
    max_env_steps: int,
):
    if answer_tokenizer.pad_token_id is None:
        raise ValueError(
            "Answer tokenizer must expose a pad token id before replay can be constructed"
        )

    device = next(nav_agent.parameters()).device
    state_shape = (nav_agent.hidden1.in_features,)
    action_shape = (nav_agent.mu_layer.out_features,)

    hidden_dim = nav_agent.hidden1.out_features
    critic_q1 = CriticQ(state_shape[0] + action_shape[0], dim_hidden=hidden_dim).to(device)
    critic_q2 = CriticQ(state_shape[0] + action_shape[0], dim_hidden=hidden_dim).to(device)
    value_net = CriticV(in_dim=state_shape[0], dim_hidden=hidden_dim).to(device)
    target_value_net = CriticV(in_dim=state_shape[0], dim_hidden=hidden_dim).to(device)
    target_value_net.load_state_dict(value_net.state_dict())

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
    gamma = nav_agent.gamma

    def soft_update(source: nn.Module, target: nn.Module, tau: float) -> None:
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.mul_(1.0 - tau)
            target_param.data.add_(tau * param.data)

    def sac_update_step(batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        nonlocal log_alpha

        states = batch["states"].to(device)
        actions = batch["actions"].to(device)
        rewards = batch["rewards"].to(device)
        next_states = batch["next_states"].to(device)
        dones = batch["dones"].to(device).float()

        alpha = log_alpha.exp()

        with torch.no_grad():
            target_values = target_value_net(next_states)
            q_target = rewards + (1.0 - dones) * gamma * target_values

        q1_pred = critic_q1(states, actions)
        q2_pred = critic_q2(states, actions)
        critic_loss = F.mse_loss(q1_pred, q_target) + F.mse_loss(q2_pred, q_target)

        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()

        policy_actions, log_probs, entropy, _, _ = nav_agent(states)
        q1_pi = critic_q1(states, policy_actions)
        q2_pi = critic_q2(states, policy_actions)
        min_q_pi = torch.min(q1_pi, q2_pi)

        value_target = (min_q_pi - alpha * log_probs.unsqueeze(-1)).detach()
        value_pred = value_net(states)
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

        return {
            "critic_loss": critic_loss.item(),
            "value_loss": value_loss.item(),
            "policy_loss": policy_loss.item(),
            "alpha_loss": alpha_loss.item(),
            "alpha": alpha.item(),
            "entropy": entropy.mean().item(),
            "log_prob": log_probs.mean().item(),
        }

    num_updates_limit = max(1, num_update_steps)
    total_gradient_updates = 0

    local_time = time.localtime()
    timestamp = time.strftime("%m%d%Y_%H%M%S", local_time)

    for epoch_id in tqdm(range(epochs), desc="Epoch"):
        nav_agent.train()
        hunch_llm.train()
        critic_q1.train()
        critic_q2.train()
        value_net.train()
        target_value_net.eval()

        for offset in tqdm(
            range(0, len(data_partitions.train), batch_size),
            desc="Collection",
            leave=False,
        ):
            if total_gradient_updates >= num_updates_limit:
                break
            mini_batch = data_partitions.train[offset : offset + batch_size]

            buffer_metrics = gather_experience_steps(
                env,
                nav_agent,
                hunch_llm,
                replay_buffer,
                mini_batch=mini_batch,
                num_steps=num_simulations_per_ques,
                max_env_steps=max_env_steps,
                pad_token_id=answer_tokenizer.pad_token_id,
                use_random_actions=(total_gradient_updates == 0),
            )

            if wandb_on:
                wandb.log({f"collect/{k}": v for k, v in buffer_metrics.items()})
            for metric_name, metric_value in buffer_metrics.items():
                writer.add_scalar(
                    f"collect/{metric_name}", metric_value, total_gradient_updates
                )

            ########################################
            # Training
            ########################################
            while (
                replay_buffer.is_ready(total_gradient_updates)
                and total_gradient_updates < num_updates_limit
            ):
                sampled_batch = replay_buffer.sample(device)
                update_metrics = sac_update_step(sampled_batch)
                if wandb_on:
                    wandb.log({f"train/{k}": v for k, v in update_metrics.items()})
                for metric_name, metric_value in update_metrics.items():
                    writer.add_scalar(
                        f"train/{metric_name}", metric_value, total_gradient_updates
                    )
                total_gradient_updates += 1

                if total_gradient_updates >= num_updates_limit:
                    break

        logger.info(
            "Epoch %d completed | replay_size=%d | gradient_updates=%d",
            epoch_id,
            len(replay_buffer),
            total_gradient_updates,
        )

        if total_gradient_updates >= num_updates_limit:
            logger.info("Reached gradient update budget; stopping training loop early.")
            break

    writer.close()
    return

    # TODO: Get the rollout working

    # Print Model Parameters + Perhaps some more information
    if verbose:
        print(
            "--------------------------\n"
            "Model Parameters\n"
            "--------------------------"
        )
        for name, param in nav_agent.named_parameters():
            print(name, param.numel(), "requires_grad={}".format(param.requires_grad))

        for name, param in env.named_parameters():
            if param.requires_grad:
                print(
                    name, param.numel(), "requires_grad={}".format(param.requires_grad)
                )

    local_time = time.localtime()
    timestamp = time.strftime("%m%d%Y_%H%M%S", local_time)
    writer = SummaryWriter(
        log_dir=f"runs/mlm/{env.knowledge_graph.model_name.lower()}/{timestamp}/"
    )

    named_param_map = {
        param: name
        for name, param in (
            list(nav_agent.named_parameters())
            + list(env.named_parameters())
            + list(hunch_llm.named_parameters())
        )
    }
    optimizer = torch.optim.Adam(  # type: ignore
        filter(
            lambda p: p.requires_grad,
            list(env.concat_projector.parameters())
            + list(nav_agent.parameters())
            + list(hunch_llm.embedding_translator.parameters()),
        ),
        lr=learning_rate,
    )

    modules_to_log: List[nn.Module] = [nav_agent]

    # Variable to pass for logging
    batch_count = 0
    bos_token_id = answer_tokenizer.bos_token_id
    eos_token_id = answer_tokenizer.eos_token_id
    pad_token_id = answer_tokenizer.pad_token_id
    if bos_token_id is None or eos_token_id is None or pad_token_id is None:
        raise ValueError(
            "Assumptions Wrong. The answer_tokenize must have a bos_token_id, eos_token_id and pad_token_id"
        )

    # Replacement for the hooks
    if track_gradients:
        grad_logger = torch_module_logging.ModuleSupervisor(
            {"navigation_agent": nav_agent, "hunch_llm": hunch_llm}
        )

    ########################################
    # Epoch Loop
    ########################################
    for epoch_id in tqdm(range(epochs), desc="Epoch"):

        logger.info("Epoch {}".format(epoch_id))
        # TODO: Perhaps evaluate the epochs?

        # Set in training mode
        nav_agent.train()

        ##############################
        # Batch Loop
        ##############################
        # TODO: update the parameters.
        for sample_offset_idx in tqdm(
            range(0, len(train_data), batch_size), desc="Training Batches", leave=False
        ):
            mini_batch = train_data[sample_offset_idx : sample_offset_idx + batch_size]

            assert isinstance(
                mini_batch, pd.DataFrame
            )  # For the lsp to give me a break

            ########################################
            # Evaluation
            ########################################
            "For debugging purposes, comment back in if needed"
            # if batch_count % mbatches_b4_eval == 0:
            #     evaluate_training(
            #         env,
            #         dev_df,
            #         nav_agent,
            #         hunch_llm,
            #         steps_in_episode,
            #         batch_size_dev,
            #         batch_count,
            #         verbose,
            #         visualize,
            #         writer,
            #         question_tokenizer,
            #         answer_tokenizer,
            #         wandb_on,
            #         iteration = epoch_id * (len(train_data) // batch_size // mbatches_b4_eval) + (batch_count // mbatches_b4_eval),
            #         timestamp = timestamp,
            #     )

            ########################################
            # Training
            ########################################
            "Forward pass"

            optimizer.zero_grad()
            pg_loss, _ = batch_loop(
                env,
                mini_batch,
                nav_agent,
                hunch_llm,
                steps_in_episode,
                bos_token_id,
                eos_token_id,
                pad_token_id,
            )

            if torch.isnan(pg_loss).any():
                logger.error("NaN detected in the loss. Aborting training.")

            # Logg the mean, std, min, max of the rewards
            reinforce_terms_mean = pg_loss.mean()
            reinforce_terms_mean_item = reinforce_terms_mean.item()
            reinforce_terms_std_item = pg_loss.std().item()
            reinforce_terms_min_item = pg_loss.min().item()
            reinforce_terms_max_item = pg_loss.max().item()
            logger.debug(
                f"Reinforce terms mean: {reinforce_terms_mean_item}, std: {reinforce_terms_std_item}, min: {reinforce_terms_min_item}, max: {reinforce_terms_max_item}"
            )

            # TODO: Uncomment and try: (but comment out the normalization in batch_loop and bacth_loop_dev)
            # pg_loss = tensor_normalization(pg_loss)

            # ---------------------------------
            "Backward pass"
            logger.debug("Bout to go backwords")
            reinforce_terms_mean.backward()

            # ---------------------------------
            "Gradient Tracking"

            if sample_offset_idx == 0:

                # Ask for the DAG to be dumped
                if track_gradients:
                    grad_logger.dump_visual_dag(destination_path=f"./figures/grads/dag_{epoch_id:02d}.png", figsize=(10, 100))  # type: ignore

            if torch.all(nav_agent.mu_layer.weight.grad == 0):
                logger.warning("Gradients are zero for mu_layer!")

            # Inspecting vanishing gradient
            if sample_offset_idx % num_batches_till_eval == 0 and verbose:
                # Retrieve named parameters from the optimizer
                named_params = [
                    (named_param_map[param], param)
                    for group in optimizer.param_groups
                    for param in group["params"]
                ]

                # Wandb hisotram of modules
                histograms = histogram_all_modules(modules_to_log, num_buckets=20)
                # Report the histograms to wandb
                if wandb_on:
                    for name, histogram in histograms.items():
                        wandb.log(
                            {
                                f"{name}/Histogram": wandb.Histogram(
                                    np_histogram=histogram
                                )
                            }
                        )

                # Iterate and calculate gradients as needed
                for name, param in named_params:
                    if (
                        param.requires_grad
                        and ("bias" not in name)
                        and (param.grad is not None)
                    ):
                        if name == "weight":
                            name = "concat_projector.weight"
                        grads = param.grad.detach().cpu()
                        weights = param.detach().cpu()

                        dist_tracker.write_dist_parameters(
                            grads, name, "Gradient", writer, epoch_id
                        )
                        dist_tracker.write_dist_parameters(
                            weights, name, "Weights", writer, epoch_id
                        )

                        if wandb_on:
                            wandb.log(
                                {
                                    f"{name}/Gradient": wandb.Histogram(
                                        grads.numpy().flatten()
                                    )
                                }
                            )
                            wandb.log(
                                {
                                    f"{name}/Weights": wandb.Histogram(
                                        weights.numpy().flatten()
                                    )
                                }
                            )
                        elif visualize:
                            dist_tracker.write_dist_histogram(
                                grads.numpy().flatten(),
                                name,
                                "g",
                                "Gradient Histogram",
                                "Grad Value",
                                "Frequency",
                                writer,
                                epoch_id,
                            )
                            dist_tracker.write_dist_histogram(
                                weights.numpy().flatten(),
                                name,
                                "b",
                                "Weights Histogram",
                                "Weight Value",
                                "Frequency",
                                writer,
                                epoch_id,
                            )

            if wandb_on:
                loss_item = pg_loss.mean().item()
                logger.info(f"Submitting train/pg_loss: {loss_item} to wandb")
                wandb.log({"train/pg_loss": loss_item})

            # ---------------------------------
            "Optimizer step"

            optimizer.step()

            batch_count += 1

        "Evaluate at the end of the epoch"
        evaluate_training(
            env,
            dev_df,
            nav_agent,
            hunch_llm,
            steps_in_episode,
            batch_size_dev,
            batch_count,
            verbose,
            visualize,
            writer,
            question_tokenizer,
            answer_tokenizer,
            wandb_on,
            iteration=epoch_id * (len(train_data) // batch_size // mbatches_b4_eval)
            + (batch_count // mbatches_b4_eval),
            timestamp=timestamp,
        )

        # !TODO: Add evaluation metrics for the model's performance at the end of epoch with dev set
    # !TODO: Add evaluation metrics for the model's performance at the end of epoch with test set


# TODO: Remove if unused
def initialize_path(questions: torch.Tensor):
    # Questions must be turned into queries
    raise NotImplementedError


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


def calculate_llm_reward_supasoft(
    hunch_llm: nn.Module,
    obtained_state: torch.Tensor,
    answer_embedding: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if obtained_state.dim() == 1:
        obtained_state = obtained_state.unsqueeze(0)
    if answer_embedding.dim() == 1:
        answer_embedding = answer_embedding.unsqueeze(0)

    with torch.no_grad():
        _, bert_ans_embeddings = hunch_llm(
            graph_embeddings=obtained_state,
            decoder_input_ids=answer_embedding,
        )

    mse_loss = F.mse_loss(
        bert_ans_embeddings,
        answer_embedding,
        reduction="none",
    ).mean(dim=-1)
    reward = -mse_loss
    return reward, bert_ans_embeddings

@torch.no_grad()
def collect_transitions_per_question(
    env: ReinforcedUnsupervisedEnv,
    actor: ContinuousPolicyGradient,
    hunch_llm: nn.Module,
    *,
    num_simulations: int,
    question_embeddings: torch.Tensor,
    answer_bert_embeddings: torch.Tensor,
    path: List[int],
    max_env_steps: int,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, float]]:
    """Roll a single question through the environment and return transitions for replay."""

    assert (
        question_embeddings.dim() == 1
    ), "collect_transitions_per_question() only takes a single question at a time. You are sending a 2d tensor."
    assert (
        answer_bert_embeddings.dim() == 1
    ), "collect_transitions_per_question() only takes a single answer at a time. You are sending a 2d tensor."

    ########################################
    # Input Processing
    ########################################

    device = question_embeddings.device
    action_dim = actor.mu_layer.out_features

    path_answer_id = torch.Tensor(path[-1]).to(torch.long).to(device)

    # Tile questions and answers.
    question_embeddings = question_embeddings.view(1,-1).repeat(num_simulations, 1)
    answer_embeddings = answer_bert_embeddings
    path_answer_id = path_answer_id.view(1,-1).repeat(num_simulations, 1)

    # Normalize any provided action overrides to [steps, batch, action_dim]
    # TODO: Look into this, if we want to enforce an overriding action. For now we comment it out
    # overrides_tensor: Optional[torch.Tensor]
    # overrides_tensor = None
    # if action_overrides is not None:
    #     if action_overrides.dim() == 2:
    #         expected_shape = (num_simulations, action_dim)
    #         if tuple(action_overrides.shape) != expected_shape:
    #             raise ValueError(
    #                 "action_overrides must have shape "
    #                 f"{expected_shape}, got {tuple(action_overrides.shape)}"
    #             )
    #             #TODO: This is weird. I cannto think of a reason we would want to squeeze this out.
    #         overrides_tensor = action_overrides.unsqueeze(0).to(device)
    #     else:
    #         raise ValueError("action_overrides must be rank 2 or 3")


    observations = env.reset(question_embeddings).to(device) # Our current approach doesnt provide any info to reseting the environment
    observations = torch.cat((observations, question_embeddings), dim=-1)

    ########################################
    # Some Data Prepping
    ########################################
    prev_state = observations
    # if prev_state.dim() < 2:
    #     prev_state = prev_state.unsqueeze(0)
    state_shape = prev_state.shape[1:]
    action_shape: Tuple[int, ...] = (action_dim,)

    metrics: DefaultDict[str, List[float]] = defaultdict(list)

    stored_states: List[torch.Tensor] = []
    stored_step_no: List[torch.Tensor] = []
    stored_actions: List[torch.Tensor] = []
    stored_rewards: List[torch.Tensor] = []
    stored_next_states: List[torch.Tensor] = []
    stored_dones: List[torch.Tensor] = []

    extra_rewards_env: List[torch.Tensor] = []
    extra_rewards_llm: List[torch.Tensor] = []
    extra_entropy: List[torch.Tensor] = []
    extra_logprob: List[torch.Tensor] = []

    active_mask = torch.ones(num_simulations, dtype=torch.bool, device=device)
    steps_taken = 0

    ########################################
    # Start Collecting Experience
    ########################################
    for step_idx in range(max_env_steps):
        if not active_mask.any():
            break

        # TODO: unsure if I want overrides here.
        # if overrides_tensor is not None:
        #     override_index = min(step_idx, overrides_tensor.shape[0] - 1)
        #     actions = overrides_tensor[override_index]
        #     log_prob = None
        #     entropy = None
        # else:
        #     actions, log_prob, entropy, _, _ = actor(prev_state)


        ########################################
        # Take Action, Step in Environment
        ########################################
        actions, log_prob, entropy, _, _ = actor(prev_state)

        # TODO: Ensure the active_mask mechanism is working properly, or even necessary.
        if not active_mask.all():
            actions = actions.clone()
            actions[~active_mask] = 0
        step_observation = ReinforcedUnsupervisedEnv.RUE_Observation(
            state=prev_state, answer_id=path_answer_id
        )

        next_observation, extrinsic_reward, done_flags = env.step(step_observation, actions)
        next_state = next_observation

        if next_state.dim() < 2:
            next_state = next_state.unsqueeze(0)

        ########################################
        # Calculate Reward
        ########################################
        # llm_input = next_state.unsqueeze(1) if next_state.dim() == 2 else next_state
        # llm_reward_tokens, _ = calculate_llm_reward_supasoft(
        llm_scalar_reward, _ = calculate_llm_reward_supasoft(
            hunch_llm,
            next_state,
            answer_bert_embeddings,
        )
        # llm_scalar_reward = (llm_reward_tokens * token_mask_float).sum(dim=1) / token_counts


        ########################################
        # Record Results
        ########################################
        extrinsic_scalar = extrinsic_reward.squeeze(-1)
        total_reward = extrinsic_scalar + llm_scalar_reward

        active_prev_state = prev_state[active_mask].detach()
        if active_prev_state.numel() == 0:
            break

        active_actions = actions[active_mask].detach()
        active_rewards = total_reward[active_mask].unsqueeze(-1).detach()
        active_next_states = next_state[active_mask].detach()
        active_done = done_flags[active_mask].detach()

        stored_states.append(active_prev_state)
        stored_step_no.append(torch.full_like(stored_states,step_idx))
        stored_actions.append(active_actions)
        stored_rewards.append(active_rewards)
        stored_next_states.append(active_next_states)
        stored_dones.append(active_done)

        extra_rewards_env.append(extrinsic_scalar[active_mask].unsqueeze(-1).detach())
        extra_rewards_llm.append(llm_scalar_reward[active_mask].unsqueeze(-1).detach())

        metrics["reward/env_mean"].append(extrinsic_scalar[active_mask].mean().item())
        metrics["reward/llm_mean"].append(llm_scalar_reward[active_mask].mean().item())
        metrics["reward/total_mean"].append(total_reward[active_mask].mean().item())
        metrics["done_ratio"].append(done_flags[active_mask].float().mean().item())

        if log_prob is not None:
            extra_logprob.append(log_prob[active_mask].unsqueeze(-1).detach())
            metrics["policy/logprob_mean"].append(log_prob[active_mask].mean().item())
        if entropy is not None:
            extra_entropy.append(entropy[active_mask].unsqueeze(-1).detach())
            metrics["policy/entropy_mean"].append(entropy[active_mask].mean().item())

        active_mask = active_mask & (~done_flags.squeeze(-1))
        prev_state = next_state
        # TODO: This feels a bit forced, think about it. Perhaps project it if you want to stick with it.
        prev_state = torch.cat((prev_state, question_embeddings), dim=-1)
        steps_taken += 1

    # Edge Case: For when we really didnt capture anything
    if not stored_states:
        empty = {
            "states": torch.empty((0,) + state_shape, device=device),
            "steps_no": torch.empty((0,) + state_shape, device=device),
            "actions": torch.empty((0,) + action_shape, device=device),
            "rewards": torch.empty((0, 1), device=device),
            "next_states": torch.empty((0,) + state_shape, device=device),
            "dones": torch.empty((0, 1), dtype=torch.bool, device=device),
            "extras": {},
        }
        summary = {k: 0.0 for k in metrics}
        summary["num_samples"] = 0
        summary["num_env_steps"] = 0
        summary["num_transitions"] = 0
        assert False, "This stored_states thing ought to never happen right?"
        return empty, summary

    transitions = {
        "states": torch.cat(stored_states, dim=0),
        "steps_no": torch.cat(stored_step_no, dim=0),
        "actions": torch.cat(stored_actions, dim=0),
        "rewards": torch.cat(stored_rewards, dim=0),
        "next_states": torch.cat(stored_next_states, dim=0),
        "dones": torch.cat(stored_dones, dim=0),
        "extras": {
            "reward_env": torch.cat(extra_rewards_env, dim=0),
            "reward_llm": torch.cat(extra_rewards_llm, dim=0),
        },
    }
    transitions_extras = {}

    if extra_entropy:
        transitions_extras["entropy"] = torch.cat(extra_entropy, dim=0)
    if extra_logprob:
        transitions_extras["log_prob"] = torch.cat(extra_logprob, dim=0)

    summary = {k: (sum(v) / len(v)) for k, v in metrics.items() if v}
    summary["num_samples"] = batch_size
    summary["num_env_steps"] = steps_taken
    summary["num_transitions"] = transitions["states"].size(0)

    return transitions, transitions_extras, summary


@torch.no_grad()
def gather_experience_steps(
    env: ReinforcedUnsupervisedEnv,
    actor: ContinuousPolicyGradient,
    hunch_llm: nn.Module,
    replay_buffer: QuestionReplayBuffer,
    *,
    mini_batch: pd.DataFrame,
    num_steps: int,
    max_env_steps: int,
    pad_token_id: int,
    use_random_actions: bool = False,
) -> Dict[str, float]:
    """Collect ``num_steps`` transitions by sampling questions with replacement."""

    device = next(actor.parameters()).device
    metrics: DefaultDict[str, List[float]] = defaultdict(list)
    summary: Dict[str, float] = {}

    question_ids: List[int] = mini_batch.index.tolist()

    question_tokens = [
        torch.as_tensor(seq, dtype=torch.long, device=device)
        for seq in mini_batch["enc_questions"].values.tolist()
    ]
    path_sequences: List[List[int]] = mini_batch["triples_ints"].values.tolist()
    answer_embeddings_raw = mini_batch.iloc[:, 3:].to_numpy()

    answer_embeddings = [
        torch.as_tensor(row, dtype=torch.float32, device=device)
        for row in answer_embeddings_raw
    ]

    question_embeddings = env.get_llm_embeddings(question_tokens, device).detach()

    question_data: Dict[int, Dict[str, torch.Tensor]] = {}
    for idx, question_id in enumerate(question_ids):
        question_embedding = question_embeddings[idx]
        answer_embedding = answer_embeddings[idx]
        path = path_sequences[idx]
        if not path:
            raise ValueError("Path sequence is empty; expected at least one entity id")
        answer_entity_id = int(path[-1])
        answer_id_tensor = torch.tensor([answer_entity_id], device=device, dtype=torch.long)

        replay_buffer.ensure_question_metadata(
            question_id,
            question_embedding=question_embedding,
            answer_embedding=answer_embedding,
            answer_id=answer_id_tensor,
        )

        question_data[question_id] = {
            "question_embedding": question_embedding.detach(),
            "answer_embedding": answer_embedding.detach(),
            "answer_id_tensor": answer_id_tensor.detach(),
        }

    sampled_indices = torch.randint(0, len(question_ids), (num_steps,), device=torch.device("cpu"))
    sampled_question_ids = [question_ids[i] for i in sampled_indices.tolist()]
    question_counts = Counter(sampled_question_ids)

    action_dim = actor.mu_layer.out_features
    total_transitions = 0
    max_history = max(1, max_env_steps)

    for question_id, repetitions in question_counts.items():
        question_info = question_data[question_id]

        if not replay_buffer.has_transitions(question_id):
            init_state = env.reset(
                question_info["question_embedding"].unsqueeze(0)
            ).to(device)
            if init_state.dim() == 2:
                init_state = init_state.squeeze(0)
            replay_buffer.add_reset_transition(question_id, init_state, action_dim)
            metrics["init_resets"].append(1.0)

        for _ in range(repetitions):
            current_path_cpu = replay_buffer.get_current_path(question_id)
            if not current_path_cpu:
                init_state = env.reset(
                    question_info["question_embedding"].unsqueeze(0)
                ).to(device)
                if init_state.dim() == 2:
                    init_state = init_state.squeeze(0)
                replay_buffer.set_current_path(question_id, [init_state])
                current_path_cpu = [init_state.detach().cpu()]

            current_path = [state.to(device) for state in current_path_cpu]
            current_state = current_path[-1]
            agent_input = current_state.unsqueeze(0)

            if use_random_actions:
                action = torch.empty(1, action_dim, device=device).uniform_(-1.0, 1.0)
                log_prob = None
                entropy = None
            else:
                action, log_prob, entropy, _, _ = actor(agent_input)

            observation = ReinforcedUnsupervisedEnv.RUE_Observation(
                state=current_state.unsqueeze(0),
                answer_id=question_info["answer_id_tensor"],
            )
            next_state, env_reward, done_flags = env.step(observation, action)

            next_state = next_state.to(device)
            if next_state.dim() == 1:
                next_state = next_state.unsqueeze(0)
            env_reward = env_reward.to(device)
            done_flags = done_flags.to(device)

            llm_reward, _ = calculate_llm_reward_supasoft(
                hunch_llm,
                next_state,
                question_info["answer_embedding"],
            )

            env_reward_scalar = env_reward.view(-1)
            llm_reward_scalar = llm_reward.view(-1)
            total_reward = env_reward_scalar + llm_reward_scalar
            done_scalar = done_flags.view(-1)

            next_state_entity = next_state.squeeze(0)
            new_path_states = current_path + [next_state_entity]

            transition = Transition(
                state=agent_input.squeeze(0).detach().cpu(),
                action=action.squeeze(0).detach().cpu(),
                reward=total_reward.detach().cpu(),
                next_state=next_state_entity.detach().cpu(),
                done=done_scalar.detach().cpu(),
                path_states=[state.detach().cpu() for state in new_path_states],
                step_index=len(new_path_states) - 1,
                env_reward=env_reward_scalar.detach().cpu(),
                llm_reward=llm_reward_scalar.detach().cpu(),
                log_prob=log_prob.detach().cpu() if log_prob is not None else None,
                entropy=entropy.detach().cpu() if entropy is not None else None,
            )
            replay_buffer.add_transition(question_id, transition)

            trimmed_path = new_path_states[-max_history:]
            if done_scalar.any():
                replay_buffer.reset_current_path(question_id)
            else:
                replay_buffer.set_current_path(question_id, trimmed_path)

            metrics["reward/env_mean"].append(env_reward_scalar.mean().item())
            metrics["reward/llm_mean"].append(llm_reward_scalar.mean().item())
            metrics["reward/total_mean"].append(total_reward.mean().item())
            metrics["done_ratio"].append(done_scalar.float().mean().item())
            metrics["path/length_mean"].append(float(len(new_path_states)))

            if log_prob is not None:
                metrics["policy/logprob_mean"].append(log_prob.mean().item())
            if entropy is not None:
                metrics["policy/entropy_mean"].append(entropy.mean().item())

            total_transitions += 1

    summary.update({
        metric: float(sum(values) / len(values))
        for metric, values in metrics.items()
        if values
    })
    summary["num_transitions"] = float(total_transitions)
    summary["num_questions_sampled"] = float(len(question_counts))
    summary["used_random_actions"] = float(use_random_actions)

    return summary


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
    for k,v in pretrained_gtllm_metadata.items():
        # Check if V is an item that can be converted to a string
        logger.info(f"The key {k} has value {v}")

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
    )

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

    # Load the QA Dataset
    # At this point we assume that `pretraining` has run so a cache is waiting for us
    # You should not recalcualte the cache anyways, it is very dependent on how pretraining determined the cache
    # Talk to @ottersome if you need more info
    raw_qadata_path = os.path.join(qna_data_path, "mquake_qna_ds.csv")
    train_df, dev_df, test_df = data_utils.load_cached_pretraining_data(args.pretraining_metadata_cache_path)

    data_partitions = DataPartitions(
        train_df, dev_df, test_df
    )

    if not isinstance(dev_df, pd.DataFrame) or not isinstance(train_df, pd.DataFrame):
        raise RuntimeError(
            "The data was not loaded properly. Please check the data loading code."
        )

    # Get the Module for Approximate Nearest Neighbor Search
    ########################################
    # Setup the ann index.
    # Will be needed for obtaining observations.
    ########################################

    logger.info(":: Setting up the ANN Index")

    ########################################
    # Setup the Vector Searchers
    ########################################
    # TODO: Improve the ANN index manager for rotational models
    if ge_geom == "pRotatE":  # for rotational kge models
        ann_index_manager_ent = ANN_IndexMan_pRotatE(
            kge_model.get_all_entity_embeddings_wo_dropout(),
            embedding_range=kge_model.embedding_range.item(),
        )
        ann_index_manager_rel = ANN_IndexMan_pRotatE(
            kge_model.get_all_relations_embeddings_wo_dropout(),
            embedding_range=kge_model.embedding_range.item(),
        )
    else:  # for non-rotational kge models
        ann_index_manager_ent = ANN_IndexMan(
            kge_model.get_all_entity_embeddings_wo_dropout(),
            exact_computation=True,
            nlist=100,
        )
        ann_index_manager_rel = ANN_IndexMan(
            kge_model.get_all_relations_embeddings_wo_dropout(),
            exact_computation=True,
            nlist=100,
        )

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

    question_replay_buffer = QuestionReplayBuffer(
        num_questions = len(train_df), # TODO: check this is correct
        state_shape=dim_entity,
        action_shape=dim_relation,
        experiences_per_question=args.experiences_per_question,
        batch_size=args.batch_size,
        question_ids=train_df.index.tolist(),
    )

    env = ReinforcedUnsupervisedEnv(
        question_embedding_module=question_embedding_module,
        knowledge_graph=kge_model,
        nav_start_emb_type=args.nav_start_emb_type,
        reached_destination_threshold=args.reached_destination_threshold,
        replay_buffer=question_replay_buffer,
    )

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
        max_path_length=args.max_env_steps * 2 -1, # We have to account for actions too
        encoder_num_layers=args.num_enc_layers,
        encoder_num_heads=args.num_enc_heads,
        encoder_dropout=args.enc_dropout,
        log_std_min=args.log_std_min,
        log_std_max=args.log_std_max,
    ).to(args.device)

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

    train_multihopkg(
        epochs=args.epochs,
        batch_size=args.batch_size,
        batch_size_dev=args.batch_size_dev,
        nav_agent=nav_agent,
        hunch_llm=hunch_llm,
        learning_rate=args.learning_rate,
        env=env,
        data_partitions=data_partitions,
        replay_buffer=question_replay_buffer,
        dev_df=dev_df,
        mbatches_b4_eval=args.batches_b4_eval,
        verbose=args.verbose,
        visualize=args.visualize,
        question_tokenizer=gtllm_tokenizer,
        answer_tokenizer=gtllm_tokenizer,
        track_gradients=args.track_gradients,
        num_batches_till_eval=args.num_batches_till_eval,
        num_simulations_per_ques=args.experiences_per_question,
        wandb_on=args.wandb,
        num_update_steps=args.num_update_steps,
        max_env_steps = args.max_env_steps
    )
    logger.info("Done with everything. Exiting...")

    # TODO: Evaluation of the model
    # metrics = inference(lf)


if __name__ == "__main__":
    main()
