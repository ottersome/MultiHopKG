#!/usr/bin/env python3

"""
Copyright (c) 2018, salesforce.com, inc.
For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

Experiment Portal.
"""

import argparse
import copy
import logging
import os
import pickle
import random
import time
import math
from collections import Counter, deque
from typing import Any, Dict, List, Optional, Sequence, Tuple

import debugpy
import numpy as np
import pandas as pd
from multihopkg.datasets import GraphEmbeddingDataset
from multihopkg.utils.data_structures import DataPartitions
from multihopkg.logging import setup_logger
import torch
import torch.nn.functional as F
from rich import traceback
from aim import Run

# PCA
from torch import nn
from tqdm import tqdm
from transformers import (
    AutoModel,
    AutoTokenizer,
    BartConfig,
    PreTrainedTokenizer,
)

import multihopkg.data_utils as data_utils
import wandb
from multihopkg.exogenous.sun_models import KGEModel, get_embeddings_from_indices
from multihopkg.models_language.classical import HunchBart
from multihopkg.rl.graph_search.cpg import ContinuousPolicyGradient
from multihopkg.rl.graph_search.sac import GraphCriticQ, GraphCriticV
from multihopkg.rl.graph_search.pn import ReinforcedUnsupervisedEnv
from multihopkg.rl.utils import QuestionReplayBuffer
from multihopkg.run_configs import rl_alpha
from multihopkg.run_configs.common import overload_parse_defaults_with_yaml
from multihopkg.utils.setup import set_seeds
from multihopkg.vector_search import ANN_IndexMan

# torch.backends.cuda.matmul.allow_tf32 = False
# torch.backends.cudnn.allow_tf32 = False
# torch.autograd.set_detect_anomaly(True)
# torch.cuda.set_sync_debug_mode(1)


traceback.install()
wandb_run = None

PATH_PADDING_VALUE = 1
BART_PADDING_VALUE = 1 # TODO:  Need to remove hardcoding on this later


class AimWriter:
    """Minimal adapter to use Aim like TensorBoard's SummaryWriter."""

    def __init__(
        self,
        *,
        repo: str,
        experiment: Optional[str] = None,
        run_name: Optional[str] = None,
    ) -> None:
        os.makedirs(repo, exist_ok=True)
        self._run = Run(experiment=experiment)
        if run_name:
            self._run.name = run_name

    def add_scalar(self, metric_name: str, value: float, step: int) -> None:
        if isinstance(value, torch.Tensor):
            value = value.item()
        elif isinstance(value, np.generic):
            value = value.item()
        context: Dict[str, str] = {}
        aim_metric = metric_name
        if "/" in metric_name:
            prefix, aim_metric = metric_name.split("/", 1)
            if prefix:
                context["subset"] = prefix
        if context:
            self._run.track(value, aim_metric, step=step, context=context)
        else:
            self._run.track(value, aim_metric, step=step)

    def close(self) -> None:
        self._run.close()

    @property
    def run(self) -> Run:
        return self._run


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

# Mostly used globally to test ANN
def ints_to_strs(list_ints: List[int], int_to_mid: Dict[int, Any], mid_to_fin: Dict[Any, str]) -> List[str]:
    return_vals: List[str] = []
    for li in list_ints:
        mid = int_to_mid[li]
        fin = mid_to_fin[mid]
        return_vals.append(fin)
    return return_vals

def _prepare_question_prompts(
    qna_tokens: torch.Tensor,
    ans_masks: torch.Tensor,
    pad_token_id: int,
    bos_token_id: Optional[int],
    eos_token_id: Optional[int],
) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    """
    Extract only the question portion (prior to the first answer token) to use as decoder prompts.
    """
    prompts: List[torch.Tensor] = []
    prompt_lengths: List[int] = []
    device = qna_tokens.device
    seq_len = qna_tokens.shape[1]

    for seq, mask in zip(qna_tokens, ans_masks):
        mask_list = mask.tolist()
        seq_list = seq.tolist()
        try:
            answer_start = mask_list.index(1)
        except ValueError:
            answer_start = len(seq_list)

        answer_start = min(answer_start, seq_len)
        prompt = [
            token
            for token in seq_list[:answer_start]
            if token != pad_token_id and token != bos_token_id and token != eos_token_id
        ]
        if not prompt:
            raise ValueError("Was expecting a prompt in evaluation")
        if bos_token_id is None:
            raise ValueError("Decoder BOS token id must be defined for prompt preparation")
        prompt.append(bos_token_id)
        prompts.append(torch.tensor(prompt, dtype=torch.long, device=device))
        prompt_lengths.append(len(prompt))

    decoder_input_ids = torch.nn.utils.rnn.pad_sequence(
        prompts, batch_first=True, padding_value=pad_token_id
    )
    decoder_attention_mask = (decoder_input_ids != pad_token_id).long()

    return decoder_input_ids, decoder_attention_mask, prompt_lengths



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
    bart_bos_token_id = hunch_llm.tokenizer.bos_token_id
    # Lets Initiate sub buffers for each question

    for i in tqdm(range(0, len(train_df), BATCH_SIZE), "Populating the replay buffer"):
        mini_batch = train_df.iloc[i : i + BATCH_SIZE]
        _inner_batch_size = len(mini_batch)

        # Get questions ready for batch Bart processing
        questions = [torch.Tensor(ques + [bart_bos_token_id]).to(torch.long) for ques in train_df.loc[mini_batch.index, "enc_questions"]]
        padded_questions_tokens = torch.nn.utils.rnn.pad_sequence(
            questions, batch_first=True, padding_value=pad_token_id
        ).to(gpu_device)
        padded_questions_tokens = padded_questions_tokens.view(-1,1,padded_questions_tokens.shape[-1])
        padded_questions_tokens = padded_questions_tokens.repeat(1,num_simulations_per_question,1).squeeze(1)

        # Answers Graph Embeddings
        paths = mini_batch.loc[:, "triples_ints"]
        starting_point_graphemb_idxs = torch.LongTensor([path[0] for path in paths]).to(gpu_device)
        starting_point_graphemb_idxs = starting_point_graphemb_idxs.unsqueeze(1).repeat(1,num_simulations_per_question).view(-1).to(gpu_device)
        answer_graphemb_idxs = torch.LongTensor([path[-1] for path in paths]).to(gpu_device) # TODO: see if we can remove device
        answer_graphemb_idxs = answer_graphemb_idxs.unsqueeze(1).repeat(1,num_simulations_per_question).view(-1).to(gpu_device)

        # This bit is mostly for teacher forcing so that we have a max length to work with 
        nominal_max_path_len = torch.LongTensor([len(path) for path in paths])

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
        # init_states = env.reset(bert_quest_emb)
        # init_states = init_states.detach().to(cpu_device)
        init_states = starting_point_graphemb_idxs
        init_states = get_embeddings_from_indices(env.knowledge_graph.entity_embedding, init_states)
        padded_path = torch.full([init_states.shape[0], max_env_steps*2 + 1, init_states.shape[1]], PATH_PADDING_VALUE, dtype=torch.float, device=gpu_device)
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
            logger = logger,
            nominal_max_path_len= nominal_max_path_len,
        )
        del bert_quest_emb, answer_graphemb_idxs, answer_bert_heuristics, init_states, next_state, llm_reward, combined_reward, action, padded_path
        torch.cuda.empty_cache()
    actor.train()
    # hunch_llm does not really need to be trained here.

    return replay_buffer

@torch.no_grad()
def get_ground_truth_paths(
    *,
    mini_batch: pd.DataFrame,
    env: ReinforcedUnsupervisedEnv,
    max_path_len: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return padded ground-truth entity/action trajectories for each question."""

    knowledge_graph = env.knowledge_graph
    entity_embeddings = knowledge_graph.entity_embedding
    relation_embeddings = knowledge_graph.relation_embedding
    state_dim = entity_embeddings.shape[1]

    batch_size = len(mini_batch)
    gt_paths = torch.full(
        (batch_size, max_path_len, state_dim),
        PATH_PADDING_VALUE,
        device=device,
        dtype=entity_embeddings.dtype,
    )
    gt_step_counts = torch.zeros(batch_size, dtype=torch.long, device=device)
    triples_list = mini_batch["triples_ints"].tolist()
    max_supported_steps = max(0, (max_path_len - 1) // 2)

    for row_idx, discrete_path in enumerate(triples_list):

        entity_ids = torch.tensor(discrete_path[0::2], dtype=torch.long, device=device)
        relation_ids = torch.tensor(discrete_path[1::2], dtype=torch.long, device=device)
        entity_vecs = get_embeddings_from_indices(entity_embeddings, entity_ids)

        _num_steps = max(0, entity_ids.numel() - 1)
        if _num_steps > max_supported_steps:
            raise RuntimeError(f"Error: No support for {_num_steps} steps found in triple_ints in the dataset. Maximum number of steps in this training is {max_supported_steps} ")
        gt_step_counts[row_idx] = _num_steps
        gt_paths[row_idx, 0, :] = entity_vecs[0]

        if _num_steps == 0 or relation_ids.numel() == 0:
            raise ValueError("Got a sample with either no entities or no relations.")

        relation_vecs = get_embeddings_from_indices(
            relation_embeddings, relation_ids[:_num_steps]
        )
        for step in range(_num_steps):
            gt_paths[row_idx, 2 * step + 1, :] = relation_vecs[step]
            gt_paths[row_idx, 2 * step + 2, :] = entity_vecs[step + 1]

    return gt_paths, gt_step_counts


@torch.no_grad()
def hydrate_replay_buffer(
    num_hydration_samples: int,
    env: ReinforcedUnsupervisedEnv,
    actor: ContinuousPolicyGradient,
    hunch_llm: nn.Module,
    replay_buffer: QuestionReplayBuffer,
    train_df: pd.DataFrame,
    pad_token_id: int,
    summary_writer: AimWriter,
    global_step:int, 
) -> int:
    """Generate new transitions by extending oldest trajectories in replay."""

    actor.eval()
    hunch_llm.eval()

    device = next(actor.parameters()).device
    cpu_device = replay_buffer.cur_states.device
    bert_embed_dim = replay_buffer.get_question_bert_emb_dim()
    # max_path_len = replay_buffer.path_states.shape[2] # TOREM: redundant
    state_dim = replay_buffer.path_states.shape[-1] #TODO: CHECK INDEXING
    max_path_len =  replay_buffer.get_max_path_len()
    all_max_num_steps = ((replay_buffer.get_max_nominal_pathlen() - 1 ) // 2).to(device)
    max_experiences_per_question = replay_buffer.get_experiences_per_question()


    # Get Samples
    question_counts = Counter(random.choices(train_df.index, k=num_hydration_samples))
    question_counts = { #TODO: CHECK INDEXING, frankly this might be the culprit. This has added a level of non-determism that is scary
        qid: min(count, max_experiences_per_question)
        for qid, count in question_counts.items()
    } 
    (
        sampled_qidx,
        bert_quest,
        path_states,
        actions,
        next_states,
        step_counter,
        done_flags,
    ) = replay_buffer.get_oldest_experiences(question_counts, device)
    actual_num_experiences = sum(list(question_counts.values()))
    sampled_question_idxs = sampled_qidx.numpy().tolist()

    mini_batch = train_df.loc[sampled_question_idxs]
    
    max_num_steps = all_max_num_steps[mini_batch.index]

    # if gt_paths is not None:
    #     assert step_counter != gt_step_counts, "Sample does not have the same amount of steps as necessary"
    #     # step_counter = torch.minimum(step_counter, gt_step_counts)
    #     prefix_lengths = step_counter * 2 + 1
    #     for row_idx in range(actual_num_experiences):
    #         fill_len = int(prefix_lengths[row_idx].item())
    #         path_states[row_idx, :fill_len, :] = gt_paths[row_idx, :fill_len, :]

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

    initial_ids: List[int] = []
    answer_ids: List[int] = []
    mini_batch_size = len(mini_batch)
    for path in mini_batch["triples_ints"].tolist():
        initial_ids.append(int(path[0]))
        answer_ids.append(int(path[-1]))
    initial_ids_tensor = torch.tensor(initial_ids, dtype=torch.long, device=device)
    answer_ids_tensor = torch.tensor(answer_ids, dtype=torch.long, device=device)

    # if torch.max(step_counter).item() == 3:
    #     debugpy.breakpoint()

    frozen_done_flags = done_flags.clone()
    if frozen_done_flags.any():
        # reset_states = env.reset(bert_quest[frozen_done_flags])
        # reset_states = env.knowledge_graph.entity_embedding[initial_ids_tensor]
        reset_states = get_embeddings_from_indices(env.knowledge_graph.entity_embedding, initial_ids_tensor)
        new_paths = torch.full(
            (mini_batch_size, max_path_len, state_dim),
            PATH_PADDING_VALUE,
            device=device,
            dtype=path_states.dtype,
        )
        new_paths[torch.arange(mini_batch_size, device=device), 0, :] = reset_states
        path_states[frozen_done_flags] = new_paths
        step_counter[frozen_done_flags] = 0
        done_flags[done_flags] = False

    notdone_flags = ~frozen_done_flags
    if frozen_done_flags.any():
        notdone_steps = step_counter[notdone_flags]
        notdone_max_num_steps = max_num_steps[notdone_flags]

        # Advancement from where they were
        path_states[notdone_flags, (notdone_steps*2) + 1] = actions[notdone_flags]
        path_states[notdone_flags, (notdone_steps*2) + 2] = next_states[notdone_flags]

        # Check if its done
        step_counter[notdone_flags]  += 1
        done_flags[notdone_flags] = notdone_steps == notdone_max_num_steps  # TODO: Fix this. AFter it says done something should be done 

    # if torch.max(step_counter) >= 3:
    #     debugpy.breakpoint()

    valid_mask = torch.zeros((actual_num_experiences, max_path_len), dtype=torch.bool)
    for row_idx in range(step_counter.shape[0]):
        valid_mask[row_idx, :step_counter[row_idx] + 1] = True
    graph_state_mask = valid_mask.unsqueeze(1).unsqueeze(2).to(device)
    valid_counts = valid_mask.sum(dim=-1)

    actions, log_probs, entropy, _, _ = actor(
        path_states,
        graph_state_mask=graph_state_mask,
        context_quest_bert_emb=bert_quest,
    )

    row_idx = torch.arange(actual_num_experiences, device=device)
    current_states = path_states[row_idx, valid_counts, :]

    observation = ReinforcedUnsupervisedEnv.RUE_Observation(
        state=current_states,
        answer_id=answer_ids_tensor,
    )

    # Take a Step
    next_states, extrinsic_reward, done = env.step(observation, actions)
    done = done.squeeze()
    # Merge environment termination with replay-buffer saturation
    done_flags = done_flags | done
    done = done_flags

    # Calculate Reward
    # TODO: Confirm this works well
    next_state_path = torch.cat([path_states.clone(), torch.zeros((path_states.shape[0], 2, path_states.shape[-1]), device=device)], dim=1)
    next_state_path[row_idx, (step_counter + 1) * 2 + 1, :] = actions
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
    summary_writer.add_scalar("hydration_llm_reward", combined_reward.mean().item(), global_step)

    # TODO: Truth action and truth entity here
    gt_action_id: List[int] = []
    gt_entity_id: List[int] = [] 
    entity_embeddings = env.knowledge_graph.entity_embedding
    relation_embeddings = env.knowledge_graph.relation_embedding
    entity_device = (
        entity_embeddings.device
        if isinstance(entity_embeddings, torch.Tensor)
        else entity_embeddings.weight.device
    )
    relation_device = (
        relation_embeddings.device
        if isinstance(relation_embeddings, torch.Tensor)
        else relation_embeddings.weight.device
    )
    for elem_idxs in range(len(mini_batch)):
        if done_flags[elem_idxs]: # Just add somethign random. It wont be used
            gt_action_id.append(0)
            gt_entity_id.append(0)
        else:
            path = mini_batch["triples_ints"].iloc[elem_idxs]
            path_step_counter = int(step_counter[elem_idxs].item())
            action_pos = 2 * path_step_counter + 1 # assume: path_step_counter is not zero because it is always increased above by 1
            entity_pos = action_pos + 1
            if entity_pos >= len(path):
                raise IndexError(
                    f"Tried to access step {path_step_counter} for path of length {len(path)} (question idx {mini_batch.index[elem_idxs]})."
                )
            action_id = int(path[action_pos])
            entity_id = int(path[entity_pos])
            if action_id >= relation_embeddings.shape[0] or entity_id >= entity_embeddings.shape[0]:
                debugpy.breakpoint()
                raise IndexError(
                    f"Tried to access action_id {action_id} or entity_id {entity_id} for path of length {len(path)} (question idx {mini_batch.index[elem_idxs]})."
                )
            gt_action_id.append(int(path[action_pos]))
            gt_entity_id.append(int(path[entity_pos]))

    entity_indices = torch.tensor(gt_entity_id, dtype=torch.long, device=entity_device)
    relation_indices = torch.tensor(gt_action_id, dtype=torch.long, device=relation_device)
    assert entity_indices.shape == (256,), f"entity_indices shape is wrong. It is {entity_indices.shape} expected (256)"
    assert relation_indices.shape == (256,), f"relation_indices shape is wrong. It is {relation_indices.shape} expected (256)"
    entity_vecs = get_embeddings_from_indices(entity_embeddings, entity_indices)
    relation_vecs = get_embeddings_from_indices(relation_embeddings, relation_indices)
    assert entity_vecs.shape == (256,500), f"entity_vecs shape is wrong. It is {entity_vecs.shape} expected (256, 500)"
    assert relation_vecs.shape == (256,500), f"relation_vecs shape is wrong. It is {relation_vecs.shape} expected (256, 500)"

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
    
    # logger.info(
    #     # "Hydrating with items (shapes, devices):\n"
    #     f"\t-questions_ids : {sampled_qidx.shape}, {sampled_qidx.device}\n"
    #     f"\t-quest_bert_emb : {bert_quest.shape}, {bert_quest.device}\n"
    #     f"\t-cur_states : {current_states.shape}, {current_states.device}\n"
    #     f"\t-actions : {relation_vecs.shape}, {relation_vecs.device}\n"
    #     f"\t-rewards : {combined_reward.shape}, {combined_reward.device}\n"
    #     f"\t-next_states : {entity_vecs.shape}, {entity_vecs.device}\n"
    #     f"\t-dones : {done.shape}, {done.device}\n"
    #     f"\t-path_states : {path_states.shape}, {path_states.device}\n"
    #     f"\t-log_probs : {log_probs.shape}, {log_probs.device}\n"
    #     f"\t-entropies : {entropy.shape}, {entropy.device}\n"
    #     f"\t-step_counter : {step_counter.shape}, {step_counter.device}\n"
    # )

    replay_buffer.add_transitions(
        questions_ids=sampled_qidx,
        quest_bert_emb=bert_quest.detach().to(cpu_device),
        cur_states=current_states.detach().to(cpu_device),
        actions=relation_vecs.detach().to(cpu_device),
        rewards=combined_reward.detach().to(cpu_device),
        next_states=entity_vecs.detach().to(cpu_device),
        dones=done.squeeze(-1).detach().to(torch.bool).cpu(),
        path_states=path_states.detach().cpu(),
        log_probs=log_probs.detach().cpu(),
        entropies=entropy.detach().cpu(),
        step_counter=step_counter.detach().cpu(),
        logger=logger,
    )

    actor.train()
    hunch_llm.train()

    return path_states.shape[0]

@torch.no_grad()
def evaluate_seq2seq_outputs(
    env: ReinforcedUnsupervisedEnv,
    ann_index_manager_ent: ANN_IndexMan,
    ann_index_manager_rel: ANN_IndexMan,
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
    writer: AimWriter,
    eid2qid: Dict[int, str],
    eid2pid: Dict[int, str],
    qid_to_title: Dict[str, str],
    pid_to_title: Dict[str, str],
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
    bart_model = getattr(hunch_llm, "bart", None)
    bart_config = getattr(bart_model, "config", None)
    decoder_bos_token_id = getattr(bart_config, "decoder_start_token_id", None)
    decoder_eos_token_id = getattr(bart_config, "eos_token_id", None)

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

    # Samples for Humans
    samples_idxs = np.random.choice(len(dataset), 4, replace=False).tolist()
    samples_idxs = deque(sorted(samples_idxs))
    samples_for_humans = []

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
        bos_bart_token_id =  question_tokenizer.bos_token_id
        question_tokens_list_tensor = [torch.tensor(q + [bos_bart_token_id], dtype=torch.long) for q in mini_batch["enc_questions"].tolist()]
        padded_questions = torch.nn.utils.rnn.pad_sequence(question_tokens_list_tensor, batch_first=True, padding_value=bart_pad_token_id).to(device)
        bart_questions_mask = padded_questions != bart_pad_token_id
        answer_tokens_list_tensor = [torch.tensor(ans + [bos_bart_token_id], dtype=torch.long) for ans in mini_batch["enc_answer"].tolist()]
        # padded_answers = torch.nn.utils.rnn.pad_sequence(answer_tokens_list_tensor, batch_first=True, padding_value=bart_pad_token_id).to(device)
        # max_answer_len = max(len(ans) for ans in answer_tokens_list_tensor)

        # TODO: Check if we actually have the answer tokens
        # labels = padded_answers.clone()
        # labels[labels == pad_token_id] = -100
        bert_quest = torch.tensor(mini_batch.iloc[:,3 + bert_dim:].values.tolist(), dtype=torch.float32, device=device)
        bert_ans = torch.tensor(mini_batch.iloc[:,3:3 + bert_dim].values.tolist(), dtype=torch.float32, device=device)

        paths = mini_batch["triples_ints"].tolist()
        answer_entity_ids = torch.tensor([path[-1] for path in paths], dtype=torch.long, device=device)
        question_entity_ids = torch.tensor([path[0] for path in paths], dtype=torch.long, device=device)

        init_states = question_entity_ids
        init_states = get_embeddings_from_indices(env.knowledge_graph.entity_embedding, init_states)
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

            step_counter[active_idx] = step_active + 1
            # Stop rollouts early when the env signals success or we hit the step budget
            done_now = done.squeeze(-1).bool()
            done_mask[active_idx] = done_now | (step_counter[active_idx] >= max_transitions)

        idxs_out_of_range = [step_counter[i] > max_transitions for i in range(len(step_counter))]
        assert not any(idxs_out_of_range), "There should be no id out of range" # TOREM: After debugging for long enough
        # Build per-sample attention mask using actual traversed length (2 * steps + 1)
        encoder_attention_mask = torch.zeros((_batch_size, max_path_len), dtype=torch.bool, device=device)
        for i in range(_batch_size):
            valid_len = int(2 * step_counter[i].item() + 1)
            encoder_attention_mask[i, :valid_len] = True

        translated_embeddings = hunch_llm.embedding_translator(path_trace) # type: ignore
        ########################################
        # TODO: Logit Loss Calculation
        ########################################
        assert isinstance(question_tokenizer.bos_token, str), "Expected question tokenizer to have bos_token and be a string."
        bos_token_id = question_tokenizer.bos_token_id
        assert isinstance(bos_token_id, int)
        qna_tokens, answer_mask = GraphEmbeddingDataset._merge_questions_and_answers(
            question_tokens_list, answer_tokens_list, bart_pad_token_id, bos_token_id
        )
        qna_tokens_tensor = [
            torch.tensor(qna_token, dtype=torch.long, device=device) for qna_token in qna_tokens
        ]
        padded_qna_tokens = torch.nn.utils.rnn.pad_sequence(qna_tokens_tensor, batch_first=True, padding_value=bart_pad_token_id)
        answer_mask_tensors = [ torch.tensor(mask, dtype=torch.long, device=device) for mask in answer_mask]
        padded_answer_masks = torch.nn.utils.rnn.pad_sequence(answer_mask_tensors, batch_first=True, padding_value=0)

        decoder_attention_mask = (padded_qna_tokens != bart_pad_token_id).long()
        decoder_prompt_ids, decoder_prompt_attention_mask, prompt_lengths = _prepare_question_prompts(
            padded_qna_tokens,
            padded_answer_masks,
            bart_pad_token_id,
            decoder_bos_token_id,
            decoder_eos_token_id,
        )
        bart_outputs = hunch_llm.bart( # type:ignore
            inputs_embeds=translated_embeddings,
            attention_mask=encoder_attention_mask,
            decoder_input_ids=padded_qna_tokens,
            decoder_attention_mask=decoder_attention_mask,
            # labels=padded_answers,
            output_hidden_states=True,
        )
        logits = bart_outputs.logits
        # TODO: Do metrics on the bert alignment thing. 
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
        max_prompt_len = max(prompt_lengths) if prompt_lengths else 0
        max_generation_len = max(padded_qna_tokens.shape[1], max_prompt_len + 1)
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

        pred_texts: List[str] = []
        generated_lengths: List[int] = []
        for sample_idx in range(generated_ids.size(0)):
            prompt_len = prompt_lengths[sample_idx]
            answer_candidate = generated_ids[sample_idx][prompt_len:]
            trimmed_tokens: List[int] = []
            for token_id in answer_candidate.tolist():
                if decoder_eos_token_id is not None and token_id == decoder_eos_token_id:
                    break
                if token_id == bart_pad_token_id:
                    continue
                trimmed_tokens.append(token_id)
            generated_lengths.append(len(trimmed_tokens))
            pred_texts.append(
                answer_tokenizer.decode(trimmed_tokens, skip_special_tokens=True).strip()
            )
        ref_texts = answer_tokenizer.batch_decode(answer_tokens_list, skip_special_tokens=True)

        total_sequences += _batch_size
        # success_count += success_flags.sum().item()
        total_steps += step_counter.float().mean().item()
        total_generated_len += sum(generated_lengths)

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
            padded_qna_tokens,
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

        # Samples For Humans collections
        # Pop idxs to set apart
        # Peek into deque if the id is mini_batch ids
        while len(samples_idxs) > 0 and samples_idxs[0] in mini_batch.index:
            sidx = samples_idxs.popleft()
            _local_sidxs = sidx % batch_size
            samples_for_humans.append({
                "question": question_tokenizer.decode(padded_questions[_local_sidxs], skip_special_tokens=True),
                "dataset_id" : sidx,
                "path_states": path_trace[_local_sidxs,:, :],
                "step_counter": step_counter[_local_sidxs],
                "predicted_texts": pred_texts[_local_sidxs],
                "reference_texts": ref_texts[_local_sidxs],
                "ref_paths": mini_batch["triples_ints"].iloc[_local_sidxs]
            })


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

    # Process Metrics for Humans
    for sample in samples_for_humans:
        idx = sample["dataset_id"]
        predicted_text = sample["predicted_texts"]
        reference_text = sample["reference_texts"]
        question = sample["question"]

        # Now the piece of resistance: Ann Finding
        step_counter = sample["step_counter"]
        paths = sample["path_states"][:(step_counter*2+3),:]
        ref_path = sample["ref_paths"]
        entities = paths[0::2,:]
        relations = paths[1::2,:]
        ref_entities = ref_path[0::2]
        ref_relations = ref_path[1::2]
        # use 
        _, entity_indices = ann_index_manager_ent.search(entities,3)
        _, rel_indices = ann_index_manager_rel.search(relations,3)
        firstrank_qids = [eid2qid[ei[0]] for ei in entity_indices]
        firstrank_pids = [eid2pid[ei[0]] for ei in rel_indices]
        ent_titles = [qid_to_title[fq] for fq in firstrank_qids]
        rel_titles = [pid_to_title[fp] for fp in firstrank_pids]

        ref_firstrank_qids = [eid2qid[ei] for ei in ref_entities]
        ref_firstrank_pids = [eid2pid[ei] for ei in ref_relations]
        ref_ent_titles = [qid_to_title[fq] for fq in ref_firstrank_qids]
        ref_rel_titles = [pid_to_title[fp] for fp in ref_firstrank_pids]

        final_path_titles = []
        for i in range(len(ent_titles) + len(rel_titles)):
            if i % 2 == 0:
                final_path_titles += [ent_titles[i//2]]
            else:
                final_path_titles += [rel_titles[i//2]]

        ref_path_titles = []
        for i in range(len(ref_path)):
            if i % 2 == 0:
                ref_path_titles += [ref_ent_titles[i//2]]
            else:
                ref_path_titles += [ref_rel_titles[i//2]]

        logger.info(
            "----------------------------------------\n"
            f"The following is the {idx}th sample.\n"
            f"Question is: {question}\n"
            f"Reference Text is: {reference_text}\n"
            f"Predicted Text is: {predicted_text}\n"
            f"Predicted Path is: {final_path_titles}\n"
            f"Ref Path Path is: {ref_path_titles}\n"
            "----------------------------------------\n"
        )

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
    run_name: str,
    ann_index_manager_ent: ANN_IndexMan,
    ann_index_manager_rel: ANN_IndexMan,
    question_tokenizer: PreTrainedTokenizer,
    answer_tokenizer: PreTrainedTokenizer,
    num_gradupdates_till_eval: int,
    num_simulations_per_ques: int,
    wandb_on: bool,
    num_update_steps: int,
    num_hydration_samples: int,
    critic_q1: GraphCriticQ,
    critic_q2: GraphCriticQ,
    value_net: GraphCriticV,
    eid2qid: Dict[int, str],
    eid2pid: Dict[int, str],
    qid_to_title: Dict[str, str],
    pid_to_title: Dict[str, str],
    teacherforce_reg_lambda: float,
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

    q1_optimizer = torch.optim.Adam(critic_q1.parameters(), lr=learning_rate)
    q2_optimizer = torch.optim.Adam(critic_q2.parameters(), lr=learning_rate)
    value_optimizer = torch.optim.Adam(value_net.parameters(), lr=learning_rate)
    policy_optimizer = torch.optim.Adam(nav_agent.parameters(), lr=learning_rate)

    log_alpha = torch.tensor(
        [math.log(0.2)], device=device, dtype=torch.float32, requires_grad=True
    )
    alpha_optimizer = torch.optim.Adam([log_alpha], lr=learning_rate)
    target_entropy = -float(action_shape[0])
    tau = 0.005
    # tau = 0.1
    bert_dim = replay_buffer.get_question_bert_emb_dim()
    gamma = nav_agent.gamma
    # hydration_interval = int(num_update_steps * 0.005)
    hydration_interval = 10
    updates_since_hydration = 0
    train_df = data_partitions.train
    question_ids = train_df.index.values.tolist()
    assert isinstance(question_ids, List)
    teacher_relation_targets: Dict[int, List[int]] = {}
    for qid, path in train_df["triples_ints"].items():
        if not isinstance(path, Sequence):
            continue
        relations = [int(rel) for rel in path[1::2]]
        if relations:
            teacher_relation_targets[int(qid)] = relations
    eval_interval_updates = max(1, num_gradupdates_till_eval)
    last_eval_updates = 0

    ########################################
    # Helper Functions Setup
    ########################################
    def soft_update(source: nn.Module, target: nn.Module, tau: float) -> None:
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.mul_(1.0 - tau)
            target_param.data.add_(tau * param.data)

    def get_teacher_action_embeddings(
        question_ids_tensor: torch.Tensor, step_counts: torch.Tensor
    ) -> Optional[torch.Tensor]:
        if not teacher_relation_targets:
            return None
        relation_indices: List[int] = []
        qid_list = question_ids_tensor.detach().cpu().tolist()
        step_list = step_counts.detach().cpu().tolist()
        for qid_value, step_value in zip(qid_list, step_list):
            relations = teacher_relation_targets[int(qid_value)]
            relation_pos = int(step_value)
            relation_indices.append(relations[relation_pos])
        if not relation_indices:
            return None
        relation_idx_tensor = torch.tensor(
            relation_indices, dtype=torch.long, device=device
        )
        return get_embeddings_from_indices(
            env.knowledge_graph.relation_embedding, relation_idx_tensor
        ).to(device)

    def sac_update_step(
        question_ids: torch.Tensor,
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

        _batch_size = states_path.shape[0]
        batch_idxs = torch.arange(_batch_size, device=device)
        ########################################
        # Prepare States and corresp. Masks
        ########################################
        # For cur_state, action, next_state
        _states_path = torch.cat([states_path.clone(), torch.full((states_path.shape[0], 2, states_path.shape[-1]), PATH_PADDING_VALUE, device=device)], dim=1)
        _max_path_len = _states_path.shape[1]
        next_states_path = _states_path.clone()
        next_states_path[batch_idxs, 2*step_counter + 1, : ] = actions
        next_states_path[batch_idxs, 2*step_counter + 2, : ] = next_states
        
        # For cur_state, action
        qa_state = _states_path.clone()
        qa_state[batch_idxs, 2*step_counter + 1, : ] = actions

        # Masking
        graph_curState_mask = torch.zeros((_batch_size, _max_path_len), dtype=torch.bool).to(device)
        graph_nextstate_mask = torch.zeros((_batch_size, _max_path_len), dtype=torch.bool).to(device)
        graph_plusAction_mask = torch.zeros((_batch_size, _max_path_len), dtype=torch.bool).to(device)
        for i in range(graph_nextstate_mask.shape[0]):
            graph_curState_mask[i, : 2 * step_counter[i] + 1] = True
            graph_plusAction_mask[i, : 2 * step_counter[i] + 2] = True
            graph_nextstate_mask[i, : 2 * step_counter[i] + 3] = True
        graph_curState_mask = graph_curState_mask.unsqueeze(1).unsqueeze(2)
        graph_plusAction_mask = graph_plusAction_mask.unsqueeze(1).unsqueeze(2)
        graph_nextstate_mask = graph_nextstate_mask.unsqueeze(1).unsqueeze(2)

        ########################################
        # Forward Propagation
        ########################################
        with torch.no_grad():
            target_values = target_value_net(
                next_states_path, graph_nextstate_mask, bert_quest_emb
            ).squeeze()
            dones_float = dones.to(torch.float32)
            q_target = rewards + (1.0 - dones_float) * gamma * target_values

        q1_pred = critic_q1(qa_state, graph_plusAction_mask, bert_quest_emb).squeeze()
        q2_pred = critic_q2(qa_state, graph_plusAction_mask, bert_quest_emb).squeeze()
        q1_loss = F.mse_loss(q1_pred, q_target)
        q2_loss = F.mse_loss(q2_pred, q_target)

        q1_optimizer.zero_grad()
        q2_optimizer.zero_grad()
        q1_loss.backward()
        q2_loss.backward()
        q1_optimizer.step()
        q2_optimizer.step()

        policy_actions, log_probs, entropy, _, _ = nav_agent(
            _states_path,
            graph_state_mask=graph_curState_mask,
            context_quest_bert_emb=bert_quest_emb,
        )
        policy_state = _states_path.clone()
        policy_state[batch_idxs, 2 * step_counter + 1, :] = policy_actions

        teacher_targets = get_teacher_action_embeddings(question_ids, step_counter)
        policy_alignment_metric = 0.0
        buffer_alignment_metric = 0.0

        if teacher_targets is not None:
            teacher_targets = teacher_targets.to(policy_actions.dtype)
            buffer_alignment_metric = (
                F.mse_loss(
                    actions,
                    teacher_targets.detach(),
                    reduction="none",
                )
                .mean(dim=-1)
                .mean()
                .item()
            )

        q1_pi = critic_q1(policy_state, graph_plusAction_mask, bert_quest_emb)
        q2_pi = critic_q2(policy_state, graph_plusAction_mask, bert_quest_emb)
        min_q_pi = torch.min(q1_pi, q2_pi)

        # Min_Q_Pi needs a different sort of action

        value_target = (min_q_pi - alpha * log_probs.unsqueeze(-1)).detach()
        # value_target = (min_q_pi - log_probs.unsqueeze(-1)).detach()
        value_pred = value_net(_states_path, graph_curState_mask, bert_quest_emb)
        value_loss = F.mse_loss(value_pred, value_target)

        value_optimizer.zero_grad()
        value_loss.backward()
        value_optimizer.step()

        policy_loss = (alpha * log_probs.unsqueeze(-1) - min_q_pi).mean()
        # policy_loss = (log_probs.unsqueeze(-1) - min_q_pi).mean()
        if teacher_targets is not None and teacherforce_reg_lambda > 0:
            reg_component = F.mse_loss(
                policy_actions, teacher_targets, reduction="none"
            ).mean(dim=-1)
            policy_loss = policy_loss + teacherforce_reg_lambda * reg_component.mean()
            policy_alignment_metric = reg_component.mean().item()
        policy_optimizer.zero_grad()
        policy_loss.backward()
        policy_optimizer.step()

        alpha_loss = -(log_alpha * (log_probs.detach() + target_entropy)).mean()
        # alpha_loss = -(log_probs.detach() + target_entropy).mean()
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
        mask_token_counts = graph_plusAction_mask.squeeze(1).squeeze(1).sum(dim=-1).float()

        return {
            "q1_loss": q1_loss.item(),
            "q2_loss": q2_loss.item(),
            "value_loss": value_loss.item(),
            "policy_loss": policy_loss.item(),
            # "alpha_loss": alpha_loss.item(),
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
            "teacherforce_policy_alignment": policy_alignment_metric,
            "teacherforce_buffer_alignment": buffer_alignment_metric,
        }

    coverage_sampler = QuestionCoverageSampler(question_ids)

    num_updates_limit = max(1, num_update_steps)
    total_gradient_updates = 0

    local_time = time.localtime()
    timestamp = time.strftime("%m%d%Y_%H%M%S", local_time)
    log_dir = os.path.join(
        "runs",
        "rl_sac",
        env.knowledge_graph.model_name.lower(),
        timestamp,
    )
    writer = AimWriter(
        repo=log_dir,
        experiment="rl_training",
        # experiment=f"rl_sac/{env.knowledge_graph.model_name.lower()}",
        run_name=f"{run_name}-{timestamp}",
    )
    writer.add_scalar("train_config/hydration_interval", hydration_interval, 0)
    if wandb_on:
        wandb.log(
            {
                "train_config/hydration_interval": hydration_interval,
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
                sampled_qids,
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
                sampled_qids,
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
                logger.info( f"Hydrating replay buffer with {num_hydration_samples} transitions")
                added = hydrate_replay_buffer(
                    num_hydration_samples=num_hydration_samples,
                    env=env,
                    actor=nav_agent,
                    hunch_llm=hunch_llm,
                    replay_buffer=replay_buffer,
                    train_df=train_df,
                    pad_token_id=pad_token_id,
                    summary_writer=writer,
                    global_step=total_gradient_updates
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
            "Epoch %d completed | gradient_updates=%d | coverage_cycles=%d",
            epoch_id,
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
            logger.info(f"Evaluating at epoch {epoch_id} with total_gradient_updates={total_gradient_updates}. We do so at intervals of {eval_interval_updates}")
            evaluate_seq2seq_outputs(
                env=env,
                ann_index_manager_ent=ann_index_manager_ent,
                ann_index_manager_rel=ann_index_manager_rel,
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
                eid2qid = eid2qid,
                eid2pid = eid2pid,
                qid_to_title = qid_to_title,
                pid_to_title = pid_to_title,
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
    #questions_masks = (ans_masks == 0) & (padding_mask)
    paths_attention_mask = ~(obtained_state == BART_PADDING_VALUE).all(dim=-1)
    _, bert_ans_embeddings = hunch_llm(
        graph_embeddings=obtained_state,
        encoder_attention_mask=paths_attention_mask,
        decoder_input_ids=question_tokens,
        questions_masks=dec_attention_mask,
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
            f"🪄 Initializing Weights and Biases. Under project name {args.wandb_project_name} and run name {args.run_name}"
        )
        wandb_run = wandb.init(
            project=args.wandb_project_name,
            name=args.run_name,
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
        tokenizer=gtllm_tokenizer,
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
    # DEBUG: Remove these globals after done with debugging
    global id2ent, id2rel, entities_info, relations_info
    id2ent, ent2id, id2rel, rel2id = data_utils.load_dictionaries(qna_data_path)

    # Load the Entity-Rel Info
    entities_info = pd.read_csv(args.path_entities_info, index_col=0)["Title"].to_dict()
    relations_info = pd.read_csv(args.path_relations_info, index_col=0)["Title"].to_dict()

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

    # TODO: Check if the model loaded is anything but TransE (and halt if so)
    # We currently dont support anything but TransE

    ########################################
    # Setup the Vector Searchers
    ########################################
    global ann_index_manager_ent
    global ann_index_manager_rel
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

    # DEBUG: Again, remove this after
    # local_time = time.localtime()
    # timestamp = time.strftime("%m%d%Y_%H%M%S", local_time)
    # writer = SummaryWriter(
    #     log_dir=f"runs/rl_sac/{env.knowledge_graph.model_name.lower()}/{timestamp}/"
    # )
    # evaluate_seq2seq_outputs(
    #     env=env,
    #     ann_index_manager_ent=ann_index_manager_ent,
    #     ann_index_manager_rel=ann_index_manager_rel,
    #     nav_agent=nav_agent,
    #     hunch_llm=hunch_llm,
    #     dataset=data_partitions.validation,
    #     question_tokenizer=gtllm_tokenizer,
    #     answer_tokenizer=gtllm_tokenizer,
    #     batch_size=args.batch_size_dev,
    #     bert_dim=replay_buffer.get_question_bert_emb_dim(),
    #     max_env_steps=replay_buffer.max_env_steps,
    #     bart_pad_token_id=PATH_PADDING_VALUE,
    #     global_step=1,
    #     prefix="dev",
    #     eid2qid=id2ent,
    #     eid2pid=id2rel,
    #     qid_to_title=entities_info,
    #     pid_to_title=relations_info,
    #     writer=writer
    # )
    # exit()
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
        run_name=args.run_name,
        ann_index_manager_ent=ann_index_manager_ent,
        ann_index_manager_rel=ann_index_manager_rel,
        question_tokenizer=gtllm_tokenizer,
        answer_tokenizer=gtllm_tokenizer,
        num_gradupdates_till_eval=args.num_gradupdates_till_eval,
        num_simulations_per_ques=args.experiences_per_question,
        wandb_on=args.wandb,
        num_update_steps=args.num_update_steps,
        num_hydration_samples=args.num_hydration_samples,
        critic_q1=critic_q1,
        critic_q2=critic_q2,
        value_net=value_net,
        eid2qid=id2ent,
        eid2pid=id2rel,
        qid_to_title=entities_info,
        pid_to_title=relations_info,
        teacherforce_reg_lambda=args.teacherforce_reg_lambda,
    )
    logger.info("Done with everything. Exiting...")

    # TODO: Evaluation of the model
    # metrics = inference(lf)


if __name__ == "__main__":
    main()
