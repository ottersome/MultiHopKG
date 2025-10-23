from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Sequence, Tuple
import pandas as pd

import torch


class QuestionReplayBuffer:
    """Replay buffer that keeps per-question sub-buffers."""

    def __init__(
        self,
        num_questions: int,
        state_shape: int,
        action_shape: int,
        bert_emb_dim: int,
        experiences_per_question: int,
        batch_size: int,
        max_env_steps: int, 
        *,
        question_ids: Sequence[int],
        warmup_size: int = 0,
        min_update_steps: int = 0,
        dtype: torch.dtype = torch.float32,
        reward_dtype: torch.dtype = torch.float32,
    ) -> None:
        self.num_questions = num_questions
        self.batch_size = batch_size
        self.warmup_size = warmup_size
        self.max_env_steps = max_env_steps
        self.min_update_steps = min_update_steps
        self.experiences_per_question = experiences_per_question
        self._state_shape = state_shape
        self._action_shape = action_shape
        self._dtype = dtype
        self._reward_dtype = reward_dtype
        self.bert_emb_dim = bert_emb_dim

        assert len(question_ids) == num_questions, "question_ids must match num_questions"
        self._question_ids = list(question_ids)
        self._question_id_to_idx = {
            question_id: idx for idx, question_id in enumerate(self._question_ids)
        }

        self.capacity = num_questions * experiences_per_question
        self._total_size = 0

        ########################################
        # Allocate Memory
        ########################################
        self.write_ptr = torch.zeros(self.num_questions, dtype=torch.long)

        self.quest_bert_emb = torch.zeros((self.num_questions, self.experiences_per_question, self.bert_emb_dim))
        self.cur_states = torch.zeros((self.num_questions, self.experiences_per_question, self._action_shape))
        self.actions = torch.zeros((self.num_questions, self.experiences_per_question, self._action_shape))
        self.rewards = torch.zeros((self.num_questions, self.experiences_per_question))
        self.next_states = torch.zeros((self.num_questions, self.experiences_per_question, self._action_shape))
        self.done = torch.zeros((self.num_questions, self.experiences_per_question), dtype=torch.bool)
        self.path_states = torch.zeros(
            (self.num_questions , self.experiences_per_question, self.max_env_steps * 2 - 1, self._action_shape)
        )
        self.log_prob = torch.zeros((self.num_questions, self.experiences_per_question))
        self.entropy = torch.zeros((self.num_questions, self.experiences_per_question))
        self.step_counter = torch.zeros((self.num_questions, self.experiences_per_question), dtype=torch.long)

    def get_question_bert_emb_dim(self):
        return self.bert_emb_dim

    def qet_question_ids(self):
        return self._question_ids

    def add_transitions(
        self,
        question_n_exp_idxs: torch.Tensor,   # [E]
        quest_bert_emb: torch.Tensor,        # [E, B]
        cur_states: torch.Tensor,            # [E, A]
        actions: torch.Tensor,               # [E, A]
        rewards: torch.Tensor,               # [E,]
        next_states: torch.Tensor,           # [E, A]
        dones: torch.Tensor,                 # [E]
        path_states: torch.Tensor,           # [E, T, A]
        log_probs: torch.Tensor,             # [E]
        entropies: torch.Tensor,             # [E]
        step_counter:  torch.Tensor,         # [E]
        # Where E indexes experience, T indexes path length, and A is action/state shape. B is Bert Pooled Embedding Dim
    ):
        """
        Performs a round-robin write into the replay buffer for multiple questions at once.
        """
        device = self.cur_states.device
        cap = self.experiences_per_question

        qids, qids_count = torch.unique(question_n_exp_idxs, return_counts=True)
        start = self.write_ptr[qids] # [E]
        start_tiled = self.write_ptr[question_n_exp_idxs] # [E]
        arange_N = torch.concat([
            torch.arange(qid_count, device=device)      # [N]
            for qid_count in qids_count
        ]) 
        exp_ids = start_tiled + arange_N % cap

        # Write into buffer (parallelized)
        # TODO: We will likely want to remove cur_states as it may covered by path_states
        self.quest_bert_emb[question_n_exp_idxs, exp_ids] = quest_bert_emb.to(device)
        self.cur_states[question_n_exp_idxs, exp_ids] = cur_states.to(device)
        self.actions[question_n_exp_idxs, exp_ids] = actions.to(device)
        self.rewards[question_n_exp_idxs, exp_ids] = rewards.to(device)
        self.next_states[question_n_exp_idxs, exp_ids] = next_states.to(device)
        self.done[question_n_exp_idxs, exp_ids] = dones.to(device)
        self.path_states[question_n_exp_idxs, exp_ids] = path_states.to(device)
        self.log_prob[question_n_exp_idxs, exp_ids] = log_probs.to(device)
        self.entropy[question_n_exp_idxs, exp_ids] = entropies.to(device)
        self.step_counter[question_n_exp_idxs, exp_ids] = step_counter.to(device)

        # Advance write pointer
        self.write_ptr[qids] = (start + qids_count) % cap

    def __len__(self) -> int:
        return self._total_size

    def is_ready(self, num_updates_done: int) -> bool:
        if self._total_size < max(self.warmup_size, self.batch_size):
            return False
        return num_updates_done >= self.min_update_steps

    def sample_transitions(self, question_counts: Dict[int,int]):
        """
        Will take a dictionary of question ids and count of experiences.
        The experiences themself will be sampled at random.

        Variables (1)
        ---------
        - question_ids (Dict[int,int]): key: question_id, value: experience count.
            Describes which questions to be sampled and what amount of them.

        Returns (1)
        ---------
        TODO:fill this later
        - experiences:
        """

        # Unzip question counts into different variables
        qids, qid_counts = zip(*question_counts.items())
        assert all([qid_count <= self.experiences_per_question for qid_count in qid_counts]), \
            f"Cannot sample more than {self.experiences_per_question} experiences from each question"

        # Generate Idxs to sample 
        qids_idxs = []
        exp_idxs = []
        for qid, qid_count in zip(qids, qid_counts):
            qids_idxs += [qid] * qid_count
            _exp_idxs = torch.randperm(self.experiences_per_question)[:qid_count].to(torch.long)
            exp_idxs.append(_exp_idxs)
        experiences_idxs = torch.concat(exp_idxs)
        qids_idxs = torch.Tensor(qids_idxs).to(torch.long)

        # Sample
        cur_states = self.cur_states[qids_idxs, experiences_idxs]
        quest_bert_emb = self.quest_bert_emb[qids_idxs, experiences_idxs]
        actions = self.actions[qids_idxs, experiences_idxs]
        rewards = self.rewards[qids_idxs, experiences_idxs]
        next_states = self.next_states[qids_idxs, experiences_idxs]
        dones = self.done[qids_idxs, experiences_idxs]
        path_states = self.path_states[qids_idxs, experiences_idxs]
        log_probs = self.log_prob[qids_idxs, experiences_idxs]
        entropies = self.entropy[qids_idxs, experiences_idxs]
        step_counter = self.step_counter[qids_idxs, experiences_idxs]

        return (
            cur_states,
            quest_bert_emb,
            actions,
            rewards,
            next_states,
            dones,
            path_states,
            log_probs,
            entropies,
            step_counter,
        )

    # def sample(self, device: torch.device, batch_size: Optional[int] = None) -> Dict[str, torch.Tensor]:
    #     if batch_size is None:
    #         batch_size = self.batch_size
    #
    #     all_transitions: List[Tuple[int, Transition]] = list(self.iter_all())
    #     if len(all_transitions) < batch_size:
    #         raise ValueError(
    #             "Cannot sample from replay buffer before it holds at least one batch"
    #         )
    #
    #     perm = torch.randperm(len(all_transitions))[:batch_size]
    #
    #     states = []
    #     actions = []
    #     rewards = []
    #     next_states = []
    #     dones = []
    #     env_rewards = []
    #     llm_rewards = []
    #     log_probs: List[torch.Tensor] = []
    #     entropies: List[torch.Tensor] = []
    #     question_embeddings = []
    #     question_ids: List[int] = []
    #     path_states: List[List[torch.Tensor]] = []
    #     step_indices: List[int] = []
    #
    #     for idx in perm.tolist():
    #         question_id, transition = all_transitions[idx]
    #         states.append(transition.state)
    #         actions.append(transition.action)
    #         rewards.append(transition.reward)
    #         next_states.append(transition.next_state)
    #         dones.append(transition.done)
    #         env_rewards.append(transition.env_reward)
    #         llm_rewards.append(transition.llm_reward)
    #         path_states.append(transition.path_states)
    #         step_indices.append(transition.step_index)
    #         question_embeddings.append(self.get_question_embedding(question_id))
    #         question_ids.append(question_id)
    #
    #         if transition.log_prob is not None:
    #             log_probs.append(transition.log_prob)
    #         if transition.entropy is not None:
    #             entropies.append(transition.entropy)
    #
    #     batch = {
    #         "states": torch.stack(states).to(device),
    #         "actions": torch.stack(actions).to(device),
    #         "rewards": torch.stack(rewards).to(device),
    #         "next_states": torch.stack(next_states).to(device),
    #         "dones": torch.stack(dones).to(device),
    #     }
    #
    #     extras: Dict[str, torch.Tensor] = {
    #         "env_reward": torch.stack(env_rewards).to(device),
    #         "llm_reward": torch.stack(llm_rewards).to(device),
    #         "question_embedding": torch.stack(question_embeddings).to(device),
    #         "question_id": torch.tensor(question_ids, dtype=torch.long, device=device),
    #         "step_index": torch.tensor(step_indices, dtype=torch.long, device=device),
    #     }
    #
    #     if log_probs:
    #         extras["log_prob"] = torch.stack(log_probs).to(device)
    #     if entropies:
    #         extras["entropy"] = torch.stack(entropies).to(device)
    #
    #     batch["extras"] = {
    #         **extras,
    #         "path_states": path_states,
    #     }
    #
    #     return batch
