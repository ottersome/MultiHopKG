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
        max_env_steps: int, 
        *,
        question_ids: Sequence[int],
        warmup_size: int = 0,
        min_update_steps: int = 0,
        dtype: torch.dtype = torch.float32,
        reward_dtype: torch.dtype = torch.float32,
    ) -> None:
        self.num_questions = num_questions
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

        self.capacity = num_questions * experiences_per_question
        self._total_size = 0

        ########################################
        # Allocate Memory
        ########################################
        self.write_ptr = torch.zeros(self.num_questions, dtype=torch.long)
        self.read_ptr = torch.zeros(self.num_questions, dtype=torch.long)
        self.count = torch.zeros(self.num_questions, dtype=torch.long)

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

    def get_max_path_len(self):
        return self.max_env_steps * 2 - 1

    def add_transitions(
        self,
        questions_ids: torch.Tensor,   # [E]
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

        qids, qids_count = torch.unique(questions_ids, return_counts=True)
        start = self.write_ptr[qids] # [E]
        start_tiled = self.write_ptr[questions_ids] # [E]
        arange_N = torch.concat([
            torch.arange(qid_count, device=device)      # [N]
            for qid_count in qids_count
        ]) 
        exp_ids = start_tiled + arange_N % cap

        # Write into buffer (parallelized)
        # TODO: We will likely want to remove cur_states as it may covered by path_states
        self.quest_bert_emb[questions_ids, exp_ids] = quest_bert_emb.to(device)
        self.cur_states[questions_ids, exp_ids] = cur_states.to(device)
        self.actions[questions_ids, exp_ids] = actions.to(device)
        self.rewards[questions_ids, exp_ids] = rewards.to(device)
        self.next_states[questions_ids, exp_ids] = next_states.to(device)
        self.done[questions_ids, exp_ids] = dones.to(device)
        self.path_states[questions_ids, exp_ids] = path_states.to(device)
        self.log_prob[questions_ids, exp_ids] = log_probs.to(device)
        self.entropy[questions_ids, exp_ids] = entropies.to(device)
        self.step_counter[questions_ids, exp_ids] = step_counter.to(device)

        # Advance write pointer
        self.write_ptr[qids] = (start + qids_count) % cap

        # Update counters and track overwrites to maintain FIFO semantics
        # qids_list = qids.tolist()
        # q_counts_list = qids_count.tolist()
        # for qid, qcount in zip(qids_list, q_counts_list):
        #     current_count = self.count[qid].item()
        #     new_count = current_count + qcount
        #     if new_count > cap:
        #         overflow = new_count - cap
        #         self.read_ptr[qid] = (self.read_ptr[qid] + overflow) % cap
        #         self.count[qid] = cap
        #     else:
        #         self.count[qid] = new_count

        # self._total_size = int(self.count.sum().item())

    def __len__(self) -> int:
        return self._total_size

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

    def get_oldest_experiences(self, question_counts: Dict[int, int], device: torch.device)\
            -> Tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor,]:
        
        # Ensure we don't get more requests then we can serve
        qids, qid_counts = zip(*question_counts.items())
        assert all([qid_count <= self.experiences_per_question for qid_count in qid_counts]), \
            f"Cannot sample more than {self.experiences_per_question} experiences from each question"

        # Generate idxs to sample 
        qids_idxs = []
        exp_idxs = []
        cap = self.experiences_per_question
        for qid, qid_count in zip(qids, qid_counts):
            qids_idxs += [qid] * qid_count
            _exp_idxs = torch.arange(self.write_ptr[qid].item(), (self.write_ptr[qid].item() + qid_count) % cap)
            exp_idxs.append(_exp_idxs)
        experiences_idxs = torch.concat(exp_idxs)
        qids_idxs = torch.Tensor(qids_idxs).to(torch.long)

        return (
            qids_idxs.to(device),
            experiences_idxs.to(device),
            self.quest_bert_emb[qids_idxs, experiences_idxs].clone().to(device),
            self.path_states[qids_idxs, experiences_idxs].clone().to(device),
            self.actions[qids_idxs, experiences_idxs].clone().to(device),
            self.next_states[qids_idxs, experiences_idxs].clone().to(device),
            self.step_counter[qids_idxs, experiences_idxs].clone().to(device),
            self.done[qids_idxs, experiences_idxs].clone().to(device),
        )
    
    # def pop_oldest_batch(self, question_ids: Sequence[int]) -> Tuple[torch.Tensor, ...]:
    #
    #     cap = self.experiences_per_question
    #     buffer_indices: List[int] = []
    #     slot_indices: List[int] = []
    #
    #     for qid in question_ids:
    #         buf_idx = qid
    #         if self.count[buf_idx] <= 0:
    #             raise ValueError("There should be no empty replay buffers. Theres a severe logic error.")
    #         buffer_indices.append(buf_idx)
    #         slot_indices.append(int(self.read_ptr[buf_idx].item()))
    #
    #     buf_idx_tensor = torch.tensor(buffer_indices, dtype=torch.long)
    #     slot_tensor = torch.tensor(slot_indices, dtype=torch.long)
    #
    #     # Advance read pointer and decrease counts to emulate FIFO pop
    #     for buf_idx in buf_idx_tensor.tolist():
    #         self.read_ptr[buf_idx] = (self.read_ptr[buf_idx] + 1) % cap
    #         if self.count[buf_idx] > 0:
    #             self.count[buf_idx] -= 1
    #
    #     self._total_size = int(self.count.sum().item())
    #
    #     return (
    #         buf_idx_tensor,
    #         self.quest_bert_emb[buf_idx_tensor, slot_tensor].clone(),
    #         self.path_states[buf_idx_tensor, slot_tensor].clone(),
    #         self.actions[buf_idx_tensor, slot_tensor].clone(),
    #         self.next_states[buf_idx_tensor, slot_tensor].clone(),
    #         self.step_counter[buf_idx_tensor, slot_tensor].clone(),
    #         self.done[buf_idx_tensor, slot_tensor].clone(),
    #         self.step_counter[buf_idx_tensor].clone()
    #     )
