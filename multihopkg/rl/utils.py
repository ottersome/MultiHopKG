from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Sequence, Tuple
import pandas as pd

import torch


@dataclass
class Transition:
    state: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    next_state: torch.Tensor
    done: torch.Tensor
    path_states: torch.Tensor
    step_index: int
    env_reward: torch.Tensor
    llm_reward: torch.Tensor
    log_prob: Optional[torch.Tensor] = None
    entropy: Optional[torch.Tensor] = None


class QuestionReplayBuffer:
    """Replay buffer that keeps per-question sub-buffers."""

    @dataclass
    class QuestionBuffer:
        transitions: Deque[Transition]
        question_embedding: Optional[torch.Tensor] = None
        answer_embedding: Optional[torch.Tensor] = None
        answer_id: Optional[torch.Tensor] = None

    def __init__(
        self,
        num_questions: int,
        state_shape: int,
        action_shape: int,
        experiences_per_question: int,
        batch_size: int,
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
        self.min_update_steps = min_update_steps
        self.experiences_per_question = experiences_per_question
        self._state_shape = state_shape
        self._action_shape = action_shape
        self._dtype = dtype
        self._reward_dtype = reward_dtype

        assert len(question_ids) == num_questions, "question_ids must match num_questions"
        self._question_ids = list(question_ids)
        self._question_id_to_idx = {
            question_id: idx for idx, question_id in enumerate(self._question_ids)
        }

        self.capacity = num_questions * experiences_per_question
        self._total_size = 0


        self._prepopulate(num_questions, experiences_per_question)

    def _prepopulate(self, num_questions, experiences_per_question):
        self.replay_buffer: List[QuestionReplayBuffer.QuestionBuffer] = [
            QuestionReplayBuffer.QuestionBuffer(
                transitions=deque(maxlen=experiences_per_question)
            )
            for _ in range(num_questions)
        ]

    def __len__(self) -> int:
        return self._total_size

    @property
    def is_full(self) -> bool:
        return self._total_size >= self.capacity

    def is_ready(self, num_updates_done: int) -> bool:
        if self._total_size < max(self.warmup_size, self.batch_size):
            return False
        return num_updates_done >= self.min_update_steps

    def resolve_question_idx(self, question_id: int) -> int:
        return self._question_id_to_idx[question_id]


    def ensure_question_metadata(
        self,
        question_id: int,
        *,
        question_embedding: torch.Tensor,
        answer_embedding: torch.Tensor,
        answer_id: torch.Tensor,
    ) -> None:
        idx = self.resolve_question_idx(question_id)
        buffer = self.replay_buffer[idx]
        if buffer.question_embedding is None:
            buffer.question_embedding = question_embedding.detach().cpu()
        if buffer.answer_embedding is None:
            buffer.answer_embedding = answer_embedding.detach().cpu()
        if buffer.answer_id is None:
            buffer.answer_id = answer_id.detach().cpu().to(torch.long)

    def has_transitions(self, question_id: int) -> bool:
        idx = self.resolve_question_idx(question_id)
        return len(self.replay_buffer[idx].transitions) > 0

    def get_last_transition(self, question_id: int) -> Optional[Transition]:
        idx = self.resolve_question_idx(question_id)
        buffer = self.replay_buffer[idx]
        if not buffer.transitions:
            return None
        return buffer.transitions[-1]

    def get_question_embedding(self, question_id: int) -> torch.Tensor:
        idx = self.resolve_question_idx(question_id)
        embedding = self.replay_buffer[idx].question_embedding
        if embedding is None:
            raise RuntimeError(
                f"Question {question_id} does not have an associated embedding"
            )
        return embedding

    def get_answer_embedding(self, question_id: int) -> torch.Tensor:
        idx = self.resolve_question_idx(question_id)
        embedding = self.replay_buffer[idx].answer_embedding
        if embedding is None:
            raise RuntimeError(
                f"Question {question_id} does not have an associated answer embedding"
            )
        return embedding

    def get_answer_id(self, question_id: int) -> torch.Tensor:
        idx = self.resolve_question_idx(question_id)
        answer_id = self.replay_buffer[idx].answer_id
        if answer_id is None:
            raise RuntimeError(
                f"Question {question_id} does not have an associated answer id"
            )
        return answer_id

    def add_transition(self, question_id: int, transition: Transition) -> None:
        idx = self.resolve_question_idx(question_id)
        buffer = self.replay_buffer[idx]

        prev_len = len(buffer.transitions)
        buffer.transitions.append(transition)
        new_len = len(buffer.transitions)
        self._total_size += new_len - prev_len

    def add_reset_transition(
        self,
        question_id: int,
        state: torch.Tensor,
        action_dim: int,
    ) -> None:
        state_cpu = state.detach().cpu()
        zero_action = torch.zeros(action_dim, dtype=self._dtype)
        zero_reward = torch.zeros(1, dtype=self._reward_dtype)
        zero_done = torch.zeros(1, dtype=torch.bool)

        path_tensor = state_cpu.unsqueeze(0)

        reset_transition = Transition(
            state=state_cpu,
            action=zero_action,
            reward=zero_reward,
            next_state=state_cpu,
            done=zero_done,
            path_states=path_tensor,
            step_index=0,
            env_reward=zero_reward,
            llm_reward=zero_reward,
        )
        self.add_transition(question_id, reset_transition)

    def iter_all(self) -> Iterable[Tuple[int, Transition]]:
        for internal_idx, question_buffer in enumerate(self.replay_buffer):
            question_id = self._question_ids[internal_idx]
            for transition in question_buffer.transitions:
                yield question_id, transition

    def sample(self, device: torch.device, batch_size: Optional[int] = None) -> Dict[str, torch.Tensor]:
        if batch_size is None:
            batch_size = self.batch_size

        all_transitions: List[Tuple[int, Transition]] = list(self.iter_all())
        if len(all_transitions) < batch_size:
            raise ValueError(
                "Cannot sample from replay buffer before it holds at least one batch"
            )

        perm = torch.randperm(len(all_transitions))[:batch_size]

        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        env_rewards = []
        llm_rewards = []
        log_probs: List[torch.Tensor] = []
        entropies: List[torch.Tensor] = []
        question_embeddings = []
        question_ids: List[int] = []
        path_states: List[List[torch.Tensor]] = []
        step_indices: List[int] = []

        for idx in perm.tolist():
            question_id, transition = all_transitions[idx]
            states.append(transition.state)
            actions.append(transition.action)
            rewards.append(transition.reward)
            next_states.append(transition.next_state)
            dones.append(transition.done)
            env_rewards.append(transition.env_reward)
            llm_rewards.append(transition.llm_reward)
            path_states.append(transition.path_states)
            step_indices.append(transition.step_index)
            question_embeddings.append(self.get_question_embedding(question_id))
            question_ids.append(question_id)

            if transition.log_prob is not None:
                log_probs.append(transition.log_prob)
            if transition.entropy is not None:
                entropies.append(transition.entropy)

        batch = {
            "states": torch.stack(states).to(device),
            "actions": torch.stack(actions).to(device),
            "rewards": torch.stack(rewards).to(device),
            "next_states": torch.stack(next_states).to(device),
            "dones": torch.stack(dones).to(device),
        }

        extras: Dict[str, torch.Tensor] = {
            "env_reward": torch.stack(env_rewards).to(device),
            "llm_reward": torch.stack(llm_rewards).to(device),
            "question_embedding": torch.stack(question_embeddings).to(device),
            "question_id": torch.tensor(question_ids, dtype=torch.long, device=device),
            "step_index": torch.tensor(step_indices, dtype=torch.long, device=device),
        }

        if log_probs:
            extras["log_prob"] = torch.stack(log_probs).to(device)
        if entropies:
            extras["entropy"] = torch.stack(entropies).to(device)

        batch["extras"] = {
            **extras,
            "path_states": path_states,
        }

        return batch
