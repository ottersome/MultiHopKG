
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple

import torch


class QuestionReplayBuffer:
    """Torch-based replay buffer tailored for SAC-style training."""

    @dataclass
    class QuestionBuffer:
        states: List[List[torch.Tensor]]
        steps_no: torch.Tensor
        actions: torch.Tensor
        rewards: torch.Tensor
        next_states: torch.Tensor
        dones: torch.Tensor

    def __init__(
        self,
        num_questions: int,
        state_shape: int,
        action_shape: int,
        experiences_per_question: int, 
        batch_size: int,
        *,
        warmup_size: int = 0,
        min_update_steps: int = 0,
        dtype: torch.dtype = torch.float32,
        reward_dtype: torch.dtype = torch.float32,
    ) -> None:

        self.batch_size = batch_size
        self.warmup_size = warmup_size
        self.min_update_steps = min_update_steps
        self.experiences_per_question = experiences_per_question

        self._state_shape = state_shape
        self._action_shape = action_shape

        self.replay_buffer: List[QuestionReplayBuffer.QuestionBuffer] = []
        for i in range(num_questions):
            storage_shape = (experiences_per_question, state_shape)
            action_storage_shape = (experiences_per_question, action_shape)

            self.replay_buffer.append(
                QuestionReplayBuffer.QuestionBuffer(
                    states = [[]] * storage_shape[0],
                    steps_no = torch.zeros(storage_shape, dtype=dtype), # Should just keep track of how many steps before this till reset. Debugging for now. 
                    next_states = torch.zeros(storage_shape, dtype=dtype),
                    actions = torch.zeros(action_storage_shape, dtype=dtype),
                    rewards = torch.zeros((experiences_per_question,), dtype=reward_dtype),
                    dones = torch.zeros((experiences_per_question,), dtype=torch.bool),
            ))


        self._extras: Dict[str, List[Any]] = {}

        self._ptr = 0
        self._size = 0

    def __len__(self) -> int:
        return self._size

    @property
    def is_full(self) -> bool:
        return self._size == self.capacity

    @property
    def state_shape(self) -> Tuple[int, ...]:
        return self._state_shape

    @property
    def action_shape(self) -> Tuple[int, ...]:
        return self._action_shape

    def is_ready(self, num_updates_done: int) -> bool:
        """Return True when it's OK to start (or continue) optimization."""
        return (
            self._size >= self.warmup_size
            and num_updates_done >= self.min_update_steps
        )

    def _allocate_extra_if_needed(self, key: str) -> None:
        if key not in self._extras:
            self._extras[key] = [None] * self.capacity

    def add_batch(
        self,
        question_idx: int,
        *,
        states: List[torch.Tensor], # For now we assume its a list of positions and actions on the graph. To be later consumed and processed by the decoder in the agent.
        steps_no: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
        #extras: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Store a batch of transitions (expects tensors shaped [B, ...])."""
        # TODO: its a bit too hardcoded  to grab from the first element.
        batch_size = states[0].shape[0]
        if batch_size == 0:
            return

        indices = (torch.arange(batch_size, dtype=torch.long) + self._ptr) % self.experiences_per_question

        self.replay_buffer[question_idx].states[indices] = [ state.detach().cpu() for state in states]
        self.replay_buffer[question_idx].steps_no[indices] = steps_no.detach().cpu()
        self.replay_buffer[question_idx].actions[indices] = actions.detach().cpu()
        rewards_cpu = rewards[question_idx].detach().cpu().view(batch_size, -1)
        if rewards_cpu.shape[1] != 1:
            raise ValueError("rewards must be of shape [batch] or [batch, 1]")
        self.replay_buffer[question_idx].rewards[indices] = rewards_cpu.squeeze(-1)
        self.replay_buffer[question_idx].next_states[indices] = next_states[question_idx].detach().cpu()
        dones_cpu = dones[question_idx].detach().cpu().view(batch_size, -1)
        if dones_cpu.shape[1] != 1:
            raise ValueError("dones must be of shape [batch] or [batch, 1]")
        self.replay_buffer[question_idx].dones[indices] = dones_cpu.squeeze(-1).to(torch.bool)

        self._ptr = (self._ptr + batch_size) % self.experiences_per_question
        self._size = min(self._size + batch_size, self.experiences_per_question)

    def sample(self, device: torch.device) -> Dict[str, torch.Tensor]:
        """Sample a random batch and move tensors onto the target device."""
        if self._size < self.batch_size:
            raise ValueError(
                "Cannot sample from replay buffer before it holds at least one batch"
            )

        idx = torch.randint(0, self._size, (self.batch_size,))

        batch = {
            "states": self.states[idx].to(device),
            "actions": self.actions[idx].to(device),
            "rewards": self.rewards[idx].unsqueeze(-1).to(device),
            "next_states": self.next_states[idx].to(device),
            "dones": self.dones[idx].unsqueeze(-1).to(device),
        }

        if self._extras:
            extras_batch: Dict[str, Any] = {}
            for key, values in self._extras.items():
                gathered = [values[i] for i in idx.tolist()]
                first_item = gathered[0]
                if isinstance(first_item, torch.Tensor):
                    extras_batch[key] = torch.stack(gathered).to(device)
                else:
                    extras_batch[key] = gathered
            batch["extras"] = extras_batch

        return batch
