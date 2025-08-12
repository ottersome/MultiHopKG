# Import Abstract Classes
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np
import torch


# TOREM: We are adding more flexibility to this. 
@dataclass
class Observation:
    state: torch.Tensor # Part of computation Graph
    kge_cur_pos: torch.Tensor # KG embedding vector of the current position
    kge_prev_pos: torch.Tensor # KG embedding vector of the previous position
    kge_action: torch.Tensor # KG embedding vector of the action

class Environment(ABC):

    @abstractmethod
    def reset(self, initial_state_info: Any) -> Any:
        """
        Args:
            - initial_state_info: any info that you wan to pass to initiazation.
        Returns:
            - position (torch.Tensor): Position in the graph
            - state (torch.Tensor): State containing informatioon for decision making.
        Both are meant to denote high abstraction so think about how to fit you are idea to them.
        """
        pass

    @abstractmethod
    def step(self, action: Any) -> Any:
        """
        Args:
            - action (torch.Tensor): The action to take
        Returns:
            - Observation (Any): The observation that comes from taking such action
        """
        pass


