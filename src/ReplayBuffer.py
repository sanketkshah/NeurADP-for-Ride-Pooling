from LearningAgent import LearningAgent
from Action import Action

from typing import Tuple, List, Deque

from abc import ABC, abstractmethod
from collections import deque
from random import sample


class ReplayBuffer(ABC):
    """docstring for ReplayBuffer"""

    def __init__(self, MAX_LEN):
        super().__init__()
        self.MAX_LEN = MAX_LEN

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def add(self, experience: Tuple[LearningAgent, List[Action], float, bool]):
        raise NotImplementedError

    @abstractmethod
    def sample(self, num_experiences) -> List[Tuple[LearningAgent, List[Action], float, bool]]:
        raise NotImplementedError


class ExperienceReplay(ReplayBuffer):

    def __init__(self, MAX_LEN):
        super().__init__(MAX_LEN)

        self.replay_buffer: Deque[Tuple[LearningAgent, List[Action], float, bool]] = deque(maxlen=MAX_LEN)

    def __len__(self):
        return len(self.replay_buffer)

    def add(self, experience: Tuple[LearningAgent, List[Action], float, bool]):
        # Ignore the weight associated with the experience
        self.replay_buffer.append(experience)

    def sample(self, num_experiences) -> List[Tuple[LearningAgent, List[Action], float, bool]]:
        return sample(self.replay_buffer, num_experiences)
