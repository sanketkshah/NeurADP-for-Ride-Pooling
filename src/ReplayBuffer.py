# ADAPTED FROM OPENAI BASELINES
# (https://github.com/openai/baselines)

from LearningAgent import LearningAgent
from Action import Action
from segment_tree import MinSegmentTree, SumSegmentTree

from typing import Tuple, List

from abc import ABC, abstractmethod
from collections import deque
from random import randint, random
import numpy as np


class ReplayBuffer(ABC):
    """docstring for ReplayBuffer"""

    def __init__(self, MAX_LEN):
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


# TODO: Factor out experiences into separate class
class SimpleReplayBuffer(ReplayBuffer):
    def __init__(self, MAX_LEN: int, **kwargs):
        """Create Replay buffer.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        super(SimpleReplayBuffer, self).__init__(MAX_LEN)

        self._storage: List[Tuple[LearningAgent, List[Action], float, bool]] = []
        self._maxsize: int = MAX_LEN
        self._next_idx: int = 0

    def __len__(self) -> int:
        return len(self._storage)

    def add(self, experience: Tuple[LearningAgent, List[Action], float, bool]):
        if self._next_idx >= len(self._storage):
            self._storage.append(experience)
        else:
            self._storage[self._next_idx] = experience
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes: List[int]) -> List[Tuple[LearningAgent, List[Action], float, bool]]:
        return [self._storage[i] for i in idxes]

    def sample(self, batch_size: int) -> List[Tuple[LearningAgent, List[Action], float, bool]]:
        """Sample a batch of experiences.

        Parameters
        ----------
        batch_size: int
            How many transitions to sample.

        Returns
        -------
        experiences: List[Experience]
            batch of experiences
        """
        idxes: List[int] = [randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)


class PrioritizedReplayBuffer(SimpleReplayBuffer):
    def __init__(self, MAX_LEN, alpha: float=0.6):
        """
        Create Prioritized Replay buffer.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        alpha: float
            how much prioritization is used
            (0 - no prioritization, 1 - full prioritization)

        See Also
        --------
        SimpleReplayBuffer.__init__
        """
        super(PrioritizedReplayBuffer, self).__init__(MAX_LEN)
        assert alpha >= 0
        self._alpha = alpha

        it_capacity = 1
        while it_capacity < MAX_LEN:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 100.0

    def add(self, *args, **kwargs):
        """See SimpleReplayBuffer.store_effect"""
        idx = self._next_idx
        super().add(*args, **kwargs)
        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha

    def _sample_proportional(self, batch_size):
        res = []
        p_total = self._it_sum.sum(0, len(self._storage) - 1)
        every_range_len = p_total / batch_size
        for i in range(batch_size):
            mass = random() * every_range_len + i * every_range_len
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def sample(self, batch_size: int, beta: float=0.4):
        """Sample a batch of experiences.

        compared to SimpleReplayBuffer.sample
        it also returns importance weights and idxes
        of sampled experiences.


        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        beta: float
            To what degree to use importance weights
            (0 - no corrections, 1 - full correction)

        Returns
        -------
        experiences: List[Experience]
            batch of experiences
        weights: np.array
            Array of shape (batch_size,) and dtype np.float32
            denoting importance weight of each sampled transition
        idxes: np.array
            Array of shape (batch_size,) and dtype np.int32
            idexes in buffer of sampled experiences
        """
        assert beta > 0

        idxes = self._sample_proportional(batch_size)

        weights = []
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * len(self._storage)) ** (-beta)

        for idx in idxes:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * len(self._storage)) ** (-beta)
            weights.append(weight / max_weight)
        weights = np.array(weights)
        encoded_sample = self._encode_sample(idxes)
        return (encoded_sample, weights, idxes)

    def update_priorities(self, idxes, priorities):
        """Update priorities of sampled transitions.

        sets priority of transition at index idxes[i] in buffer
        to priorities[i].

        Parameters
        ----------
        idxes: [int]
            List of idxes of sampled transitions
        priorities: [float]
            List of updated priorities corresponding to
            transitions at the sampled idxes denoted by
            variable `idxes`.
        """
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):
            assert priority > 0
            assert 0 <= idx < len(self._storage)
            self._it_sum[idx] = priority ** self._alpha
            self._it_min[idx] = priority ** self._alpha

            self._max_priority = max(self._max_priority, priority)
