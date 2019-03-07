from LearningAgent import LearningAgent
from Action import Action

from typing import List, Tuple

from abc import ABCMeta, abstractmethod


class ValueFunction(metaclass=ABCMeta):
    """docstring for ValueFunction"""

    def __init__(self):
        super(ValueFunction, self).__init__()

    @abstractmethod
    def get_value(self, agent: LearningAgent, feasible_actions: List[Action]) -> List[Tuple[Action, float]]:
        raise NotImplementedError

    @abstractmethod
    def update(self, current_rewards: List[float], current_state: List[LearningAgent], current_actions: List[Action]):
        raise NotImplementedError


class ImmediateReward(ValueFunction):
    """docstring for ImmediateReward"""

    def __init__(self):
        super(ImmediateReward, self).__init__()

    def get_value(self, agent: LearningAgent, feasible_actions: List[Action]) -> List[Tuple[Action, float]]:
        scored_actions: List[Tuple[Action, float]] = []
        for action in feasible_actions:
            score = sum([request.value for request in action.requests])
            scored_actions.append((action, score))

        return scored_actions

    def update(self, current_rewards: List[float], current_state: List[LearningAgent], current_actions: List[Action]):
        pass


class RewardPlusDelay(ValueFunction):
    """docstring for RewardPlusDelay"""

    def __init__(self, DELAY_COEFFICIENT: float=10e-4):
        super(RewardPlusDelay, self).__init__()
        self.DELAY_COEFFICIENT = DELAY_COEFFICIENT

    def get_value(self, agent: LearningAgent, feasible_actions: List[Action]) -> List[Tuple[Action, float]]:
        scored_actions: List[Tuple[Action, float]] = []
        for action in feasible_actions:
            assert action.new_path

            immediate_reward = sum([request.value for request in action.requests])
            remaining_delay_bonus = self.DELAY_COEFFICIENT * action.new_path.total_delay
            score = immediate_reward + remaining_delay_bonus

            scored_actions.append((action, score))

        return scored_actions

    def update(self, current_rewards: List[float], current_state: List[LearningAgent], current_actions: List[Action]):
        pass
