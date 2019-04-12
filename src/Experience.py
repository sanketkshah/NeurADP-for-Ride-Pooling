from LearningAgent import LearningAgent
from Action import Action

from typing import NamedTuple, List


class Experience(NamedTuple):
    agents: List[LearningAgent]
    feasible_actions_all_agents: List[List[Action]]
    time: float
    is_terminal: bool
