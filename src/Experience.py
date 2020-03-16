from LearningAgent import LearningAgent
from Action import Action
from Environment import Environment

from typing import List, Optional, Dict, Any


class Experience(object):
    """docstring for Experience"""
    envt: Optional[Environment] = None

    def __init__(self, agents: List[LearningAgent], feasible_actions_all_agents: List[List[Action]], time: float, num_requests: int):
        super(Experience, self).__init__()
        self.agents = agents
        self.feasible_actions_all_agents = feasible_actions_all_agents
        self.time = time
        self.num_requests = num_requests

        assert self.envt is not None
        assert len(agents) == self.envt.NUM_AGENTS
        assert len(feasible_actions_all_agents) == self.envt.NUM_AGENTS

        self.representation: Dict[str, Any] = {}
