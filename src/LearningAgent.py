from typing import Dict, List
from Request import Request
from Path import Path


class LearningAgent(object):
    """
    A LearningAgent corresponds to a single decision making unit.

    In our formulation a learning agent corresponds to a single
    vehicle. It learns a value function based on the rewards it gets
    from the environment. It generates prefences for different actions
    using this value funciton and submits it to the CentralAgent for
    arbitration.
    """

    def __init__(self, agent_id: int, initial_location: int):
        # Initialising the state of the agent
        self.id = agent_id
        self.position = AgentLocation(initial_location)
        self.path: Path = Path()


class AgentLocation(object):
    """Define the current position of an Agent."""

    def __init__(self, next_location: int, time_to_next_location: float=0):
        self.next_location = next_location
        self.time_to_next_location = time_to_next_location
