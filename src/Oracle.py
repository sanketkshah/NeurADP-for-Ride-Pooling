from Request import Request
from Action import Action
from typing import List


class Oracle(object):
    """An Oracle returns a list of feasible actions for a given agent."""

    def __init__(self):
        super(Oracle, self).__init__()

    def get_feasible_actions(self, agent, requests) -> List[Action]:
        pass
