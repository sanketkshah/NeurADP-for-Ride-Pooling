from Request import Request
from Path import Path
from typing import Set, FrozenSet, List, Iterable, Optional


class Action(object):
    """
    An Action is the output of an Agent for a decision epoch.

    In our formulation corresponds to an Agent accepting a given set
    of Requests.
    """

    def __init__(self, requests: Iterable[Request]) -> None:
        self.requests = frozenset(requests)
        self.new_path: Optional[Path] = None

    def __eq__(self, other):
        return (self.requests == other.requests)

    def __hash__(self):
        return hash(self.requests)
