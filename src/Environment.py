from LearningAgent import LearningAgent
from Action import Action
from abc import ABCMeta, abstractmethod


class Environment(metaclass=ABCMeta):
    """Defines a class for simulating the Environment for the RL agent"""

    def __init__(self):
        super(Environment, self).__init__()

        # Load environment
        self.initialise_environment()

    @abstractmethod
    def initialise_environment(self):
        raise NotImplementedError

    @abstractmethod
    def get_reqeust_batch(self):
        raise NotImplementedError

    @abstractmethod
    def get_travel_time(self, source, destination):
        raise NotImplementedError

    @abstractmethod
    def get_nextlocation_shortestpath(self, source, destination):
        raise NotImplementedError

    def simulate_motion(self, agent: LearningAgent):
        raise NotImplementedError

    def get_reward(self, action: Action):
        """
        Return the reward to an agent for a given (feasible) action.

        (Feasibility is not checked!)
        Defined in Environment class because of Reinforcement Learning
        convention in literature.
        """
        return sum([request.value for request in action])


class NYEnvironment(Environment):
    """Define an Environment using the cleaned NYC Yellow Cab dataset."""

    NUM_LOCATIONS = 4461
    TRAVELTIME_FILE = 'data/ny/zone_traveltime.csv'
    SHORTESTPATH_FILE = 'data/ny/zone_path.csv'

    def __init__(self):
        super(NYEnvironment, self).__init__()

    def initialise_environment(self):
        # Read travel time matrix from file

        # Read shortest path matrix from file
        pass
