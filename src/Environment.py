from LearningAgent import LearningAgent
from Action import Action
from Request import Request
from typing import Type, List, Generator
from abc import ABCMeta, abstractmethod
from pandas import read_csv
import pdb
import re


class Environment(metaclass=ABCMeta):
    """Defines a class for simulating the Environment for the RL agent"""

    def __init__(self):
        # Load environment
        self.initialise_environment()

    @abstractmethod
    def initialise_environment(self):
        raise NotImplementedError

    @abstractmethod
    def get_request_batch(self):
        raise NotImplementedError

    @abstractmethod
    def get_travel_time(self, source, destination):
        raise NotImplementedError

    @abstractmethod
    def get_next_location(self, source, destination):
        raise NotImplementedError

    @abstractmethod
    def get_initial_state(self, num_agents):
        raise NotImplementedError

    @abstractmethod
    def get_epoch_length(self):
        raise NotImplementedError

    def simulate_motion(self, agents: List[LearningAgent]) -> None:
        for agent in agents:
            time_remaining: float = self.get_epoch_length()

            while(time_remaining >= 0):
                time_remaining -= agent.position.time_to_next_location

                # If we reach an intersection, make a decision about where to go next
                if (time_remaining >= 0):
                    # If the intersection is an existing pick-up or drop-off location, update the Agent's path
                    if (agent.position.next_location == agent.path.get_next_location()):
                        agent.path.visit_next_location()

                    # Go to the next location in the path, if it exists
                    if (not agent.path.is_empty()):
                        next_location = self.get_next_location(agent.position.next_location, agent.path.get_next_location())
                        agent.position.time_to_next_location = self.get_travel_time(agent.position.next_location, next_location)
                        agent.position.next_location = next_location

                    # TODO: Perform rebalancing if there is no next location
                    else:
                        break
                # Else, continue down the road you're on
                else:
                    agent.position.time_to_next_location -= (time_remaining + agent.position.time_to_next_location)

    def get_reward(self, action: Action) -> float:
        """
        Return the reward to an agent for a given (feasible) action.

        (Feasibility is not checked!)
        Defined in Environment class because of Reinforcement Learning
        convention in literature.
        """
        return sum([request.value for request in action.requests])


class NYEnvironment(Environment):
    """Define an Environment using the cleaned NYC Yellow Cab dataset."""

    NUM_LOCATIONS: int = 4461
    NUM_MAX_AGENTS: int = 1000

    MAX_CAPACITY: int = 10
    EPOCH_LENGTH: float = 60.0

    DATA_DIR: str = '../data/'
    TRAVELTIME_FILE: str = DATA_DIR + 'ny/zone_traveltime.csv'
    SHORTESTPATH_FILE: str = DATA_DIR + 'ny/zone_path.csv'
    INITIALZONES_FILE: str = DATA_DIR + 'ny/taxi_randominit_1000.txt'
    IGNOREDZONES_FILE: str = DATA_DIR + 'ny/ignorezonelist.txt'
    DATA_FILE_PREFIX: str = DATA_DIR + 'ny/test_flow_5000_'

    def __init__(self):
        super().__init__()

    def initialise_environment(self):
        print('Loading Environment...')

        # TODO: Make current_time something that all environments have
        self.current_time: float = 0.0

        self.travel_time = read_csv(self.TRAVELTIME_FILE,
                                    header=None).values

        self.shortest_path = read_csv(self.SHORTESTPATH_FILE,
                                      header=None).values

        self.ignored_zones = read_csv(self.IGNOREDZONES_FILE,
                                      header=None).values.flatten()

        self.initial_zones = read_csv(self.INITIALZONES_FILE,
                                      header=None).values.flatten()

    def get_request_batch(self, day: int=2) -> Generator[List[Request], None, None]:
        # Open file to read
        with open(self.DATA_FILE_PREFIX + str(day) + '.txt', 'r') as data_file:
            num_batches: int = int(data_file.readline().strip())

            # Defines the 2 possible RE for lines in the data file
            new_epoch_re = re.compile(r'Flows:(\d+)-\d+')
            request_re = re.compile(r'(\d+),(\d+),(\d+)\.0')

            # Parsing first 'new_epoch' line
            new_epoch = re.match(new_epoch_re, data_file.readline().strip())
            assert new_epoch is not None  # Make sure we got the formatting right
            current_epoch = int(new_epoch.group(1))
            self.current_time = current_epoch * self.EPOCH_LENGTH

            # Parsing rest of the file
            request_list: List[Request] = []
            for line in data_file.readlines():
                line = line.strip()

                is_new_epoch = re.match(new_epoch_re, line)
                if (is_new_epoch is not None):
                    yield request_list
                    current_epoch = int(is_new_epoch.group(1))
                    self.current_time = current_epoch * self.EPOCH_LENGTH
                    request_list.clear()  # starting afresh for new batch
                else:
                    request_data = re.match(request_re, line)
                    assert request_data is not None  # Make sure there's nothing funky going on with the formatting

                    num_requests = int(request_data.group(3))
                    for _ in range(num_requests):
                        source = int(request_data.group(1))
                        destination = int(request_data.group(2))
                        if (source not in self.ignored_zones and destination not in self.ignored_zones):
                            travel_time = self.get_travel_time(source, destination)
                            request_list.append(Request(source, destination, self.current_time, travel_time))

            yield request_list

    def get_travel_time(self, source: int, destination: int) -> float:
        return self.travel_time[source, destination]

    def get_next_location(self, source: int, destination: int) -> int:
        return self.shortest_path[source, destination]

    def get_initial_state(self, num_agents: int) -> List[int]:
        """Give initial states for num_agents agents"""
        assert (num_agents <= self.NUM_MAX_AGENTS)
        return self.initial_zones[:num_agents]

    def get_epoch_length(self) -> float:
        return self.EPOCH_LENGTH

    def has_valid_path(self, agent: LearningAgent) -> bool:
        """Attempt to check if the request order meets deadline and capacity constraints"""
        def invalid_path_trace(issue: str) -> bool:
            print(issue)
            print('Agent {}:'.format(agent.id))
            print('Requests -> {}'.format(agent.path.requests))
            print('Request Order -> {}'.format(agent.path.request_order))
            print()
            return False

        # Make sure that it visits all the requests that it has accepted
        if (not agent.path.is_complete()):
            return invalid_path_trace('Incomplete path.')

        # Start at global_time and current_capacity
        current_time = self.current_time + agent.position.time_to_next_location
        current_location = agent.position.next_location
        current_capacity = agent.path.current_capacity

        # Iterate over path
        available_delay: float = 0
        for node_idx, node in enumerate(agent.path.request_order):
            next_location, deadline = agent.path.get_info(node)

            # Delay related checks
            travel_time = self.get_travel_time(current_location, next_location)
            if (current_time + travel_time > deadline):
                return invalid_path_trace('Does not meet deadline at node {}'.format(node_idx))

            current_time += travel_time
            current_location = next_location

            # Updating available delay
            if (node.expected_visit_time != current_time):
                invalid_path_trace("(Ignored) Visit time incorrect at node {}".format(node_idx))
                node.expected_visit_time = current_time

            if (node.is_dropoff):
                available_delay += deadline - node.expected_visit_time

            # Capacity related checks
            if (current_capacity > self.MAX_CAPACITY):
                return invalid_path_trace('Exceeds MAX_CAPACITY at node {}'.format(node_idx))

            if (node.is_dropoff):
                next_capacity = current_capacity - 1
            else:
                next_capacity = current_capacity + 1
            if (node.current_capacity != next_capacity):
                invalid_path_trace("(Ignored) Capacity incorrect at node {}".format(node_idx))
                node.current_capacity = next_capacity
            current_capacity = node.current_capacity

        agent.path.total_delay = available_delay
        return True
