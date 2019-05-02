from LearningAgent import LearningAgent
from Action import Action
from Request import Request
from Path import PathNode, RequestInfo

from typing import Type, List, Generator, Tuple, Deque

from abc import ABCMeta, abstractmethod
from random import choice
from pandas import read_csv
from collections import deque
from docplex.mp.model import Model  # type: ignore
import re
from random import randint, random


class Environment(metaclass=ABCMeta):
    """Defines a class for simulating the Environment for the RL agent"""

    REQUEST_HISTORY_SIZE: int = 500

    def __init__(self, NUM_LOCATIONS: int, MAX_CAPACITY: int, EPOCH_LENGTH: float, NUM_AGENTS: int, START_EPOCH: float, STOP_EPOCH: float):
        # Load environment
        self.NUM_LOCATIONS = NUM_LOCATIONS
        self.MAX_CAPACITY = MAX_CAPACITY
        self.EPOCH_LENGTH = EPOCH_LENGTH
        self.NUM_AGENTS = NUM_AGENTS
        self.START_EPOCH = START_EPOCH
        self.STOP_EPOCH = STOP_EPOCH

        self.num_days_trained = 0
        self.recent_request_history: Deque[Request] = deque(maxlen=self.REQUEST_HISTORY_SIZE)
        self.current_time: float = 0.0

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
    def get_initial_states(self, num_agents, is_training):
        raise NotImplementedError

    def simulate_motion(self, agents: List[LearningAgent], current_requests: List[Request]=[], rebalance: bool=True) -> None:
        # Move all agents
        agents_to_rebalance: List[Tuple[LearningAgent, float]] = []
        for agent in agents:
            time_remaining: float = self.EPOCH_LENGTH
            time_remaining = self._move_agent(agent, time_remaining)
            # If it has visited all the locations it needs to and has time left, rebalance
            if (time_remaining > 0):
                agents_to_rebalance.append((agent, time_remaining))

        # Update recent_requests list
        self.update_recent_requests(current_requests)

        # Perform Rebalancing
        if (rebalance and agents_to_rebalance):
            rebalancing_targets = self._get_rebalance_targets([agent for agent, _ in agents_to_rebalance])

            # Move cars according to the rebalancing_targets
            for idx, target in enumerate(rebalancing_targets):
                agent, time_remaining = agents_to_rebalance[idx]

                # Insert dummy target
                agent.path.requests.append(RequestInfo(target, False))
                agent.path.request_order.append(PathNode(False, 0))  # adds pickup location to 'to-visit' list

                # Move according to dummy target
                self._move_agent(agent, time_remaining)

                # Delete dummy target
                agent.path.request_order.clear()
                agent.path.requests.clear()

    def _move_agent(self, agent: LearningAgent, time_remaining: float) -> float:
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

                # If no additional locations need to be visited, stop
                else:
                    break
            # Else, continue down the road you're on
            else:
                agent.position.time_to_next_location -= (time_remaining + agent.position.time_to_next_location)

        return time_remaining

    def _get_rebalance_targets(self, agents: List[LearningAgent]) -> List[Request]:
        # Get a list of possible targets by sampling from recent_requests
        possible_targets: List[Request] = []
        for _ in range(len(agents)):
            target = choice(self.recent_request_history)
            possible_targets.append(target)

        # Solve an LP to assign each agent to closest possible target
        model = Model()

        # Define variables, a matrix defining the assignment of agents to targets
        assignments = model.continuous_var_matrix(range(len(agents)), range(len(possible_targets)), name='assignments')

        # Make sure one target can only be assigned to one agentfor m in matches
        for agent_id in range(len(agents)):
            model.add_constraint(model.sum(assignments[agent_id, target_id] for target_id in range(len(possible_targets))) == 1)

        # Make sure one agent can only be assigned to one target
        for target_id in range(len(possible_targets)):
            model.add_constraint(model.sum(assignments[agent_id, target_id] for agent_id in range(len(agents))) == 1)

        # Define the objective: Minimise distance travelled
        model.minimize(model.sum(assignments[agent_id, target_id] * self.get_travel_time(agents[agent_id].position.next_location, possible_targets[target_id].pickup) for target_id in range(len(possible_targets)) for agent_id in range(len(agents))))

        # Solve
        solution = model.solve()
        assert solution  # making sure that the model doesn't fail

        # Get the assigned targets
        assigned_targets: List[Request] = []
        for agent_id in range(len(agents)):
            for target_id in range(len(possible_targets)):
                if (solution.get_value(assignments[agent_id, target_id]) == 1):
                    assigned_targets.append(possible_targets[target_id])
                    break

        return assigned_targets

    def get_reward(self, action: Action) -> float:
        """
        Return the reward to an agent for a given (feasible) action.

        (Feasibility is not checked!)
        Defined in Environment class because of Reinforcement Learning
        convention in literature.
        """
        return sum([request.value for request in action.requests])

    def update_recent_requests(self, recent_requests: List[Request]):
        self.recent_request_history.extend(recent_requests)


class NYEnvironment(Environment):
    """Define an Environment using the cleaned NYC Yellow Cab dataset."""

    NUM_MAX_AGENTS: int = 1000
    NUM_LOCATIONS: int = 4461
    EPOCH_LENGTH: float = 60.0

    DATA_DIR: str = '../data/'
    TRAVELTIME_FILE: str = DATA_DIR + 'ny/zone_traveltime.csv'
    SHORTESTPATH_FILE: str = DATA_DIR + 'ny/zone_path.csv'
    INITIALZONES_FILE: str = DATA_DIR + 'ny/taxi_randominit_1000.txt'
    IGNOREDZONES_FILE: str = DATA_DIR + 'ny/ignorezonelist.txt'
    DATA_FILE_PREFIX: str = DATA_DIR + 'ny/test_flow_5000_'

    def __init__(self, NUM_AGENTS: int, START_EPOCH: float, STOP_EPOCH: float, MAX_CAPACITY: int=10):
        assert NUM_AGENTS <= self.NUM_MAX_AGENTS
        super().__init__(NUM_LOCATIONS=self.NUM_LOCATIONS, MAX_CAPACITY=MAX_CAPACITY, EPOCH_LENGTH=self.EPOCH_LENGTH, NUM_AGENTS=NUM_AGENTS, START_EPOCH=START_EPOCH, STOP_EPOCH=STOP_EPOCH)

        self.initialise_environment()

    def initialise_environment(self):
        print('Loading Environment...')

        self.travel_time = read_csv(self.TRAVELTIME_FILE,
                                    header=None).values

        self.shortest_path = read_csv(self.SHORTESTPATH_FILE,
                                      header=None).values

        self.ignored_zones = read_csv(self.IGNOREDZONES_FILE,
                                      header=None).values.flatten()

        self.initial_zones = read_csv(self.INITIALZONES_FILE,
                                      header=None).values.flatten()

    def get_request_batch(self,
                          start_hour: int=0,
                          end_hour: int=24,
                          day: int=2,
                          downsample: float=1) -> Generator[List[Request], None, None]:

        assert 0 < downsample <= 1
        request_id = 0

        # Open file to read
        with open(self.DATA_FILE_PREFIX + str(day) + '.txt', 'r') as data_file:
            num_batches: int = int(data_file.readline().strip())

            # Defines the 2 possible RE for lines in the data file
            new_epoch_re = re.compile(r'Flows:(\d+)-\d+')
            request_re = re.compile(r'(\d+),(\d+),(\d+)\.0')

            # Parsing rest of the file
            request_list: List[Request] = []
            is_first_epoch = True
            for line in data_file.readlines():
                line = line.strip()

                is_new_epoch = re.match(new_epoch_re, line)
                if (is_new_epoch is not None):
                    if not is_first_epoch:
                        current_hour = int(self.current_time / 3600)
                        if (current_hour >= start_hour and current_hour < end_hour):
                            yield request_list

                        current_epoch = int(is_new_epoch.group(1))
                        self.current_time = current_epoch * self.EPOCH_LENGTH
                        request_list.clear()  # starting afresh for new batch
                    else:
                        is_first_epoch = False
                else:
                    request_data = re.match(request_re, line)
                    assert request_data is not None  # Make sure there's nothing funky going on with the formatting

                    num_requests = int(request_data.group(3))
                    for _ in range(num_requests):
                        # Take request according to downsampled rate
                        rand_num = random()
                        if (rand_num <= downsample):
                            source = int(request_data.group(1))
                            destination = int(request_data.group(2))
                            if (source not in self.ignored_zones and destination not in self.ignored_zones and source != destination):
                                    travel_time = self.get_travel_time(source, destination)
                                    request_list.append(Request(request_id, source, destination, self.current_time, travel_time))
                                    request_id += 1

            if (current_hour >= start_hour and current_hour < end_hour):
                yield request_list

    def get_travel_time(self, source: int, destination: int) -> float:
        return self.travel_time[source, destination]

    def get_next_location(self, source: int, destination: int) -> int:
        return self.shortest_path[source, destination]

    def get_initial_states(self, num_agents: int, is_training: bool) -> List[int]:
        """Give initial states for num_agents agents"""
        assert (num_agents <= self.NUM_MAX_AGENTS)

        # If it's training, get random states
        if is_training:
            initial_states = []

            for _ in range(num_agents):
                initial_state = randint(0, self.NUM_LOCATIONS - 1)
                # Make sure it's not an ignored zone
                while (initial_state in self.ignored_zones):
                    initial_state = randint(0, self.NUM_LOCATIONS - 1)

                initial_states.append(initial_state)
        # Else, pick deterministic initial states
        else:
            initial_states = self.initial_zones[:num_agents]

        return initial_states

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

        # Check total_delay
        if (agent.path.total_delay != available_delay):
            invalid_path_trace("(Ignored) Total delay incorrect.")
        agent.path.total_delay = available_delay

        return True
