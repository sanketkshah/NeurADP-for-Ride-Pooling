from Request import Request
from Action import Action
from LearningAgent import LearningAgent, AgentLocation
from Path import Path, PathNode, RequestInfo
from Environment import NYEnvironment

from typing import List, Tuple, Optional, FrozenSet, Dict, Any, Set

from copy import deepcopy
from collections import namedtuple
import heapq


# TODO: Factor out template from implementation like in Environment
class Oracle(object):
    """An Oracle returns a list of feasible actions for a given agent."""

    def __init__(self, envt):
        self.envt: NYEnvironment = envt

    def get_feasible_actions(self,
                             agents: List[LearningAgent],
                             requests: List[Request],
                             MAX_ACTIONS: int = -1,
                             MAX_TRIPS_SIZE_1: int = 30,
                             MAX_IS_FEASIBLE_CALLS: int = 150) -> List[List[Action]]:
        """Get a list of the best feasible actions for each agent."""

        # Associate requests with closest MAX_TRIPS_SIZE_1 vehicles
        def _distance_to_request(agent: LearningAgent, request: Request):
            return self.envt.get_travel_time(agent.position.next_location, request.pickup) + agent.position.time_to_next_location

        MAX_TRIPS_SIZE_1 = min(MAX_TRIPS_SIZE_1, len(agents))
        requests_for_each_agent: List[List[Request]] = [[] for _ in range(len(agents))]
        for request_idx, request in enumerate(requests):
            times_to_pickup = [(agent_idx, _distance_to_request(agent, request)) for agent_idx, agent in enumerate(agents)]
            times_to_pickup.sort(key=lambda x: x[1])
            for idx in range(MAX_TRIPS_SIZE_1):
                agent_idx = times_to_pickup[idx][0]
                requests_for_each_agent[agent_idx].append(request)

        # Get feasible trips for each vehicle
        feasible_actions_all_agents = []
        for requests_for_agent, agent in zip(requests_for_each_agent, agents):
            trips: List[Action] = []
            tested_actions: Set[Action] = set()
            num_is_feasible_calls = 0

            # Create null action
            null_action = Action([])
            null_action.new_path = agent.path
            trips.append(null_action)

            # Check feasibility for individual requests
            for request in requests_for_agent:
                action = Action([request])
                action.new_path = self.get_new_path(agent, agent.path, request)

                if action.new_path is not None:
                    trips.append(action)

                num_is_feasible_calls += 1
                tested_actions.add(action)

            # Get feasible trips of size > 1, with a fixed budget of MAX_IS_FEASIBLE_CALLS
            def trip_priority(trip: Action) -> float:
                assert trip.new_path
                return - trip.new_path.total_delay / len(trip.new_path.requests)

            trips_size_1 = list(range(1, len(trips)))  # not considering the null trip
            nodes_to_expand = [(trip_priority(trip), trip_idx + 1) for trip_idx, trip in enumerate(trips[1:])]
            heapq.heapify(nodes_to_expand)
            while len(nodes_to_expand) > 0 and num_is_feasible_calls < MAX_IS_FEASIBLE_CALLS:
                _, trip_idx = heapq.heappop(nodes_to_expand)
                action = trips[trip_idx]
                for trip_size_1_idx in trips_size_1:
                    prev_requests = action.requests
                    new_requests = trips[trip_size_1_idx].requests
                    new_action = Action(prev_requests.union(new_requests))

                    if (new_action not in tested_actions):
                        # Hacky way to get request the only request from the frozenset requests
                        new_request, = new_requests
                        assert action.new_path is not None  # Make sure no invalid actions slipped past
                        new_path = self.get_new_path(agent, action.new_path, new_request)

                        if (new_path is not None):
                            new_action.new_path = new_path
                            trips.append(new_action)
                            heapq.heappush(nodes_to_expand, (trip_priority(new_action), len(trips) - 1))

                        num_is_feasible_calls += 1
                        tested_actions.add(new_action)

                # Create only MAX_ACTIONS actions
                if (MAX_ACTIONS >= 0 and len(trips) >= MAX_ACTIONS):
                    break

            feasible_actions_all_agents.append(trips)

        return feasible_actions_all_agents

    def get_new_path(self, agent: LearningAgent, current_path: Path, new_request: Request, SEARCH_THRESHOLD: int=-1) -> Optional[Path]:
        # Create new Path variable to return
        new_path = deepcopy(current_path)

        # Add new_request to new_path
        new_request_info = RequestInfo(new_request, False)
        new_path.requests.append(new_request_info)

        # Find a way to fulfill these requests
        # Complete Search: num_requests <= threshold
        if (len(new_path.request_order) <= SEARCH_THRESHOLD):
            # Find best possible path for all requests using complete search
            new_path = self.get_new_path_complete_search(agent, new_path)

        # Heuristic Search: num_requests > threshold
        else:
            # Find best possible path formed by inserting new requests into the current path
            new_path = self.get_new_path_insertion(agent, new_path)

        # Check if we found any valid paths
        if (new_path.is_complete()):
            return new_path
        else:
            return None

    def get_new_path_complete_search(self, agent: LearningAgent, path: Path) -> Path:
        # TODO: Factor out remaining_delay
        path.total_delay = Path.NOT_APPLICABLE  # indicates that we haven't built a new path yet

        # Create list of nodes to visit
        possible_next_nodes: List[PathNode] = []
        for request_id, request in enumerate(path.requests):
            possible_next_nodes.append(PathNode(request.has_been_picked_up, request_id))
        current_index = 0
        current_remaining_delay = 0.0

        # Explore all combinations of requests
        stack: List[Tuple[List[PathNode], int, float]] = []
        current_request_order: List[PathNode] = []

        while True:
            # Check if you can go deeper into the search space
            stepBack = False
            if (current_index < len(possible_next_nodes)):
                # Expand the current tree
                next_node = possible_next_nodes[current_index]
                current_index += 1  # update current_index

                # If adding the next node is not feasible, continue
                next_location, deadline = path.get_info(next_node)
                if not current_request_order:
                    current_location = agent.position.next_location
                    current_time = self.envt.current_time + agent.position.time_to_next_location
                    current_capacity: int = path.current_capacity
                else:
                    current_location, _ = path.get_info(current_request_order[-1])
                    current_time = current_request_order[-1].expected_visit_time
                    current_capacity = current_request_order[-1].current_capacity

                travel_time = self.envt.get_travel_time(current_location, next_location)
                time_at_next_location = current_time + travel_time

                # If pick-up, check if it violates the current capacity
                if (not next_node.is_dropoff and current_capacity + 1 > self.envt.MAX_CAPACITY):
                    stepBack = True

                # Check if it meets deadline
                if (time_at_next_location > deadline):
                    stepBack = True

            # Else, check if this path has been completed
            else:
                if (path.is_complete(current_request_order)):
                    # Check if this is the best path
                    previous_lowest_delay = path.total_delay
                    if (previous_lowest_delay == Path.NOT_APPLICABLE or current_remaining_delay > previous_lowest_delay):
                        # Save, if it is
                        path.request_order = deepcopy(current_request_order)
                        path.total_delay = current_remaining_delay

                stepBack = True

            # If you can't go deeper, take a step back
            if (stepBack):
                if stack:
                    possible_next_nodes, current_index, current_remaining_delay = stack.pop()
                    current_request_order.pop()
                else:
                    break

            # Else, go one step deeper
            else:
                # Add to the current path, given it is feasible
                next_node.expected_visit_time = time_at_next_location
                if (next_node.is_dropoff):
                    next_node.current_capacity = current_capacity - 1
                else:
                    next_node.current_capacity = current_capacity + 1

                current_request_order.append(next_node)

                # Go one step deeper into the search space
                # Store state at current depth
                stack.append((deepcopy(possible_next_nodes), current_index, current_remaining_delay))

                # Remove next_node from possible nodes at lower depth
                possible_next_nodes.pop(current_index - 1)

                # If it is a pickup location, add dropoff to possible_next_nodes
                if not next_node.is_dropoff:
                    corresponding_dropoff = PathNode(True, next_node.relevant_request_id)
                    possible_next_nodes.append(corresponding_dropoff)

                # Update state for search at the next depth
                current_index = 0
                if (next_node.is_dropoff):
                    current_remaining_delay += (deadline - time_at_next_location)  # only dropoff delay is relevant

        return path

    def get_new_path_insertion(self, agent: LearningAgent, path: Path) -> Path:
        # DROPOFF
        relevant_request_id = len(path.requests) - 1  # Assume the new request is the last one
        new_node = PathNode(True, relevant_request_id)
        path, insert_idx = self._insert_path(agent, path, new_node)

        # PICKUP
        if (insert_idx != Path.NOT_APPLICABLE):
            new_node = PathNode(False, relevant_request_id)
            path, _ = self._insert_path(agent, path, new_node, insert_idx)

        return path

    def _insert_path(self, agent: LearningAgent, path: Path, new_node: PathNode, max_insert_idx: int=-1) -> Tuple[Path, int]:
        # Deal with deafult values
        if (max_insert_idx == -1):
            max_insert_idx = len(path.request_order)

        # Get info about new_node
        location, deadline = path.get_info(new_node)
        is_dropoff = new_node.is_dropoff

        # Find insertion point with minimum dropoff delay
        num_dropoffs_after = 0
        min_future_delay = float('inf')
        max_total_delay = Path.NOT_APPLICABLE
        min_delay_idx = Path.NOT_APPLICABLE

        # Check all insertion points by iterating backward through the request order
        for insert_idx in range(len(path.request_order), -1, -1):
            delay, delay_for_new_node, min_future_delay = self._can_insert_node(agent, path, new_node, insert_idx, min_future_delay)

            # If pickup node, only insert if it's before dropoff node
            if (insert_idx <= max_insert_idx):
                # If it can be inserted, check if it has less delay than previous feasible paths
                if (delay != Path.NOT_APPLICABLE):
                    total_delay = path.total_delay - (delay * num_dropoffs_after) + delay_for_new_node

                    # Save if it has less delay
                    if (total_delay >= max_total_delay):
                        max_total_delay = total_delay
                        min_individual_delay = delay
                        min_delay_idx = insert_idx

                # Special check for capacity constratints
                if (not is_dropoff):
                    _, _, _, current_capacity = self._get_node_info(agent, path, insert_idx - 1)
                    if (current_capacity >= self.envt.MAX_CAPACITY):
                        break

            # Update num_dropoffs_after
            if insert_idx > 0 and path.request_order[insert_idx - 1].is_dropoff:
                num_dropoffs_after += 1

        # If an insertion location is found, insert into path
        if (min_delay_idx != Path.NOT_APPLICABLE):
            # Fill in new_node's info
            self._insert_pathnode(agent, path, new_node, min_delay_idx, min_individual_delay, max_insert_idx)
            # Update total_delay
            path.total_delay = max_total_delay

        # If no insertion point was found, abort
        return path, min_delay_idx

    def _get_node_info(self, agent: LearningAgent, path: Path, idx: int):
        if (idx != Path.NOT_APPLICABLE):
            node = path.request_order[idx]
            location, deadline = path.get_info(node)
            visit_time = node.expected_visit_time
            current_capacity = node.current_capacity
        else:
            location = agent.position.next_location
            deadline = float('inf')
            visit_time = self.envt.current_time + agent.position.time_to_next_location
            current_capacity = path.current_capacity

        return location, deadline, visit_time, current_capacity

    def _can_insert_node(self,
                         agent: LearningAgent,
                         path: Path,
                         new_node: PathNode,
                         insert_idx: int,
                         min_future_delay: float) -> Tuple[float, float, float]:

        # Get info about new_node
        location, deadline = path.get_info(new_node)
        delay_for_new_node = 0.0
        node_delay = float(Path.NOT_APPLICABLE)

        # Get info about prev_node
        prev_location, prev_deadline, visit_time, current_capacity = self._get_node_info(agent, path, insert_idx - 1)

        # Check if it violates the capacity constraint
        if (new_node.is_dropoff or current_capacity < self.envt.MAX_CAPACITY):
            # Check if it meets later nodes' deadlines
            travel_time_prev = self.envt.get_travel_time(prev_location, location)
            delay = 0.0
            if (insert_idx < len(path.request_order)):
                next_location, _ = path.get_info(path.request_order[insert_idx])
                travel_time_next = self.envt.get_travel_time(location, next_location)
                travel_time_default = self.envt.get_travel_time(prev_location, next_location)
                delay = travel_time_prev + travel_time_next - travel_time_default

            if (delay <= min_future_delay):
                # Check if inserted node meets its own deadline
                if (deadline >= visit_time + travel_time_prev):
                    node_delay = delay
                    # Find out what delay for the new_node is
                    if (new_node.is_dropoff):
                        delay_for_new_node = deadline - (visit_time + travel_time_prev)

        # Update min_future_delay
        prev_node_max_delay = prev_deadline - visit_time
        min_future_delay = min(min_future_delay, prev_node_max_delay)

        return node_delay, delay_for_new_node, min_future_delay

    def _insert_pathnode(self,
                         agent: LearningAgent,
                         path: Path,
                         new_node: PathNode,
                         min_delay_idx: int,
                         delay: float,
                         corresponding_dropoff_idx: int=Path.NOT_APPLICABLE):
        # Node information
        location, _ = path.get_info(new_node)
        is_dropoff = new_node.is_dropoff

        # Fill in details for new_node
        if (min_delay_idx == 0):
            new_node.current_capacity = path.current_capacity
            prev_time = self.envt.current_time + agent.position.time_to_next_location
            prev_loc = agent.position.next_location
        else:
            prev_idx = min_delay_idx - 1
            prev_node = path.request_order[prev_idx]
            new_node.current_capacity = prev_node.current_capacity

            prev_time = prev_node.expected_visit_time
            prev_loc, _ = path.get_info(prev_node)

        travel_time = self.envt.get_travel_time(prev_loc, location)
        new_node.expected_visit_time = prev_time + travel_time
        if not is_dropoff:
            new_node.current_capacity += 1

        # Insert new_node
        path.request_order.insert(min_delay_idx, new_node)

        # Update details of future nodes
        for idx in range(min_delay_idx + 1, len(path.request_order)):
            # Update visit time
            node = path.request_order[idx]
            node.expected_visit_time += delay

            # Increase capacity by one until dropoff
            if not is_dropoff and idx <= corresponding_dropoff_idx:
                node.current_capacity += 1

    def get_path_insertion_2(self, agent: LearningAgent, path: Path) -> Path:
        request = path.requests[-1]
        max_total_delay = Path.NOT_APPLICABLE

        # Collecting all the information necessary to do modifications in one place
        InsertInfo: Dict[str, Any] = {'min_delay_idx': Path.NOT_APPLICABLE,
                                      'min_delay': Path.NOT_APPLICABLE,
                                      'num_dropoffs_after': 0,
                                      'min_future_delay': float('inf'),
                                      'new_node': None}

        pickup = deepcopy(InsertInfo)
        pickup['new_node'] = PathNode(False, len(path.requests) - 1)

        dropoff = deepcopy(InsertInfo)
        dropoff['new_node'] = PathNode(True, len(path.requests) - 1)

        # Iterate over possible locations to insert dropoff
        for dropoff_idx in range(len(path.request_order), -1, -1):
            dropoff_delay, new_node_delay, dropoff_future_delay = self._can_insert_node(agent, path, dropoff['new_node'], dropoff_idx, dropoff['min_future_delay'])

            # If feasible to insert, consider pickup locations
            if (dropoff_delay != Path.NOT_APPLICABLE):
                # Temporarily insert dropoff
                path.request_order.insert(dropoff_idx, dropoff['new_node'])

                # Iterate over pickup locations for given dropoff
                pickup['min_future_delay'] = min(dropoff['min_future_delay'] - dropoff_delay, new_node_delay)
                pickup['num_dropoffs_after'] = dropoff['num_dropoffs_after'] + 1

                for pickup_idx in range(dropoff_idx, -1, -1):
                    pickup_delay, _, pickup['min_future_delay'] = self._can_insert_node(agent, path, pickup['new_node'], pickup_idx, pickup['min_future_delay'])

                    # If feasible to pickup, this is a valid combination
                    if (pickup_delay != Path.NOT_APPLICABLE):
                        # Check if this combination has less delay than previous feasible ones
                        total_delay = path.total_delay
                        total_delay -= (dropoff_delay * dropoff['num_dropoffs_after'])  # subtracting delay due to pickup
                        total_delay -= (pickup_delay * pickup['num_dropoffs_after'])  # subtracting delay due to dropoff
                        total_delay += new_node_delay  # adding available delay for the new request

                        # Save if it has less delay
                        if (total_delay >= max_total_delay):
                            max_total_delay = total_delay
                            dropoff['min_individual_delay'] = dropoff_delay
                            dropoff['min_delay_idx'] = dropoff_idx
                            pickup['min_individual_delay'] = pickup_delay
                            pickup['min_delay_idx'] = pickup_idx

                    # Special check for capacity constratints
                    _, _, _, current_capacity = self._get_node_info(agent, path, pickup_idx - 1)
                    if (current_capacity >= self.envt.MAX_CAPACITY):
                        break

                    # Update num_dropoffs_after
                    if pickup_idx > 0:
                        if path.request_order[pickup_idx - 1].is_dropoff:
                            pickup['num_dropoffs_after'] += 1

                # Delete temporary dropoff
                path.request_order.pop(dropoff_idx)

            # Update Dropoff delay
            dropoff['min_future_delay'] = dropoff_future_delay

            # Update num_dropoffs_after
            if dropoff_idx > 0:
                if path.request_order[dropoff_idx - 1].is_dropoff:
                    dropoff['num_dropoffs_after'] += 1

        # If an insertion locations are found, insert into path
        if (max_total_delay != Path.NOT_APPLICABLE):
            # Dropoff
            self._insert_pathnode(agent, path, dropoff['new_node'], dropoff['min_delay_idx'], dropoff['min_individual_delay'])
            # Pickup
            self._insert_pathnode(agent, path, pickup['new_node'], pickup['min_delay_idx'], pickup['min_individual_delay'], dropoff['min_delay_idx'])
            # Update total_delay
            path.total_delay = max_total_delay

        return path

    # def get_request_combinations(self, current_requests: List[Request]) -> Dict[Set[Request]]:
    #     return
