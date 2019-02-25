from Request import Request
from Action import Action
from LearningAgent import LearningAgent, AgentLocation
from Path import Path, PathNode, RequestInfo
from Environment import NYEnvironment

from typing import List, Tuple, Optional, FrozenSet

from copy import deepcopy


# TODO: Factor out template from implementation like in Environment
class Oracle(object):
    """An Oracle returns a list of feasible actions for a given agent."""

    def __init__(self, envt):
        self.envt: NYEnvironment = envt

    def get_feasible_actions(self,
                             agent: LearningAgent,
                             requests: List[Request],
                             MAX_ACTIONS: int = -1,
                             MAX_TRIPS_SIZE_1: int = 30,
                             MAX_IS_FEASIBLE_CALLS: int = 1000) -> List[Action]:
        """Get a list of the best feasible actions for each agent."""

        trips: List[Action] = []
        tested_actions: List = []
        num_is_feasible_calls = 0

        # Create null action
        null_action = Action([])
        null_action.new_path = agent.path
        trips.append(null_action)

        # Get feasible trips of size 1
        # Order requests by pickup distance
        distances_from_pickup = []
        agent_location = agent.position.next_location
        for request_idx, request in enumerate(requests):
            time_to_pickup = self.envt.get_travel_time(
                agent_location, request.pickup)
            distances_from_pickup.append((request_idx, time_to_pickup))
        distances_from_pickup.sort(key=lambda x: x[1])

        # Check feasibility for closest MAX_TRIPS_SIZE_1 requests
        num_trips_size_1 = min(MAX_TRIPS_SIZE_1, len(requests))
        for request_idx, _ in distances_from_pickup[:num_trips_size_1]:
            action = Action([requests[request_idx]])
            action.new_path = self.get_new_path(agent, agent.path, requests[request_idx])

            if action.new_path is not None:
                trips.append(action)

            num_is_feasible_calls += 1
            tested_actions.append(action)

        # Get feasible trips of size > 1, with a fixed budget
        # TODO: Consider ordering new trips by symmetric difference to existing ones
        trips_size_1 = list(range(1, len(trips)))  # not considering the null trip
        nodes_to_expand = list(range(1, len(trips)))

        while len(nodes_to_expand) > 0 and num_is_feasible_calls < MAX_IS_FEASIBLE_CALLS:
            trip_idx = nodes_to_expand.pop()
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
                        nodes_to_expand.append(len(trips) - 1)

                    num_is_feasible_calls += 1
                    tested_actions.append(new_action)

        # Select best MAX_ACTIONS actions
        # TODO: Select 'best'. Currently selecting first MAX_ACTIONS actions.
        if (MAX_ACTIONS >= 0 and len(trips) > MAX_ACTIONS):
            trips = trips[:MAX_ACTIONS]

        return trips

    def get_new_path(self, agent: LearningAgent, current_path: Path, new_request: Request) -> Optional[Path]:
        # Create new Path variable to return
        new_path = deepcopy(current_path)

        # Add new_request to new_path
        new_request_info = RequestInfo(new_request, False)
        new_path.requests.append(new_request_info)

        # Find a way to fulfill these requests
        THRESHOLD = 6  # Length of existing path at which we stop trying a complete search for feasibility

        # Complete Search: num_requests <= threshold
        if (len(new_path.request_order) <= THRESHOLD):
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

        # Explore all combinations of requests
        stack: List[List[PathNode]] = []
        current_request_order: List[PathNode] = []

        while stack or possible_next_nodes:
            if possible_next_nodes:
                # Expand the current tree
                next_node = possible_next_nodes.pop()

                # If adding the next node is not feasible, continue
                next_location, deadline = path.get_info(next_node)
                if not current_request_order:
                    current_location = agent.position.next_location
                    current_time = self.envt.current_time + agent.position.time_to_next_location
                    current_capacity: int = path.current_capacity
                    remaining_delay: float = 0.0
                else:
                    current_location, _ = path.get_info(current_request_order[-1])
                    current_time = current_request_order[-1].expected_visit_time
                    current_capacity = current_request_order[-1].current_capacity
                    remaining_delay = current_request_order[-1].cumulative_remaining_delay

                travel_time = self.envt.get_travel_time(current_location, next_location)
                time_at_next_location = current_time + travel_time

                # If pick-up, check if it violates the current capacity
                if (not next_node.is_dropoff and current_capacity + 1 > self.envt.MAX_CAPACITY):
                    continue

                # Check if it meets deadline
                if (time_at_next_location > deadline):
                    continue

                # Add to the current path, given it is feasible
                next_node.expected_visit_time = time_at_next_location
                if (next_node.is_dropoff):
                    next_node.current_capacity = current_capacity - 1
                    next_node.cumulative_remaining_delay = remaining_delay + (deadline - time_at_next_location)  # only dropoff delay is relevant
                else:
                    next_node.current_capacity = current_capacity + 1
                    next_node.cumulative_remaining_delay = remaining_delay

                current_request_order.append(next_node)

                # Go one step deeper into the search space
                stack.append(deepcopy(possible_next_nodes))

                # If it is a pickup location, add dropoff to possible_next_nodes
                if not next_node.is_dropoff:
                    corresponding_dropoff = PathNode(True, next_node.relevant_request_id)
                    possible_next_nodes.append(corresponding_dropoff)

            else:
                # Check if this path has been completed
                if (path.is_complete(current_request_order)):
                    # Check if this is the best path
                    current_remaining_delay = current_request_order[-1].cumulative_remaining_delay
                    previous_lowest_delay = path.total_delay
                    if (previous_lowest_delay == Path.NOT_APPLICABLE or current_remaining_delay > previous_lowest_delay):
                        # Save, if it is
                        path.request_order = deepcopy(current_request_order)
                        path.total_delay = current_remaining_delay

                # Take a step back
                possible_next_nodes = stack.pop()
                current_request_order.pop()

        return path

    def get_new_path_insertion(self, agent: LearningAgent, path: Path) -> Path:
        # DROPOFF
        relevant_request_id = len(path.requests) - 1  # Assume the new request is the last one
        is_dropoff = True
        new_node = PathNode(is_dropoff, relevant_request_id)
        path, insert_idx = self.insert_path(agent, path, new_node)

        # PICKUP
        if (insert_idx != Path.NOT_APPLICABLE):
            is_dropoff = False
            new_node = PathNode(is_dropoff, relevant_request_id)
            path, _ = self.insert_path(agent, path, new_node, insert_idx)

        return path

    def insert_path(self, agent: LearningAgent, path: Path, new_node: PathNode, max_insert_idx: int=-1) -> Tuple[Path, int]:
        # Find insertion point with minimum dropoff delay
        location, deadline = path.get_info(new_node)
        is_dropoff = new_node.is_dropoff
        if (max_insert_idx == -1):
            max_insert_idx = len(path.request_order)

        current_remaining_delay = path.total_delay
        num_dropoffs_after = 0
        min_future_delay = float('inf')
        min_total_delay = float('inf')
        min_delay_idx = Path.NOT_APPLICABLE

        # Check all insertion points by iterating backward through the request order
        max_index = len(path.request_order) - 1
        prev_node = None
        for insert_idx in range(max_index, -2, -1):
            next_node = prev_node

            # Get info about prev_node
            if (insert_idx != Path.NOT_APPLICABLE):
                prev_node = path.request_order[insert_idx]
                prev_location, prev_deadline = path.get_info(prev_node)
                visit_time = prev_node.expected_visit_time
                current_capacity = prev_node.current_capacity
            else:
                prev_node = None
                prev_location = agent.position.next_location
                prev_deadline = float('inf')
                visit_time = self.envt.current_time + agent.position.time_to_next_location
                current_capacity = path.current_capacity

            # If pickup node, only insert if it's before dropoff node
            if (insert_idx < max_insert_idx):
                # Check if it violates the capacity constraint
                if (is_dropoff or current_capacity < self.envt.MAX_CAPACITY):
                    # Check if it meets later nodes' deadlines
                    travel_time_prev = self.envt.get_travel_time(prev_location, location)
                    delay = 0
                    if (next_node):
                        next_location, _ = path.get_info(next_node)
                        travel_time_next = self.envt.get_travel_time(location, next_location)
                        travel_time_default = self.envt.get_travel_time(prev_location, next_location)
                        delay = travel_time_prev + travel_time_next - travel_time_default

                    if (delay <= min_future_delay):
                        # Check if inserted node meets its own deadline
                        if (deadline >= visit_time + travel_time_prev):
                            # Check if it has less delay than previous feasible paths
                            delay_for_new_node = 0
                            if (is_dropoff):
                                delay_for_new_node = deadline - (visit_time + travel_time_prev)
                            total_delay = current_remaining_delay - (delay * num_dropoffs_after) + delay_for_new_node

                            # Save if it has less delay
                            if (total_delay <= min_total_delay):
                                min_total_delay = total_delay
                                min_individual_delay = delay
                                min_delay_idx = insert_idx + 1

                # Make sure you don't violate the capacity constraint at any point
                else:
                    break

            # State update
            prev_node_max_delay = prev_deadline - visit_time
            min_future_delay = min(min_future_delay, prev_node_max_delay)
            if prev_node and prev_node.is_dropoff:
                num_dropoffs_after += 1

        # Insert into path
        if (min_delay_idx != Path.NOT_APPLICABLE):
            # Insert new node into request order
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

            path.request_order.insert(min_delay_idx, new_node)

            # Update request_order
            for idx in range(min_delay_idx + 1, len(path.request_order)):
                node = path.request_order[idx]
                node.expected_visit_time += min_individual_delay
                if not is_dropoff and idx <= max_insert_idx:
                    node.current_capacity += 1

            # Update total_delay
            path.total_delay = min_total_delay

        # If no insertion point was found, abort
        return path, min_delay_idx
