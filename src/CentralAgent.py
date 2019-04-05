from Request import Request
from Action import Action
from LearningAgent import LearningAgent

from typing import List, Dict, Tuple, Set, Any, Optional

from docplex.mp.model import Model
from docplex.mp.linear import Var
from random import gauss, shuffle, randint


# TODO: Factor out value function into a separate class
class CentralAgent(object):
    """
    A CentralAgent arbitrates between different Agents.

    It takes the users 'preferences' for different actions as input
    and chooses the combination that maximises the sum of utilities
    for all the agents.

    It also trains the Learning Agents' shared value function by
    querying the rewards from the environment and the next state.
    """

    def __init__(self):
        super(CentralAgent, self).__init__()

    def choose_actions(self, agent_action_choices: List[List[Tuple[Action, float]]], is_training: bool=True, epoch_num: int=1) -> List[Action]:
        # TODO: Implement notion of a "random" action
        return self.choose_actions_ILP(agent_action_choices, is_training, epoch_num)

    def choose_actions_ILP(self, agent_action_choices: List[List[Tuple[Action, float]]], is_training: bool=True, epoch_num: int=1) -> List[Action]:
        # Model as ILP
        model = Model()

        # For converting Action -> action_id and back
        action_to_id: Dict[Action, int] = {}
        id_to_action: Dict[int, Action] = {}
        current_action_id = 0

        # For constraint 2
        requests: Set[Request] = set()

        # Create decision variables and their coefficients in the objective
        # There is a decision variable for each (Action, Agent).
        # The coefficient is the value associated with the decision variable
        decision_variables: Dict[int, Dict[int, Tuple[Any, float]]] = {}
        for agent_idx, scored_actions in enumerate(agent_action_choices):
            for action, value in scored_actions:
                # Convert action -> id if it hasn't already been done
                if action not in action_to_id:
                    action_to_id[action] = current_action_id
                    id_to_action[current_action_id] = action
                    current_action_id += 1

                    action_id = current_action_id - 1
                    decision_variables[action_id] = {}
                else:
                    action_id = action_to_id[action]

                # Update set of requests in actions
                for request in action.requests:
                    if request not in requests:
                        requests.add(request)

                # Create variable for (action_id, agent_id)
                variable = model.binary_var(name='x{},{}'.format(action_id, agent_idx))

                # Save to decision_variable data structure
                decision_variables[action_id][agent_idx] = (variable, value)

        # Create Constraint 1: Only one action per Agent
        for agent_idx in range(len(agent_action_choices)):
            agent_specific_variables: List[Any] = []
            for action_dict in decision_variables.values():
                if agent_idx in action_dict:
                    agent_specific_variables.append(action_dict[agent_idx])
            model.add_constraint(model.sum(variable for variable, _ in agent_specific_variables) == 1)

        # Create Constraint 2: Only one action per Request
        for request in requests:
            relevent_action_dicts: List[Dict[int, Tuple[Any, float]]] = []
            for action_id in decision_variables:
                if (request in id_to_action[action_id].requests):
                    relevent_action_dicts.append(decision_variables[action_id])
            model.add_constraint(model.sum(variable for action_dict in relevent_action_dicts for variable, _ in action_dict.values()) <= 1)

        # Create Objective
        # Add noise during training for exploration
        def get_noise(variable: Var) -> float:
            stdev = (4000 if 'x0,' in variable.get_name() else 2000) / (epoch_num + 1000)
            return abs(gauss(0, stdev) if is_training else 0)

        score = model.sum((value + get_noise(variable)) * variable for action_dict in decision_variables.values() for (variable, value) in action_dict.values())
        model.maximize(score)

        # Solve ILP
        solution = model.solve()
        assert solution  # making sure that the model doesn't fail

        # Get vehicle specific actions from ILP solution
        assigned_actions: Dict[int, int] = {}
        for action_id, action_dict in decision_variables.items():
            for agent_idx, (variable, _) in action_dict.items():
                if (solution.get_value(variable) == 1):
                    assigned_actions[agent_idx] = action_id

        final_actions: List[Action] = []
        for agent_idx in range(len(agent_action_choices)):
            assigned_action_id = assigned_actions[agent_idx]
            assigned_action = id_to_action[assigned_action_id]
            final_action = None
            for action, _ in agent_action_choices[agent_idx]:
                if (action == assigned_action):
                    final_action = action
                    break

            assert final_action is not None
            final_actions.append(final_action)

        return final_actions

    def _choose_actions_random(self, agent_action_choices: List[List[Tuple[Action, float]]], is_training: bool=True, epoch_num: int=1) -> List[Action]:
        final_actions: List[Optional[Action]] = [None] * len(agent_action_choices)
        consumed_requests: Set[Request] = set()

        # Create a random ordering
        order = list(range(len(agent_action_choices)))
        shuffle(order)

        # Pick agents in a random order
        for agent_idx in order:
            # Create a list of feasible actions
            allowable_actions_idxs: List[int] = []
            for action_idx, (action, _) in enumerate(agent_action_choices[agent_idx]):
                is_not_consumed = [(request in consumed_requests) for request in action.requests]
                if sum(is_not_consumed) == 0:
                    allowable_actions_idxs.append(action_idx)

            # Pick a random feasible action
            final_action_idx = randint(0, len(allowable_actions_idxs) - 1)
            final_action, _ = agent_action_choices[agent_idx][final_action_idx]
            final_actions[agent_idx] = final_action

            # Update inefasible action information
            for request in final_action.requests:
                consumed_requests.add(request)

        for action in final_actions:
            assert action is not None

        return final_actions
