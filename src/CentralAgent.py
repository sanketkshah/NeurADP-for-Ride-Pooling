from Request import Request
from Action import Action
from LearningAgent import LearningAgent

from typing import List, Dict, Tuple, Set, Any

from docplex.mp.model import Model


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
        self.__initialise_value_function()

    def __initialise_value_function(self):
        pass

    def update_value_function(self, current_rewards, current_state, current_actions):
        # TODO: Fill this in
        # Perform update
        # Save current rewards for use in next epoch
        pass

    def get_value(self, agent: LearningAgent, feasible_actions: List[Action]) -> List[Tuple[Action, float]]:
        scored_actions: List[Tuple[Action, float]] = []
        for action in feasible_actions:
            score = len(action.requests)
            scored_actions.append((action, score))

        return scored_actions

    def choose_actions(self, agent_action_choices: List[List[Tuple[Action, float]]]) -> List[Action]:
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
        score = model.sum(value * variable for action_dict in decision_variables.values() for (variable, value) in action_dict.values())
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
