from typing import List
from Action import Action


class CentralAgent(object):
    """
    A CentralAgent arbitrates between different Agents.

    It takes the users 'preferences' for different actions as input
    and chooses the combination that maximises the sum of utilities
    for all the agents.

    It also trains the Learning Agents' shared value function by
    querying the rewards from the environment and the next state.
    """

    def __init__(self, arg):
        super(CentralAgent).__init__()

        self.__initialise_value_function()

    def __initialise_value_function():
        pass

    def update_value_function(self, current_rewards, current_state, current_actions):
        # Perform update

        # Save current rewards for use in next epoch
        pass

    def choose_actions(self, agent_action_choices: List[List[(Action, float)]]): 
        pass
