class LearningAgent(object):
    """
    A LearningAgent corresponds to a single decision making unit.

    In our formulation a learning agent corresponds to a single
    vehicle. It learns a value function based on the rewards it gets
    from the environment. It generates prefences for different actions
    using this value funciton and submits it to the CentralAgent for
    arbitration.
    """

    def __init__(self, current_location, value_function):
        super(LearningAgent, self).__init__()

        # Initialising the state of the agent
        self.current_location = current_location
        self.time_to_current_location = 0

        self.active_requests = []
        self.current_path = []

        # Loading the value function
        self.value_function = value_function

    def get_value(self, action):
        pass
