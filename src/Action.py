class Action(object):
    """
    An Action is the output of an Agent for a decision epoch.

    In our formulation corresponds to an Agent accepting a given set
    of Requests.
    """

    def __init__(self, requests):
        super(Action, self).__init__()

        self.requests = set(requests)
        self.new_path = []
