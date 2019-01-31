class Request(object):
    """
    A Request is the atomic unit in an Action.

    It represents a single customer's *request* for a ride
    """

    def __init__(self, source, destination, value=1):
        super(Request, self).__init__()

        self.source: int = source
        self.destination: int = destination
        self.value: float = value  # In the deafult case, all requests have equal value
