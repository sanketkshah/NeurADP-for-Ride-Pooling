class Request(object):
    """
    A Request is the atomic unit in an Action.

    It represents a single customer's *request* for a ride
    """

    # TODO: Define __str__ for the class

    MAX_PICKUP_DELAY: float = 300.0
    MAX_DROPOFF_DELAY: float = 600.0

    def __init__(self,
                 source: int,
                 destination: int,
                 current_time: float,
                 travel_time: float,
                 value: float=1,
                 ):
        self.pickup: int = source
        self.dropoff: int = destination
        self.value: float = value  # In the deafult case, all requests have equal value
        self.pickup_deadline = current_time + self.MAX_PICKUP_DELAY
        self.dropoff_deadline = current_time + travel_time + self.MAX_DROPOFF_DELAY

    def __deepcopy__(self, memo):
        return self

    def __str__(self):
        return("{}->{}".format(self.pickup, self.dropoff))

    def __repr__(self):
        return str(self)
