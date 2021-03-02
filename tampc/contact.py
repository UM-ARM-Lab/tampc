class ContactObject:
    def __init__(self):
        # (x, u, dx) tuples associated with this object for fitting local model
        self.transitions = []
        # points on this object that are tracked
        self.points = []

    def add_transition(self, x, u, dx):
        self.transitions.append((x, u, dx))
        self.points.append(x)
        # move all points by dx
        # TODO generalize the x + dx operation
        self.points = [xx + dx for xx in self.points]

    def is_part_of_object(self, x, length_parameter, dist_function):
        for xx in self.points:
            if dist_function(x, xx) < length_parameter:
                return True
        return False
