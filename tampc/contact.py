from tampc.dynamics import online_model

class ContactObject:
    def __init__(self, empty_local_model: online_model.OnlineDynamicsModel):
        # (x, u, dx) tuples associated with this object for fitting local model
        self.transitions = []
        # points on this object that are tracked
        self.points = []
        self.actions = []
        self.dynamics = empty_local_model

    def add_transition(self, x, u, dx):
        self.points.append(x)
        self.actions.append(u)
        # move all points by dx
        # TODO generalize the x + dx operation
        self.points = [xx + dx for xx in self.points]

        self.transitions.append((x, u, dx))
        self.dynamics.update(x, u, x + dx)

    def is_part_of_object(self, x, u, length_parameter, dist_function, u_similarity):
        for i in range(len(self.points)):
            # distance in state x action space, with u_similarity giving comparison in action space
            u_sim = u_similarity(u, self.actions[i])
            if u_sim == 0:
                continue
            if dist_function(x, self.points[i]) / u_sim < length_parameter:
                return True
        return False
