import torch
from tampc.dynamics import online_model
from arm_pytorch_utilities import tensor_utils, optim


class ContactObject:
    def __init__(self, empty_local_model: online_model.OnlineDynamicsModel):
        # (x, u, dx) tuples associated with this object for fitting local model
        self.transitions = []
        # points on this object that are tracked
        self.points = None
        self.actions = None
        self.dynamics = empty_local_model

    def add_transition(self, x, u, dx):
        dtype = torch.float64 if not torch.is_tensor(x) else x.dtype
        x, u, dx = tensor_utils.ensure_tensor(optim.get_device(), dtype, x, u, dx)
        if self.points is None:
            self.points = x
            self.actions = u
        else:
            self.points = torch.cat((self.points.view(-1, x.numel()), x.view(1, -1)))
            self.actions = torch.cat((self.actions.view(-1, u.numel()), u.view(1, -1)))

        # move all points by dx
        # TODO generalize the x + dx operation
        self.points += dx
        # self.points = [xx + dx for xx in self.points]

        self.transitions.append((x, u, dx))
        self.dynamics.update(x, u, x + dx)

    def is_part_of_object(self, x, u, length_parameter, dist_function, u_similarity):
        u_sim = u_similarity(u, self.actions)
        valid = u_sim != 0
        p, u = self.points[valid], u_sim[valid]
        if len(p) is 0:
            return False
        d = dist_function(x, p) / u
        return torch.any(d < length_parameter)

    def is_part_of_object_batch(self, x, u, length_parameter, dist_function, u_similarity):
        res = []
        for i in range(len(x)):
            res.append(self.is_part_of_object(x[i], u[i], length_parameter, dist_function, u_similarity))
        return torch.tensor(res, device=optim.get_device())
