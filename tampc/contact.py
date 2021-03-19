import torch
import typing
import copy
from tampc.dynamics import online_model
from arm_pytorch_utilities import tensor_utils, optim


class ContactObject:
    def __init__(self, empty_local_model: online_model.OnlineDynamicsModel, object_centered_dynamics=True,
                 assume_linear_scaling=True):
        # (x, u, dx) tuples associated with this object for fitting local model
        self.transitions = []
        # points on this object that are tracked
        self.points = None
        self.center_point = None
        self.actions = None
        self.dynamics = empty_local_model

        self.object_centered = object_centered_dynamics
        self.assume_linear_scaling = assume_linear_scaling

    def add_transition(self, x, u, dx):
        dtype = torch.float64 if not torch.is_tensor(x) else x.dtype
        x, u, dx = tensor_utils.ensure_tensor(optim.get_device(), dtype, x, u, dx)
        if self.points is None:
            self.points = x.view(1, -1)
            self.actions = u.view(1, -1)
            self.center_point = x
        else:
            self.points = torch.cat((self.points, x.view(1, -1)))
            self.actions = torch.cat((self.actions, u.view(1, -1)))

        centered_x = x - self.center_point if self.object_centered else x
        self.move_all_points(dx)

        self.transitions.append((x, u, dx))

        if self.assume_linear_scaling:
            u_scale = u.norm()
            u /= u_scale
            dx /= u_scale
        self.dynamics.update(centered_x, u, centered_x + dx)

    def move_all_points(self, dx):
        # TODO generalize the x + dx operation
        self.points += dx
        self.center_point = self.points.mean(dim=0)

    @tensor_utils.ensure_2d_input
    def predict(self, x, u, **kwargs):
        if self.object_centered:
            x = x - self.center_point

        u_scale = 1
        if self.assume_linear_scaling:
            u_scale = u.norm(dim=1)
            u /= u_scale.view(-1, 1)

        nx = self.dynamics.predict(None, None, x, u, **kwargs)
        if self.assume_linear_scaling:
            dx = nx - x
            dx *= u_scale.view(-1, 1)
            nx = x + dx

        if self.object_centered:
            nx = nx + self.center_point
        return nx

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


class ContactSet:
    def __init__(self, contact_max_linkage_dist, state_dist, u_sim):
        self._obj: typing.List[ContactObject] = []
        # cached center of all points
        self.center_points = None

        self.contact_max_linkage_dist = contact_max_linkage_dist
        self.state_dist = state_dist
        self.u_sim = u_sim

    def __len__(self):
        return len(self._obj)

    def __iter__(self):
        return self._obj.__iter__()

    def append(self, contact: ContactObject):
        self._obj.append(contact)

    def updated(self):
        if self._obj:
            self.center_points = torch.stack([obj.center_point for obj in self._obj])

    def move_object(self, dx, obj_index):
        # new_set = copy.deepcopy(self)
        # new_set._obj[obj_index].move_all_points(dx)
        # new_set.center_points[obj_index] = new_set._obj[obj_index].center_point

        new_set = copy.copy(self)
        new_set._obj = [copy.deepcopy(self._obj[i]) if i is obj_index else self._obj[i] for i in range(len(self))]
        new_set._obj[obj_index].move_all_points(dx)
        new_set.center_points = new_set.center_points.clone()
        new_set.center_points[obj_index] = new_set._obj[obj_index].center_point
        return new_set

    def goal_cost(self, goal_x):
        if not self._obj:
            return 0

        d = self.state_dist(self.center_points, goal_x).view(-1)
        return (1 / d).sum()

    def check_which_object_applies(self, x, u):
        if not self._obj:
            return None, None

        d = self.state_dist(self.center_points, x).view(-1)

        for i, cc in enumerate(self._obj):
            # first use heuristic to filter out points based on state distance to object centers
            if d[i] > 2 * self.contact_max_linkage_dist:
                continue
            # we're using the x before contact because our estimate of the object points haven't moved yet
            # TODO handle when multiple contact objects claim it is part of them
            if cc.is_part_of_object(x, u, self.contact_max_linkage_dist, self.state_dist, self.u_sim):
                return cc, i

        return None, None

    # @tensor_utils.ensure_2d_input
    # def check_which_object_applies(self, x, u):
    #     N = x.shape[0]
    #     res: typing.List[typing.Optional[ContactObject]] = [None for _ in range(N)]
    #     if not self._obj:
    #         return res
    #
    #     for cc in self._obj:
    #         # first use heuristic to filter out points based on state distance to object centers
    #         d = self.state_dist(cc.center_point, x)
    #         if d.min() < 2 * self.contact_max_linkage_dist:
    #             continue
    #         # we're using the x before contact because our estimate of the object points haven't moved yet
    #         # TODO handle when multiple contact objects claim it is part of them
    #         applicable = cc.is_part_of_object_batch(x, u, self.contact_max_linkage_dist, self.state_dist, self.u_sim)
    #         for i in range(N):
    #             if applicable[i]:
    #                 res[i] = cc
    #     return res
