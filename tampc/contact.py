import torch
import typing
import copy
from tampc.dynamics import online_model
from arm_pytorch_utilities import tensor_utils, optim


class ContactObject:
    def __init__(self, empty_local_model: online_model.OnlineDynamicsModel, state_to_pos, pos_to_state,
                 assume_linear_scaling=True):
        # (x, u, dx) tuples associated with this object for fitting local model
        self.transitions = []
        # points on this object that are tracked
        self.points = None
        self.center_point = None
        self.actions = None
        self.dynamics = empty_local_model
        self.state_to_pos = state_to_pos
        self.pos_to_state = pos_to_state

        self.assume_linear_scaling = assume_linear_scaling

    def add_transition(self, x, u, dx):
        dtype = torch.float64 if not torch.is_tensor(x) else x.dtype
        x, u, dx = tensor_utils.ensure_tensor(optim.get_device(), dtype, x, u, dx)
        # save only positions
        x = self.state_to_pos(x)
        if self.points is None:
            self.points = x.view(1, -1)
            self.actions = u.view(1, -1)
            self.center_point = x
        else:
            self.points = torch.cat((self.points, x.view(1, -1)))
            self.actions = torch.cat((self.actions, u.view(1, -1)))

        centered_x = x - self.center_point
        self.move_all_points(self.state_to_pos(dx))

        # self.transitions.append((x, u, dx))

        if self.assume_linear_scaling:
            u_scale = u.norm()
            u /= u_scale
            dx /= u_scale
        # convert back to state
        centered_x = self.pos_to_state(centered_x).view(-1)
        self.dynamics.update(centered_x, u, centered_x + dx)

    def move_all_points(self, dx):
        self.points += dx
        self.center_point = self.points.mean(dim=0)

    @tensor_utils.ensure_2d_input
    def predict_dpos(self, pos, u, **kwargs):
        u_scale = 1
        if self.assume_linear_scaling:
            u_scale = u.norm(dim=1)
            u /= u_scale.view(-1, 1)

        # nx here is next position relative to center point
        nx = self.dynamics.predict(None, None, self.pos_to_state(pos), u, **kwargs)
        npos = self.state_to_pos(nx)
        dpos = npos - pos
        if self.assume_linear_scaling:
            dpos *= u_scale.view(-1, 1)
        return dpos

    @tensor_utils.ensure_2d_input
    def predict(self, x, u, **kwargs):
        x = self.state_to_pos(x)
        x = x - self.center_point

        dpos = self.predict_dpos(x, u, **kwargs)
        npos = x + dpos

        npos = npos + self.center_point
        return self.pos_to_state(npos), dpos

    def is_part_of_object(self, x, u, length_parameter, u_similarity):
        u_sim = u_similarity(u, self.actions)
        valid = u_sim != 0
        p, u = self.points[valid], u_sim[valid]
        if len(p) is 0:
            return False
        x = self.state_to_pos(x)
        d = (x - p).norm(dim=1) / u
        return torch.any(d < length_parameter)

    def is_part_of_object_batch(self, x, u, length_parameter, u_similarity):
        res = []
        for i in range(len(x)):
            res.append(self.is_part_of_object(x[i], u[i], length_parameter, u_similarity))
        return torch.tensor(res, device=optim.get_device())


class ContactSet:
    def __init__(self, contact_max_linkage_dist, state_to_pos, u_sim):
        self._obj: typing.List[ContactObject] = []
        # cached center of all points
        self.center_points = None

        self.contact_max_linkage_dist = contact_max_linkage_dist
        self.u_sim = u_sim
        self.state_to_pos = state_to_pos

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
        new_set = copy.copy(self)
        new_set._obj = [copy.deepcopy(self._obj[i]) if i is obj_index else self._obj[i] for i in range(len(self))]
        new_set._obj[obj_index].move_all_points(dx)
        new_set.center_points = new_set.center_points.clone()
        new_set.center_points[obj_index] = new_set._obj[obj_index].center_point
        return new_set

    def goal_cost(self, goal_x, contact_data):
        if not self._obj:
            return 0

        center_points, points, actions = contact_data
        # norm across spacial dimension, sum across each object
        d = (center_points - self.state_to_pos(goal_x)).norm(dim=-1)
        return (1 / d).sum(dim=0)

    def check_which_object_applies(self, x, u):
        if not self._obj:
            return None, None

        d = (self.center_points - self.state_to_pos(x)).norm(dim=1).view(-1)

        for i, cc in enumerate(self._obj):
            # first use heuristic to filter out points based on state distance to object centers
            if d[i] > 2 * self.contact_max_linkage_dist:
                continue
            # we're using the x before contact because our estimate of the object points haven't moved yet
            # TODO handle when multiple contact objects claim it is part of them
            if cc.is_part_of_object(x, u, self.contact_max_linkage_dist, self.u_sim):
                return cc, i

        return None, None

    def get_batch_data_for_dynamics(self, total_num):
        if not self._obj:
            return None, None, None
        center_points = self.center_points.repeat(total_num, 1, 1).transpose(0, 1)
        points = [c.points.repeat(total_num, 1, 1).transpose(0, -2) for c in self._obj]
        actions = [c.actions.repeat(total_num, 1, 1).transpose(0, -2) for c in self._obj]
        return center_points, points, actions

    def dynamics(self, x, u, contact_data):
        center_points, points, actions = contact_data
        assert len(x.shape) == 2 and len(x.shape) == len(u.shape)
        total_num = x.shape[0]
        without_contact = torch.ones(total_num, dtype=torch.bool, device=x.device)

        # all of t
        if center_points is not None:
            pos = self.state_to_pos(x)
            rel_pos = pos - center_points
            d = rel_pos.norm(dim=-1)

            # loop over objects (can't batch over this since each one has a different dynamics function)
            for i in range(len(self._obj)):
                c = self._obj[i]
                pts = points[i]
                act = actions[i]
                dd = d[i]
                # first filter on distance to center of this object
                # disallow multiple contact interactions
                candidates = without_contact & (dd < 2 * self.contact_max_linkage_dist)
                if not torch.any(candidates):
                    continue

                # u_sim[stored action index (# points of object), total_num], sim of stored action and sampled action
                u_sim = self.u_sim(u[candidates], act[:, candidates]) + 1e-8  # avoid divide by 0
                dd = (pos[candidates] - pts[:, candidates]).norm(dim=-1) / u_sim
                accepted = torch.any(dd < self.contact_max_linkage_dist, dim=0)
                if not torch.any(accepted):
                    continue
                candidates[candidates] = accepted

                # now the candidates are fully filtered, can apply dynamics to those
                dpos = c.predict_dpos(rel_pos[i, candidates], u[candidates])
                # pos is already rel_pos + center_points
                npos = pos[candidates] + dpos
                x[candidates] = c.pos_to_state(npos)
                without_contact[candidates] = False
                # move the center points of those states
                center_points[i, candidates] += dpos
                pts[:, candidates] += dpos

        return x, without_contact, (center_points, points, actions)
