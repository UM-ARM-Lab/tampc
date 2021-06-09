import torch
import typing
import copy
import logging
from tampc.dynamics import online_model
from arm_pytorch_utilities import tensor_utils, optim, serialization

logger = logging.getLogger(__name__)


class ContactObject(serialization.Serializable):
    def __init__(self, empty_local_model: online_model.OnlineDynamicsModel, state_to_pos, pos_to_state,
                 assume_linear_scaling=True):
        # (x, u, dx) tuples associated with this object for fitting local model
        self.transitions = []
        # points on this object that are tracked
        self.points = None
        self.center_point = None
        self.actions = None
        self.probability = 1

        self.dynamics = empty_local_model
        self.state_to_pos = state_to_pos
        self.pos_to_state = pos_to_state

        self.assume_linear_scaling = assume_linear_scaling

        # UKF stuff

    def state_dict(self) -> dict:
        state = {'points': self.points,
                 'center_point': self.center_point,
                 'actions': self.actions,
                 'probability': self.probability,
                 'dynamics': self.dynamics.state_dict()
                 }
        return state

    def load_state_dict(self, state: dict) -> bool:
        self.points = state['points']
        self.center_point = state['center_point']
        self.actions = state['actions']
        self.probability = state['probability']
        if not self.dynamics.load_state_dict(state['dynamics']):
            return False
        return True

    def add_transition(self, x, u, dx):
        self.probability = -1  # dummy value to indicate for updated to set it to 1
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
        # TODO change existing data to be relative to new data center point
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

    def merge_objects(self, other_objects):
        self.points = torch.cat([self.points] + [obj.points for obj in other_objects])
        self.actions = torch.cat([self.actions] + [obj.actions for obj in other_objects])
        self.center_point = self.points.mean(dim=0)

    @tensor_utils.ensure_2d_input
    def predict_dpos(self, pos, u, **kwargs):
        # pos is relative to the object center
        u_scale = 1
        if self.assume_linear_scaling:
            u_scale = u.norm(dim=1)
            to_scale = u_scale != 0
            u[to_scale] /= u_scale[to_scale].view(-1, 1)

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

    def is_part_of_object(self, x, u, length_parameter, u_similarity, x_is_pos=False):
        u_sim = u_similarity(u, self.actions)
        valid = u_sim != 0
        p, u = self.points[valid], u_sim[valid]
        if len(p) is 0:
            return False
        if not x_is_pos:
            x = self.state_to_pos(x)
        d = (x - p).norm(dim=1) / u
        return torch.any(d < length_parameter)

    def is_part_of_object_batch(self, x, u, length_parameter, u_similarity, candidates=None, act=None, pts=None,
                                x_is_pos=False):
        total_num = x.shape[0]
        if x_is_pos:
            pos = x
        else:
            pos = self.state_to_pos(x)

        if candidates is None:
            candidates = torch.ones(total_num, dtype=torch.bool, device=x.device)
        if act is None:
            act = self.actions
        if pts is None:
            pts = self.points

        u_sim = u_similarity(u[candidates], act[:, candidates]) + 1e-8  # avoid divide by 0
        dd = (pos[candidates] - pts[:, candidates]).norm(dim=-1) / u_sim
        accepted = torch.any(dd < length_parameter, dim=0)
        if not torch.any(accepted):
            return candidates, True
        candidates[candidates] = accepted
        return candidates, False


class ContactSet(serialization.Serializable):
    def __init__(self, contact_max_linkage_dist, state_to_pos, u_sim, fade_per_contact=0.8,
                 fade_per_no_contact=0.95,
                 ignore_below_probability=0.25, immovable_collision_checker=None,
                 contact_object_factory=None,
                 contact_force_threshold=0.5):
        self._obj: typing.List[ContactObject] = []
        # cached center of all points
        self.center_points = None

        self.contact_max_linkage_dist = contact_max_linkage_dist
        self.u_sim = u_sim
        self.state_to_pos = state_to_pos
        self.fade_per_contact = fade_per_contact
        self.fade_per_no_contact = fade_per_no_contact
        self.ignore_below_probability = ignore_below_probability

        self.immovable_collision_checker = immovable_collision_checker
        # used to produce contact objects during loading
        self.contact_object_factory = contact_object_factory
        # process and measurement model uses this to decide when force is high or low magnitude
        self.contact_force_threshold = contact_force_threshold

    def state_dict(self) -> dict:
        state = {'center_points': self.center_points, 'num_obj': len(self._obj)}
        for i, obj in enumerate(self._obj):
            state.update({'o{}'.format(i): obj.state_dict()})
        return state

    def load_state_dict(self, state: dict) -> bool:
        self.center_points = state['center_points']
        if self.contact_object_factory is None:
            logger.error("Need contact object factory in contact set to create contact objects during loading")
            return False
        n = state['num_obj']
        self._obj = []
        for i in range(n):
            c = self.contact_object_factory()
            if not c.load_state_dict(state['o{}'.format(i)]):
                return False
            self._obj.append(c)
        return True

    def __len__(self):
        return len(self._obj)

    def __iter__(self):
        return self._obj.__iter__()

    def append(self, contact: ContactObject):
        self._obj.append(contact)

    def _keep_high_prob_contacts(self):
        if self._obj:
            init_len = len(self._obj)
            self._obj = [c for c in self._obj if c.probability > self.ignore_below_probability]
            if len(self._obj) != init_len:
                if len(self._obj):
                    self.center_points = torch.stack([obj.center_point for obj in self._obj])
                else:
                    self.center_points = None

    def updated(self):
        if self._obj:
            for c in self._obj:
                if c.probability is -1:
                    c.probability = 1
                else:
                    c.probability *= self.fade_per_contact
            self._keep_high_prob_contacts()
        if self._obj:
            self.center_points = torch.stack([obj.center_point for obj in self._obj])

    def stepped_without_contact(self):
        for c in self._obj:
            c.probability *= self.fade_per_no_contact
        self._keep_high_prob_contacts()

    def move_object(self, dx, obj_index):
        new_set = copy.copy(self)
        new_set._obj = [copy.deepcopy(self._obj[i]) if i is obj_index else self._obj[i] for i in range(len(self))]
        new_set._obj[obj_index].move_all_points(dx)
        new_set.center_points = new_set.center_points.clone()
        new_set.center_points[obj_index] = new_set._obj[obj_index].center_point
        return new_set

    def merge_objects(self, obj_indices):
        obj_to_combine = [self._obj[i] for i in obj_indices]
        self._obj = [obj for obj in self._obj if obj not in obj_to_combine]
        c = copy.deepcopy(obj_to_combine[0])
        c.merge_objects(obj_to_combine[1:])
        self.append(c)
        return c

    def goal_cost(self, goal_x, contact_data):
        if not self._obj:
            return 0

        center_points, points, actions = contact_data
        # norm across spacial dimension, sum across each object
        d = (center_points - self.state_to_pos(goal_x)).norm(dim=-1)
        # modify by probability of contact object existing
        prob = torch.tensor([c.probability for c in self._obj], dtype=d.dtype, device=d.device)
        return (1 / d * prob.view(-1, 1)).sum(dim=0)

    def check_which_object_applies(self, x, u):
        res_c = []
        res_i = []
        if not self._obj:
            return res_c, res_i

        d = (self.center_points - self.state_to_pos(x)).norm(dim=1).view(-1)

        for i, cc in enumerate(self._obj):
            # first use heuristic to filter out points based on state distance to object centers
            if d[i] > 2 * self.contact_max_linkage_dist:
                continue
            # we're using the x before contact because our estimate of the object points haven't moved yet
            # TODO handle when multiple contact objects claim it is part of them
            if cc.is_part_of_object(x, u, self.contact_max_linkage_dist, self.u_sim):
                res_c.append(cc)
                res_i.append(i)
                # still can't deal with merging very well, so revert back to returning a single object
                break

        return res_c, res_i

    def update(self, x, u, dx, reaction):
        # TODO move this inside the measurement and process model of the UKFs?
        if reaction.norm() < self.contact_force_threshold:
            self.stepped_without_contact()
            return

        # associate each contact to a single object (max likelihood estimate on which object it is)
        cc, ii = self.check_which_object_applies(x, u)
        # couldn't find an existing contact
        if not len(cc):
            # if using object-centered model, don't use preprocessor, else use default
            c = self.contact_object_factory()
            self.append(c)
        # matches more than 1 contact set, combine them
        elif len(cc) > 1:
            c = self.merge_objects(ii)
        else:
            c = cc[0]
        c.add_transition(x, u, dx)
        self.updated()

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
                candidates, none = c.is_part_of_object_batch(pos, u, self.contact_max_linkage_dist, self.u_sim,
                                                             candidates=candidates, act=act, pts=pts, x_is_pos=True)
                if none:
                    continue

                # now the candidates are fully filtered, can apply dynamics to those
                dpos = c.predict_dpos(rel_pos[i, candidates], u[candidates])
                # pos is already rel_pos + center_points
                ppos = pos[candidates]
                npos = ppos + dpos
                x[candidates] = c.pos_to_state(npos)

                if self.immovable_collision_checker is not None:
                    _, nx = self.immovable_collision_checker(x[candidates])
                    dpos = c.state_to_pos(nx) - ppos
                    x[candidates] = nx

                without_contact[candidates] = False
                # move the center points of those states
                center_points[i, candidates] += dpos
                pts[:, candidates] += dpos

        return x, without_contact, (center_points, points, actions)
