from dataclasses import dataclass
import torch
import typing
import copy
import logging
from tampc.dynamics import online_model
from arm_pytorch_utilities import tensor_utils, optim, serialization
from tampc.filters.ukf import EnvConditionedUKF

logger = logging.getLogger(__name__)


@dataclass
class ContactParameters:
    state_to_pos: typing.Callable[[torch.tensor], torch.tensor]
    pos_to_state: typing.Callable[[torch.tensor], torch.tensor]
    control_similarity: typing.Callable[[torch.tensor, torch.tensor], float]
    state_to_reaction: typing.Callable[[torch.tensor], torch.tensor]
    max_pos_move_per_action: float
    length: float = 0.1
    weight_multiplier: float = 0.1
    ignore_below_weight: float = 0.2
    force_threshold: float = 0.5


class ContactObject(serialization.Serializable):
    def __init__(self, empty_local_model: typing.Optional[online_model.OnlineDynamicsModel], params: ContactParameters,
                 assume_linear_scaling=True):
        # (x, u, dx) tuples associated with this object for fitting local model
        self.transitions = []
        # points on this object that are tracked
        self.points = None
        self.actions = None

        self.dynamics = empty_local_model
        self.state_to_pos = params.state_to_pos
        self.pos_to_state = params.pos_to_state

        self.assume_linear_scaling = assume_linear_scaling

        # UKF stuff
        self.length_parameter = params.length
        self.u_similarity = params.control_similarity
        self.n_x = 2  # state is center point
        self.n_y = 2  # observe robot position
        self.n_u = 2
        # process and observation noise, respectively
        self.device = optim.get_device()
        self.sigma = 0.0001
        Q = torch.eye(self.n_x, device=self.device).view(1, self.n_x, self.n_x) * self.sigma
        R = torch.eye(self.n_y, device=self.device).view(1, self.n_y, self.n_y) * self.sigma
        # kwargs for sigma point selector
        self.ukf = EnvConditionedUKF(self.n_x, self.n_y, self.n_u, Q, R, self.device, kappa=0, alpha=0.3)
        # posterior gaussian of position (init to prior)
        self.mu = None
        self.cov = torch.eye(self.n_x, device=self.device).repeat(1, 1, 1) * self.sigma * 2
        # prior gaussian of position
        self.mu_bar = None
        self.cov_bar = None
        self.weight_multiplier = params.weight_multiplier / (self.sigma * self.n_x)

    @property
    def center_point(self):
        return self.mu[0]

    @property
    def weight(self):
        tr = self.cov[0].trace()
        # convert to (0, 1)
        weight_exp = torch.exp(- self.weight_multiplier * tr)
        # weight = self.n_x * self.sigma / tr.item()
        return weight_exp

    def state_dict(self) -> dict:
        state = {'points': self.points,
                 'actions': self.actions,
                 'dynamics': self.dynamics.state_dict() if self.dynamics is not None else None,
                 'mu': self.mu,
                 'cov': self.cov,
                 'mu_bar': self.mu_bar,
                 'cov_bar': self.cov_bar
                 }
        return state

    def load_state_dict(self, state: dict) -> bool:
        self.points = state['points']
        self.actions = state['actions']

        self.mu = state['mu']
        self.cov = state['cov']
        self.mu_bar = state['mu_bar']
        self.cov_bar = state['cov_bar']
        if self.dynamics is not None and not self.dynamics.load_state_dict(state['dynamics']):
            return False
        return True

    def add_transition(self, x, u, dx):
        dtype = torch.float64 if not torch.is_tensor(x) else x.dtype
        x, u, dx = tensor_utils.ensure_tensor(optim.get_device(), dtype, x, u, dx)
        # save only positions
        x = self.state_to_pos(x)
        if self.points is None:
            self.points = x.view(1, -1)
            self.actions = u.view(1, -1)
            # first time init prior
            self.mu = x.clone().view(1, -1)
        else:
            self.points = torch.cat((self.points, x.view(1, -1)))
            self.actions = torch.cat((self.actions, u.view(1, -1)))

        centered_x = x - self.center_point

        if self.assume_linear_scaling:
            u_scale = u.norm()
            u /= u_scale
            dx /= u_scale
        # convert back to state
        centered_x = self.pos_to_state(centered_x).view(-1)
        if self.dynamics is not None:
            self.dynamics.update(centered_x, u, centered_x + dx)

    def move_all_points(self, dpos):
        self.points += dpos
        # TODO may need to change the training input points for dynamics to be self.points - self.mu

    def merge_objects(self, other_objects):
        # self.points = torch.cat([self.points] + [obj.points for obj in other_objects])
        # self.actions = torch.cat([self.actions] + [obj.actions for obj in other_objects])
        # self.center_point = self.points.mean(dim=0)
        raise NotImplementedError("Currently merging the estimates of the two objects is not handled")

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

    def clusters_to_object(self, x, u, length_parameter, u_similarity, candidates=None, act=None, pts=None,
                           x_is_pos=False):
        total_num = x.shape[0]
        if x_is_pos:
            pos = x
        else:
            pos = self.state_to_pos(x)

        if candidates is None:
            candidates = torch.ones(total_num, dtype=torch.bool, device=x.device)
        if act is None:
            act = self.actions.repeat(total_num, 1, 1).transpose(0, -2)
        if pts is None:
            pts = self.points.repeat(total_num, 1, 1).transpose(0, -2)

        u_sim = u_similarity(u[candidates], act[:, candidates]) + 1e-8  # avoid divide by 0
        dd = (pos[candidates] - pts[:, candidates]).norm(dim=-1) / u_sim
        accepted = torch.any(dd < length_parameter, dim=0)
        candidates[candidates] = accepted
        if not torch.any(accepted):
            return candidates, True
        return candidates, False

    def measurement_fn(self, state, environment):
        """Measurement that would be seen given state (object position) and [robot position, action] in environment

        Measurement needs to be continuous, and ideally in R^n to facilitate innovation calculations.
        Should only be used in update calls to the object that is in contact."""
        measurement = state
        return measurement

    def measurement_fn_reaction_force(self, state, environment):
        # version of measurement function where measurement space is unit reaction force
        # NOTE this is an alternative measurement function that more directly uses directionality
        # this is the robot position before movement
        robot_position, action, _ = environment

        # represent direction as unit vector
        measurement = torch.zeros((state.shape[0], self.n_y), device=state.device, dtype=state.dtype)

        # if obj pos not in the direction of the push
        # vector from robot to object center
        rob_to_obj = state - robot_position
        rob_to_obj_norm = rob_to_obj.norm(dim=1)
        # anything too far is disqualified (based on maximum distance we expect robot to move in 1 action)
        too_far = rob_to_obj_norm > 0.05
        # anything too angled is disqualified
        # TODO generalize function for getting predicted ee movement direction from actions
        unit_pos_diff = (rob_to_obj / rob_to_obj_norm.view(-1, 1))
        unit_action_dir = (action / action.norm())
        # dot product between position diff and action
        dot_pos_diff_action = unit_pos_diff @ unit_action_dir
        too_angled = dot_pos_diff_action < 0.5

        # those that are not too far or too angled should feel some force
        # uncertainty over contact surface, assume both are spheres so average between them
        # in_contact = torch.ones(state.shape[0]).to(dtype=torch.bool)
        in_contact = ~(too_far | too_angled)
        measurement[in_contact, :2] = -(unit_action_dir + unit_pos_diff[in_contact]) / 2
        measurement[in_contact, :2] /= measurement[in_contact, :2].norm(dim=1).view(-1, 1)
        return measurement

    def dynamics_fn(self, state, action, environment):
        """Predict change in object center position"""
        robot_position, _, dx = environment
        return state + self.state_to_pos(dx)

    def dynamics_fn_cluster_per_sample(self, state, action, environment):
        # version of dynamics function where instead of assuming all input is clustered/expected movement, we cluster
        # each input sample
        robot_position, _, dx = environment
        new_state = state.clone()
        # first check if we cluster into it
        # only relative position matters, so instead of translating the contact points, we translate the robot position
        # assume first element is mean, so we get relative translation
        dpos_input = state - state[0]
        # to get the same effect, we translate the robot in the opposite direction and magnitude
        robot_translated_pos = robot_position - dpos_input
        in_contact, none = self.clusters_to_object(robot_translated_pos, action, self.length_parameter,
                                                   self.u_similarity, x_is_pos=True)
        if none:
            return new_state

        x = robot_translated_pos[in_contact] - state[0]
        dpos = self.predict_dpos(x, action[in_contact])
        new_state[in_contact] += dpos
        return new_state

    def dynamics_no_movement_fn(self, state, action, environment):
        return state

    def ukf_update(self, measurement, environment):
        """Update with a force measurement; only call on the object in contact"""
        center_point_before = self.mu.clone()
        self.mu, self.cov = self.ukf.update(measurement, self.mu_bar, self.cov_bar, self.measurement_fn,
                                            environment=environment)
        self.move_all_points(self.mu - center_point_before)

    def ukf_update_no_contact(self):
        self.mu, self.cov = self.mu_bar, self.cov_bar

    def ukf_predict(self, control, environment, expect_movement=True):
        self.mu_bar, self.cov_bar, _ = self.ukf.predict(control, self.mu, self.cov,
                                                        self.dynamics_fn if expect_movement else self.dynamics_no_movement_fn,
                                                        environment=environment)


class ContactSet(serialization.Serializable):
    def __init__(self, params: ContactParameters, immovable_collision_checker=None,
                 contact_object_factory: typing.Callable[[], ContactObject] = None):
        self._obj: typing.List[ContactObject] = []
        # cached center of all points
        self.center_points = None

        self.p = params

        self.immovable_collision_checker = immovable_collision_checker
        # used to produce contact objects during loading
        self.contact_object_factory = contact_object_factory
        # process and measurement model uses this to decide when force is high or low magnitude

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

    def _keep_high_weight_contacts(self):
        if self._obj:
            init_len = len(self._obj)
            self._obj = [c for c in self._obj if c.weight > self.p.ignore_below_weight]
            if len(self._obj) != init_len:
                if len(self._obj):
                    self.center_points = torch.stack([obj.center_point for obj in self._obj])
                else:
                    self.center_points = None

    def updated(self):
        if self._obj:
            self._keep_high_weight_contacts()
        if self._obj:
            self.center_points = torch.stack([obj.center_point for obj in self._obj])

    def stepped_without_contact(self, u, environment):
        for ci in self._obj:
            ci.ukf_predict(u, environment, expect_movement=False)
            ci.ukf_update_no_contact()
        self._keep_high_weight_contacts()

    def merge_objects(self, obj_indices) -> ContactObject:
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
        d = (center_points - self.p.state_to_pos(goal_x)).norm(dim=-1)
        # modify by weight of contact object existing
        weights = torch.tensor([c.weight for c in self._obj], dtype=d.dtype, device=d.device)
        return (1 / d * weights.view(-1, 1)).sum(dim=0)

    def check_which_object_applies(self, x, u) -> typing.Tuple[typing.List[ContactObject], typing.List[int]]:
        res_c = []
        res_i = []
        if not self._obj:
            return res_c, res_i

        d = (self.center_points - self.p.state_to_pos(x)).norm(dim=1).view(-1)

        for i, cc in enumerate(self._obj):
            # first use heuristic to filter out points based on state distance to object centers
            if d[i] > 2 * self.p.length:
                continue
            # we're using the x before contact because our estimate of the object points haven't moved yet
            # TODO handle when multiple contact objects claim it is part of them
            clustered, _ = cc.clusters_to_object(x.view(1, -1), u.view(1, -1), self.p.length,
                                                 self.p.control_similarity)
            if clustered[0]:
                res_c.append(cc)
                res_i.append(i)
                # still can't deal with merging very well, so revert back to returning a single object
                break

        return res_c, res_i

    def update(self, x, u, dx, reaction):
        environment = [self.p.state_to_pos(x), u, dx]
        if reaction.norm() < self.p.force_threshold:
            self.stepped_without_contact(u, environment)
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
        # update c with observation (only this one; other UFKs didn't receive an observation)
        unit_reaction = reaction / reaction.norm()
        # also do prediction
        for ci in self._obj:
            if ci is not c:
                ci.ukf_predict(u, environment, expect_movement=False)
                ci.ukf_update_no_contact()
            else:
                ci.ukf_predict(u, environment, expect_movement=True)
                # c.ukf_update(unit_reaction, environment)
                # where the contact point center would be taking last point into account
                c.ukf_update(c.points.mean(dim=0) + self.p.state_to_pos(dx), environment)

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
            pos = self.p.state_to_pos(x)
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
                candidates = without_contact & (dd < 2 * self.p.length)
                if not torch.any(candidates):
                    continue

                # u_sim[stored action index (# points of object), total_num], sim of stored action and sampled action
                candidates, none = c.clusters_to_object(pos, u, self.p.length, self.p.control_similarity,
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
