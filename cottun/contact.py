from dataclasses import dataclass
import torch
import typing
import copy
import logging
import abc
import enum
import math
from arm_pytorch_utilities import tensor_utils, optim, serialization, linalg, math_utils, draw
from torch.distributions.multivariate_normal import MultivariateNormal
from cottun.filters.ukf import EnvConditionedUKF

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


class MeasurementType(enum.Enum):
    POSITION = 0
    REACTION = 1


def approx_conic_similarity(a_norm, a_origin, b_norm, b_origin):
    """Comparison elementwise of 2 lists of cones offset to some origin with a vector representing cone direction"""
    dir_sim = torch.cosine_similarity(a_norm, b_norm, dim=-1).clamp(0, 1) + 1e-8  # avoid divide by 0
    dd = (a_origin - b_origin).norm(dim=-1) / dir_sim
    return dd


class ContactObject(serialization.Serializable):
    def __init__(self, empty_local_model: typing.Optional[typing.Any], params: ContactParameters,
                 assume_linear_scaling=True):
        # (x, u, dx) tuples associated with this object for fitting local model
        self.transitions = []
        # points on this object that are tracked
        self.points = None
        self.actions = None

        # every input we get is dx = t (param_c * u) + (1 - t) (freespace dynamics(u))
        # however, since freespace dynamics(u) != 0, any dx ~= 0 we receive forces param_c = 0
        self.known_zero_dynamics = False
        self.dynamics = empty_local_model
        self.state_to_pos = params.state_to_pos
        self.pos_to_state = params.pos_to_state

        self.assume_linear_scaling = assume_linear_scaling

        self.p = params
        self.length_parameter = params.length
        self.u_similarity = params.control_similarity
        self.n_x = 2  # state is center point
        self.n_y = 2  # observe robot position
        self.n_u = 2

        # TODO move this in to params
        self.cluster_close_to_ratio = 0.5

    @property
    def measurement_type(self) -> MeasurementType:
        """What data does our measurement function expect"""
        return MeasurementType.POSITION

    @property
    @abc.abstractmethod
    def center_point(self):
        """What we are tracking"""

    @property
    @abc.abstractmethod
    def weight(self):
        """How important or certain we are about this contact object"""

    def state_dict(self) -> dict:
        state = {'points': self.points,
                 'actions': self.actions,
                 'dynamics': self.dynamics.state_dict() if self.dynamics is not None else None,
                 }
        return state

    def load_state_dict(self, state: dict) -> bool:
        self.points = state['points']
        self.actions = state['actions']

        if self.dynamics is not None and not self.dynamics.load_state_dict(state['dynamics']):
            return False
        return True

    def add_transition(self, x, u, dx):
        dtype = torch.float64 if not torch.is_tensor(x) else x.dtype
        x, u, dx = tensor_utils.ensure_tensor(optim.get_device(), dtype, x, u, dx)
        # save only positions
        x = self.state_to_pos(x)
        if self.points is None:
            self.points = x.clone().view(1, -1)
            self.actions = u.clone().view(1, -1)
        else:
            self.points = torch.cat((self.points, x.view(1, -1)))
            self.actions = torch.cat((self.actions, u.view(1, -1)))

        if dx.norm() < 1e-7:
            self.known_zero_dynamics = True

        if self.known_zero_dynamics:
            # move the robot's prev location to its current location
            self.points[-1] += self.state_to_pos(dx).view(-1)

        if self.dynamics is not None:
            centered_x = x - self.center_point

            if self.assume_linear_scaling:
                u_scale = u.norm()
                u = u / u_scale
                dx = dx / u_scale
            # convert back to state
            centered_x = self.pos_to_state(centered_x).view(-1)
            self.dynamics.update(centered_x, u, centered_x + dx)

    def move_all_points(self, dpos):
        if not self.known_zero_dynamics:
            self.points += dpos
        # TODO may need to change the training input points for dynamics to be self.points - self.mu

    def merge_objects(self, other_objects):
        self.points = torch.cat([self.points] + [obj.points for obj in other_objects])
        self.actions = torch.cat([self.actions] + [obj.actions for obj in other_objects])

    def merging_preserves_convexity(self, other_obj):
        combined_u = torch.cat([self.actions, other_obj.actions])
        combined_p = torch.cat([self.points, other_obj.points])
        total_num = len(combined_u)
        # get pairwise distance
        dd = approx_conic_similarity(combined_u, combined_p, combined_u.repeat(total_num, 1, 1).transpose(0, -2),
                                     combined_p.repeat(total_num, 1, 1).transpose(0, -2))

        at_least_this_many_close = math.ceil(self.cluster_close_to_ratio * total_num)
        partitioned_dist = torch.kthvalue(dd, k=at_least_this_many_close, dim=0)[0]
        accepted = partitioned_dist < self.p.length
        return torch.all(accepted)

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

        dd = approx_conic_similarity(u[candidates], pos[candidates], act[:, candidates], pts[:, candidates])

        at_least_this_many_close = math.ceil(self.cluster_close_to_ratio * total_num)
        partitioned_dist = torch.kthvalue(dd, k=at_least_this_many_close, dim=0)[0]
        accepted = partitioned_dist < length_parameter
        # accepted = torch.any(dd < length_parameter, dim=0)

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
        robot_position, action = environment['robot'], environment['control']

        # represent direction as unit vector
        measurement = torch.zeros((state.shape[0], self.n_y), device=state.device, dtype=state.dtype)

        # if obj pos not in the direction of the push
        # vector from robot to object center
        rob_to_obj = state - robot_position
        rob_to_obj_norm = rob_to_obj.norm(dim=1)
        # anything too far is disqualified (based on maximum distance we expect robot to move in 1 action)
        # and also expected object size
        # TODO use parameters for these; these should be learned on the training set
        expected_max_move = 0.03
        expected_max_obj_radius = 0.1
        too_far = rob_to_obj_norm > (expected_max_move + expected_max_obj_radius)
        # anything too angled is disqualified
        # TODO generalize function for getting predicted ee movement direction from actions
        unit_pos_diff = (rob_to_obj / rob_to_obj_norm.view(-1, 1))
        unit_action_dir = (action / action.norm())
        # dot product between position diff and action
        dot_pos_diff_action = unit_pos_diff @ unit_action_dir
        too_angled = dot_pos_diff_action < 0.3

        # those that are not too far or too angled should feel some force
        # uncertainty over contact surface, assume both are spheres so average between them
        # in_contact = torch.ones(state.shape[0]).to(dtype=torch.bool)
        in_contact = ~(too_far | too_angled)
        measurement[in_contact, :2] = -(unit_action_dir + unit_pos_diff[in_contact]) / 2
        measurement[in_contact, :2] /= measurement[in_contact, :2].norm(dim=1).view(-1, 1)
        return measurement

    def dynamics_fn(self, state, action, environment):
        """Predict change in object center position"""
        return state + self.state_to_pos(environment['dx'])

    def dynamics_fn_cluster_per_sample(self, state, action, environment):
        # version of dynamics function where instead of assuming all input is clustered/expected movement, we cluster
        # each input sample
        robot_position = environment['robot']
        dx = environment['dx']
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

        new_state[in_contact] += dx
        return new_state

    def dynamics_no_movement_fn(self, state, action, environment):
        return state

    @abc.abstractmethod
    def filter_update(self, measurement, environment, observed_movement=True):
        """Update step of the Bayes filter"""

    @abc.abstractmethod
    def filter_predict(self, control, environment, expect_movement=True):
        """Predict step of teh Bayes filter"""


class ContactUKF(ContactObject):
    def __init__(self, empty_local_model: typing.Optional[typing.Any], params: ContactParameters, **kwargs):
        super(ContactUKF, self).__init__(empty_local_model, params, **kwargs)
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
        state = super(ContactUKF, self).state_dict()
        state.update({
            'mu': self.mu,
            'cov': self.cov,
            'mu_bar': self.mu_bar,
            'cov_bar': self.cov_bar
        })
        return state

    def load_state_dict(self, state: dict) -> bool:
        if not super(ContactUKF, self).load_state_dict(state):
            return False

        self.mu = state['mu']
        self.cov = state['cov']
        self.mu_bar = state['mu_bar']
        self.cov_bar = state['cov_bar']
        return True

    def merge_objects(self, other_objects):
        super(ContactUKF, self).merge_objects(other_objects)
        # self.center_point = self.points.mean(dim=0)
        weight = [len(self.points)] + [len(obj.points) for obj in other_objects]
        weight = torch.tensor(weight, dtype=self.points.dtype, device=self.device)
        weight = weight / weight.sum()
        mu = torch.cat([self.mu] + [obj.mu for obj in other_objects])
        self.mu = (mu * weight.view(-1, 1)).sum(dim=0, keepdim=True)
        cov = torch.cat([self.cov] + [obj.cov for obj in other_objects])
        self.cov = (cov * weight.view(-1, 1, 1)).sum(dim=0, keepdim=True)
        self.mu_bar = None
        self.cov_bar = None

    def filter_update(self, measurement, environment, observed_movement=True):
        """Update with a force measurement; only call on the object in contact"""
        if observed_movement:
            # center_point_before = self.mu.clone()
            # self.mu from before hasn't incorporated the current point
            center_point_before = self.points.mean(dim=0)
            self.mu, self.cov = self.ukf.update(measurement, self.mu_bar, self.cov_bar, self.measurement_fn,
                                                environment=environment)
            # these 2 are pretty much equivalent now
            # self.move_all_points(self.mu - center_point_before)
            self.move_all_points(self.state_to_pos(environment['dx']))
        else:
            self.mu, self.cov = self.mu_bar, self.cov_bar

    def filter_predict(self, control, environment, expect_movement=True):
        self.mu_bar, self.cov_bar, _ = self.ukf.predict(control, self.mu, self.cov,
                                                        self.dynamics_fn if expect_movement else self.dynamics_no_movement_fn,
                                                        environment=environment)

    def add_transition(self, x, u, dx):
        dtype = torch.float64 if not torch.is_tensor(x) else x.dtype
        x = tensor_utils.ensure_tensor(self.device, dtype, x)
        x = self.state_to_pos(x)
        if self.mu is None:
            self.mu = x.clone().view(1, -1)
        return super(ContactUKF, self).add_transition(x, u, dx)


import matplotlib.pyplot as plt


class ContactPF(ContactObject):
    def __init__(self, empty_local_model: typing.Optional[typing.Any], params: ContactParameters, n_particles=500,
                 n_eff_threshold=1.,
                 **kwargs):
        super(ContactPF, self).__init__(empty_local_model, params, **kwargs)
        self.device = optim.get_device()

        self.n_particles = n_particles
        self.particles = None
        self.hypothesis = None
        # can only init when given an observation
        # sigma (assume symmetric) for initial gaussian prior of particles
        self.init_spread = 0.005
        self.dynamics_noise_spread = 0.0001
        # this is easier to tune since measurement space is bounded, so error is also bounded
        self.squared_error_sigma = 0.3

        self.n_eff_threshold = n_eff_threshold
        self.weights = torch.ones(self.n_particles, device=self.device) / self.n_particles
        self.weights_normalization = 1
        # additional noise for each action step
        self.dynamics_noise_dist = MultivariateNormal(torch.zeros(self.n_x, device=self.device),
                                                      torch.eye(self.n_x,
                                                                device=self.device) * self.dynamics_noise_spread)

        self.weight_multiplier = 2 * params.weight_multiplier / (self.dynamics_noise_spread * self.n_x)
        # don't care about weight for now

        self.plot = False
        if self.plot:
            plt.ion()
            f, self.axes = plt.subplots(2, 2, figsize=(8, 8), constrained_layout=True)
            self.axes[0, 0].set_title('predict input: state space')
            self.axes[0, 0].set_ylim(-0.5, 0.5)
            self.axes[0, 0].set_xlim(-0.5, 0.5)
            self.axes[0, 1].set_title('predict output: state space')
            self.axes[0, 1].set_ylim(-0.5, 0.5)
            self.axes[0, 1].set_xlim(-0.5, 0.5)

            self.axes[1, 0].set_ylim(-1., 1.)
            self.axes[1, 0].set_xlim(-1., 1.)
            self.axes[1, 0].set_title('update input: measurement space')
            self.axes[1, 1].set_ylim(-0.5, 0.5)
            self.axes[1, 1].set_xlim(-0.5, 0.5)
            self.axes[1, 1].set_title('update output: state space')

    @property
    def measurement_type(self):
        return MeasurementType.REACTION

    def init_pf(self, environment):
        robot_position = environment['robot']
        action = environment['control']
        # TODO make expected max object radius a parameter
        action_dir = action / action.norm()
        sample_dist = MultivariateNormal(robot_position.view(-1) + action_dir * 0.1,
                                         covariance_matrix=torch.eye(self.n_x, device=self.device) * self.init_spread)
        new_sample = sample_dist.sample((self.n_particles,))
        self.particles = new_sample

    @property
    def center_point(self):
        # take the weighted mean
        return torch.sum(self.particles * self.weights.view(-1, 1), dim=0)

    @property
    def weight(self):
        # TODO how to assign weight to particles? Currently using empirical covariance
        cov = linalg.cov(self.particles)
        tr = cov.trace()
        # convert to (0, 1)
        weight_exp = torch.exp(- self.weight_multiplier * tr)
        # weight = self.n_x * self.sigma / tr.item()
        return weight_exp

    def state_dict(self) -> dict:
        # TODO
        pass

    def load_state_dict(self, state: dict) -> bool:
        # TODO
        pass

    def filter_update(self, measurement, environment, observed_movement=True):
        if observed_movement:
            center_point_before = self.center_point.clone()
            self._pf_update(measurement, environment)
            # can't just move points like before since our initial estimate of robot location will be bad
            # self.move_all_points(self.center_point - center_point_before)
            self.move_all_points(self.state_to_pos(environment['dx']))
        else:
            pass

    def _pf_update(self, measurement, environment):
        # hypothesis observations
        self.hypothesis = self.measurement_fn_reaction_force(self.particles, environment)

        # compare hypothesized measurement with actual measurement
        obs_weights = squared_error(self.hypothesis, measurement, self.squared_error_sigma)
        # obs_weights = torch.zeros(self.n_particles, device=self.device)
        valid = self.hypothesis.norm(dim=1) > 0
        obs_weights[~valid] = 0
        # obs_weights[valid] = torch.cosine_similarity(self.hypothesis[valid], measurement.view(1, -1))
        # obs_weights[valid] = torch.exp(-obs_weights[valid] / (2.0 * 0.5 ** 2))

        if self.plot:
            ax = self.axes[1, 0]
            draw.clear_ax_content(ax)
            c = [(0, 0, 1, min(1, max(0.01, w))) for w in obs_weights]
            ax.scatter(self.hypothesis[:, 0], self.hypothesis[:, 1], label='hypotheses', c=c)
            ax.scatter(measurement[0], measurement[1], label='measured', c='g')
            ax.legend()

        self.weights = self.weights * obs_weights
        # force positive
        self.weights[self.weights < 0] = 0

        # TODO consider internal weights to combine inverse and forward models

        # normalize weights to resampling probabilities
        self.weights_normalization = self.weights.sum()
        self.weights = self.weights / self.weights_normalization

        # from pfilter
        # Compute effective sample size and entropy of weighting vector.
        # These are useful statistics for adaptive particle filtering.
        self.n_eff = (1.0 / (self.weights ** 2).sum()) / self.n_particles
        self.weight_entropy = torch.sum(self.weights * torch.log(self.weights))

        # preserve current sample set before any replenishment
        self.original_particles = self.particles.clone()

        # store MAP estimate
        argmax_weight = torch.argmax(self.weights)
        self.map_state = self.particles[argmax_weight]
        self.map_hypothesis = self.hypothesis[argmax_weight]

        # resampling (systematic resampling) step
        if self.n_eff < self.n_eff_threshold:
            indices = resample(self.weights)
            self.particles = self.particles[indices, :]
            self.weights = torch.ones(self.n_particles, device=self.device) / self.n_particles

        if self.plot:
            ax = self.axes[1, 1]
            draw.clear_ax_content(ax)
            c = [(0, 0, 1, w * self.n_particles / 2) for w in self.weights]
            ax.scatter(self.particles[:, 0], self.particles[:, 1], label='particles', c=c)
            robot_xy = environment['robot']
            ax.scatter(robot_xy[:, 0], robot_xy[:, 1], label='robot', c='g')
            m = self.center_point
            ax.scatter(m[0], m[1], label='mean particle', c='c')
            ax.scatter(self.map_state[0], self.map_state[1], label='MAP particle', c='y')

            colors = 'rkm'
            obj_poses = environment.get('obj', None)
            if obj_poses is not None:
                for obj_id, obj_pose in obj_poses.items():
                    ax.scatter(obj_pose[0], obj_pose[1], label=f"object {obj_id}", c=colors[obj_id % len(colors)])

            ax.legend()

    def filter_predict(self, control, environment, expect_movement=True):
        if self.particles is None:
            self.init_pf(environment)

        if self.plot:
            ax = self.axes[0, 0]
            draw.clear_ax_content(ax)
            c = [(0, 0, 1, w * self.n_particles / 2) for w in self.weights]
            ax.scatter(self.particles[:, 0], self.particles[:, 1], label='particles', c=c)
            robot_xy = environment['robot']
            ax.scatter(robot_xy[:, 0], robot_xy[:, 1], label='robot', c='g')
            m = self.center_point
            ax.scatter(m[0], m[1], label='mean particle', c='c')
            # ax.scatter(sp[1:, 0], sp[1:, 1], label='sp', alpha=0.3)
            ax.legend()

        if expect_movement:
            self.particles = self.dynamics_fn(self.particles, control, environment)
        # apply dynamics to particles
        else:
            # do nothing
            pass
        noise = self.dynamics_noise_dist.sample((self.n_particles,))
        self.particles += noise

        if self.plot:
            ax = self.axes[0, 1]
            draw.clear_ax_content(ax)
            c = [(0, 0, 1, w * self.n_particles / 2) for w in self.weights]
            ax.scatter(self.particles[:, 0], self.particles[:, 1], label='particles', c=c)
            robot_xy = environment['robot']
            ax.scatter(robot_xy[:, 0], robot_xy[:, 1], label='robot', c='g')
            m = self.center_point
            ax.scatter(m[0], m[1], label='mean particle', c='c')
            # ax.scatter(sp[1:, 0], sp[1:, 1], label='sp', alpha=0.3)
            ax.legend()


# resample function from http://scipy-cookbook.readthedocs.io/items/ParticleFilter.html
def resample(weights):
    n = len(weights)
    indices = []
    C = [0.0] + [torch.sum(weights[: i + 1]) for i in range(n)]
    u0, j = torch.rand((1,)), 0
    for u in [(u0 + i) / n for i in range(n)]:
        while j < len(C) and u > C[j]:
            j += 1
        indices.append(j - 1)
    return indices


def squared_error(x, y, sigma=1.):
    """
        from pfilter
        RBF kernel, supporting masked values in the observation
        Parameters:
        -----------
        x : array (N,D) array of values
        y : array (N,D) array of values
        Returns:
        -------
        distance : scalar
            Total similarity, using equation:
                d(x,y) = e^((-1 * (x - y) ** 2) / (2 * sigma ** 2))
            summed over all samples. Supports masked arrays.
    """
    dx = (x - y) ** 2
    d = torch.sum(dx, dim=1)
    return torch.exp(-d / (2.0 * sigma ** 2))


class ContactSet(serialization.Serializable):
    def __init__(self, params: ContactParameters, immovable_collision_checker=None):
        self.p = params
        self.immovable_collision_checker = immovable_collision_checker

    @abc.abstractmethod
    def update(self, x, u, dx, reaction, info=None):
        """Update contact set with observed transition"""

    @abc.abstractmethod
    def get_batch_data_for_dynamics(self, total_num):
        """Return initial contact data needed for dynamics for total_num batch size"""

    @abc.abstractmethod
    def dynamics(self, x, u, contact_data):
        """Perform batch dynamics on x with control u using contact data from get_batch_data_for_dynamics

        :return next x, without contact mask, updated contact data"""


class ContactSetHard(ContactSet):
    def __init__(self, *args, contact_object_factory: typing.Callable[[], ContactObject] = None, **kwargs):
        super(ContactSetHard, self).__init__(*args, **kwargs)
        self._obj: typing.List[ContactObject] = []
        # cached center of all points
        self.center_points = None

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
        # TODO for evaluating clustering and filtering, don't have to forget low weight objects
        return
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
            ci.filter_predict(u, environment, expect_movement=False)
            ci.filter_update(None, environment, observed_movement=False)
        self._keep_high_weight_contacts()

    def merge_objects(self, obj_indices) -> ContactObject:
        obj_to_combine = [self._obj[i] for i in obj_indices]
        self._obj = [obj for obj in self._obj if obj not in obj_to_combine]
        c = copy.deepcopy(obj_to_combine[0])
        separate_objects = [c]
        # check if they should be combined
        for cc in obj_to_combine[1:]:
            if c.merging_preserves_convexity(cc):
                c.merge_objects([cc])
            else:
                separate_objects.append(cc)
        for cc in separate_objects:
            self.append(cc)
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
            clustered, _ = cc.clusters_to_object(x.view(1, -1), u.view(1, -1), self.p.length,
                                                 self.p.control_similarity)
            if clustered[0]:
                res_c.append(cc)
                res_i.append(i)

        return res_c, res_i

    def update(self, x, u, dx, reaction, info=None):
        """Returns updated contact object"""
        environment = {'robot': self.p.state_to_pos(x), 'control': u, 'dx': dx}
        # debugging info
        if info is not None:
            environment['obj'] = info['object_poses']
        if reaction.norm() < self.p.force_threshold:
            self.stepped_without_contact(u, environment)
            return None, None

        # associate each contact to a single object (max likelihood estimate on which object it is)
        cc, ii = self.check_which_object_applies(x, u)
        # couldn't find an existing contact
        if not len(cc):
            # if using object-centered model, don't use preprocessor, else use default
            c = self.contact_object_factory()
            self.append(c)
        else:
            c = cc[0]
        c.add_transition(x, u, dx)
        # matches more than 1 contact set, combine them
        if len(cc) > 1:
            c = self.merge_objects(ii)
        # update c with observation (only this one; other UFKs didn't receive an observation)
        unit_reaction = reaction / reaction.norm()
        # also do prediction
        for ci in self._obj:
            if ci is not c:
                ci.filter_predict(u, environment, expect_movement=False)
                ci.filter_update(None, environment, observed_movement=False)
            else:
                ci.filter_predict(u, environment, expect_movement=True)
                # where the contact point center would be taking last point into account
                if c.measurement_type == MeasurementType.POSITION:
                    c.filter_update(c.points.mean(dim=0) + self.p.state_to_pos(dx), environment, observed_movement=True)
                elif c.measurement_type == MeasurementType.REACTION:
                    c.filter_update(unit_reaction, environment, observed_movement=True)
                else:
                    raise RuntimeError(f"Unknown measurement type {c.measurement_type}")

        self.updated()
        return c, cc

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


class ContactSetSoft(ContactSet):
    def __init__(self, *args, n_particles=100, **kwargs):
        super(ContactSetSoft, self).__init__(*args, **kwargs)
        self.adjacency = None
        self.connection_prob = None
        self.n_particles = n_particles

        self.pts = None
        self.acts = None

        self.sampled_pts = None

    def _compute_full_adjacency(self):
        # don't typically need to compute full adjacency
        total_num = self.pts.shape[0]
        dd = approx_conic_similarity(self.acts, self.pts,
                                     self.acts.repeat(total_num, 1, 1).transpose(0, -2),
                                     self.pts.repeat(total_num, 1, 1).transpose(0, -2))
        self.adjacency = self._distance_to_probability(dd)
        return self.adjacency

    def _distance_to_probability(self, distance):
        # parameter where higher means a greater drop off in probability with distance
        sigma = self.p.length
        return torch.exp(-sigma * distance)

    def predict_particles(self, dx):
        # assume we just added latest pt and act to self
        dd = approx_conic_similarity(self.acts[-1], self.pts[-1],
                                     self.acts.repeat(self.n_particles, 1, 1),
                                     self.sampled_pts)

        # convert to probability
        connection_prob = self._distance_to_probability(dd)

        # sample particles which make hard assignments
        # independently sample uniform [0, 1) and compare against prob - note that connections are symmetric
        # sampled_prob[i,j] is the ith particle's probability of connection between the latest point and the jth point
        sampled_prob = torch.rand(connection_prob.shape, device=self.pts.device)
        adjacent = sampled_prob < connection_prob

        # don't actually need to label connected components because we just need to propagate for the latest
        # apply dx to each particle's cluster that contains the latest x
        # sampled_pts = self.pts.repeat(self.n_particles, 1, 1)
        self.sampled_pts[adjacent] += self.p.state_to_pos(dx)

    def update(self, x, u, dx, reaction, info=None):
        environment = {'robot': self.p.state_to_pos(x), 'control': u, 'dx': dx}
        # debugging info
        if info is not None:
            environment['obj'] = info['object_poses']
        if reaction.norm() < self.p.force_threshold:
            # TODO step without contact
            return None, None

        x = self.p.state_to_pos(x)

        if self.pts is None:
            self.pts = x.view(1, -1)
            self.acts = u.view(1, -1)
            self.sampled_pts = self.pts.repeat(self.n_particles, 1, 1)
        else:
            self.pts = torch.cat((self.pts, x.view(1, -1)))
            self.acts = torch.cat((self.acts, u.view(1, -1)))
            self.sampled_pts = torch.cat([self.sampled_pts, x.view(1, -1).repeat(self.n_particles, 1, 1)], dim=1)

        self.predict_particles(dx)
        # TODO update step
        return None, None

    def dynamics(self, x, u, contact_data):
        raise NotImplementedError()

    def get_batch_data_for_dynamics(self, total_num):
        raise NotImplementedError()
