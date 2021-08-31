from tampc.util import UseTsf

try:
    import sys

    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except:
    pass

import time
import math
import random
import torch
import pybullet as p
import numpy as np
import matplotlib.pyplot as plt
import logging
import os
from datetime import datetime

from arm_pytorch_utilities import rand, load_data
from arm_pytorch_utilities.optim import get_device
from arm_pytorch_utilities import draw

from tampc import cfg
from cottun import tracking
from tampc.controller import controller
from tampc.env import arm
from tampc.env.arm import task_map, Levels
from tampc.env_getters.arm import RetrievalGetter
from tampc.env.pybullet_env import ContactInfo, state_action_color_pairs, closest_point_on_surface
from cottun import icp
from cottun import detection

from pytorch_kinematics import transforms as tf

ch = logging.StreamHandler()
fh = logging.FileHandler(os.path.join(cfg.ROOT_DIR, "logs", "{}.log".format(datetime.now())))

logging.basicConfig(level=logging.DEBUG,
                    format='[%(levelname)s %(asctime)s %(pathname)s:%(lineno)d] %(message)s',
                    datefmt='%m-%d %H:%M:%S', handlers=[ch, fh])

logging.getLogger('matplotlib.font_manager').disabled = True

logger = logging.getLogger(__name__)


def rot_2d_mat_to_angle(T):
    """T: bx3x3 homogenous transforms or bx2x2 rotation matrices"""
    return torch.atan2(T[:, 1, 0], T[:, 0, 0])


class RetrievalController(controller.Controller):

    def __init__(self, contact_detector: detection.ContactDetector, nu, dynamics, cost_to_go,
                 contact_set: tracking.ContactSetHard, u_min, u_max, num_samples=100,
                 walk_length=3):
        super().__init__()
        self.contact_detector = contact_detector
        self.nu = nu
        self.u_min = u_min
        self.u_max = u_max
        self.dynamics = dynamics
        self.cost = cost_to_go
        self.num_samples = num_samples

        self.max_walk_length = walk_length
        self.remaining_random_actions = 0

        self.x_history = []
        self.u_history = []

        self.contact_set = contact_set

    def command(self, obs, info=None):
        d = self.dynamics.device
        dtype = self.dynamics.dtype

        self.x_history.append(obs)

        if self.contact_detector.in_contact():
            self.remaining_random_actions = self.max_walk_length
            self.contact_set.update(self.x_history[-2], torch.tensor(self.u_history[-1]),
                                    self.x_history[-1] - self.x_history[-2],
                                    self.contact_detector, torch.tensor(info['reaction']), info=info)

        if self.remaining_random_actions > 0:
            u = np.random.uniform(low=self.u_min, high=self.u_max, size=self.nu)
            self.remaining_random_actions -= 1
        else:
            # take greedy action if not in contact
            state = torch.from_numpy(obs).to(device=d, dtype=dtype).repeat(self.num_samples, 1)
            u = np.random.uniform(low=self.u_min, high=self.u_max, size=(self.num_samples, self.nu))
            u = torch.from_numpy(u).to(device=d, dtype=dtype)

            next_state = self.dynamics(state, u)
            costs = self.cost(torch.from_numpy(self.goal).to(device=d, dtype=dtype), next_state)
            min_i = torch.argmin(costs)
            u = u[min_i].cpu().numpy()

        self.u_history.append(u)
        return u


class RetrievalPredeterminedController(controller.Controller):

    def __init__(self, contact_detector: detection.ContactDetector, contact_set: tracking.ContactSetHard, controls):
        super().__init__()
        self.contact_detector = contact_detector
        self.controls = controls
        self.i = 0

        self.x_history = []
        self.u_history = []

        self.contact_set = contact_set

    def command(self, obs, info=None):
        self.x_history.append(obs)

        if self.contact_detector.in_contact():
            self.contact_set.update(self.x_history[-2], torch.tensor(self.u_history[-1]),
                                    self.x_history[-1] - self.x_history[-2],
                                    self.contact_detector, torch.tensor(info['reaction']), info=info)

        if self.i < len(self.controls):
            u = self.controls[self.i]
            self.i += 1
        else:
            u = [0 for _ in range(len(self.controls[0]))]

        self.u_history.append(u)
        return u


# sample model points from object
def sample_model_points(object_id, num_points=100, reject_too_close=0.002, force_z=None, seed=0, name=""):
    fullname = os.path.join(cfg.DATA_DIR, f'model_points_cache.pkl')
    if os.path.exists(fullname):
        cache = torch.load(fullname)
        if name not in cache:
            cache[name] = {}
        if seed in cache[name]:
            return cache[name][seed]
    else:
        cache = {name: {}}

    with rand.SavedRNG():
        rand.seed(seed)
        orig_pos, orig_orientation = p.getBasePositionAndOrientation(object_id)
        z = orig_pos[2]
        # first reset to canonical location
        canonical_pos = [0, 0, z]
        p.resetBasePositionAndOrientation(object_id, canonical_pos, [0, 0, 0, 1])

        points = []
        sigma = 0.1
        while len(points) < num_points:
            tester_pos = np.r_[np.random.randn(2) * sigma, z]
            # sample an object at random points around this object and find closest point to it
            closest = closest_point_on_surface(object_id, tester_pos)
            pt = closest[ContactInfo.POS_A]
            if force_z is not None:
                pt = (pt[0], pt[1], force_z)
            if len(points) > 0:
                d = np.subtract(points, pt)
                d = np.linalg.norm(d, axis=1)
                if np.any(d < reject_too_close):
                    continue
            points.append(pt)

    p.resetBasePositionAndOrientation(object_id, orig_pos, orig_orientation)

    points = torch.tensor(points)

    cache[name][seed] = points
    torch.save(cache, fullname)

    return points


def test_icp(env):
    # test ICP using fixed set of points
    o = p.getBasePositionAndOrientation(env.target_object_id)[0]
    contact_points = np.stack([
        [o[0] - 0.045, o[1] - 0.05],
        [o[0] - 0.05, o[1] - 0.01],
        [o[0] - 0.045, o[1] + 0.02],
        [o[0] - 0.045, o[1] + 0.04],
        [o[0] - 0.01, o[1] + 0.05]
    ])
    actions = np.stack([
        [0.7, -0.7],
        [0.9, 0.2],
        [0.8, 0],
        [0.5, 0.6],
        [0, -0.8]
    ])
    contact_points = np.stack(contact_points)

    angle = 0.5
    dx = -0.4
    dy = 0.2
    c, s = math.cos(angle), math.sin(angle)
    rot = np.array([[c, -s],
                    [s, c]])
    contact_points = np.dot(contact_points, rot.T)
    contact_points[:, 0] += dx
    contact_points[:, 1] += dy
    actions = np.dot(actions, rot.T)

    state_c, action_c = state_action_color_pairs[0]
    env.visualize_state_actions("fixed", contact_points, actions, state_c, action_c, 0.05)

    model_points = sample_model_points(env.target_object_id, num_points=50, force_z=z, seed=0, name="cheezit")
    for i, pt in enumerate(model_points):
        env._dd.draw_point(f"mpt{i}", pt, color=(0, 0, 1), length=0.003)

    # perform ICP and visualize the transformed points
    # history, transformed_contact_points = icp.icp(model_points[:, :2], contact_points,
    #                                               point_pairs_threshold=len(contact_points), verbose=True)

    # better to have few A than few B and then invert the transform
    T, distances, i = icp.icp_2(contact_points, model_points[:, :2])
    # transformed_contact_points = np.dot(np.c_[contact_points, np.ones((contact_points.shape[0], 1))], T.T)
    # T, distances, i = icp.icp_2(model_points[:, :2], contact_points)
    transformed_model_points = np.dot(np.c_[model_points[:, :2], np.ones((model_points.shape[0], 1))],
                                      np.linalg.inv(T).T)
    for i, pt in enumerate(transformed_model_points):
        pt = [pt[0], pt[1], z]
        env._dd.draw_point(f"tmpt{i}", pt, color=(0, 1, 0), length=0.003)

    while True:
        env.step([0, 0])
        time.sleep(0.2)


force_gui = True
env = RetrievalGetter.env(level=Levels.SIMPLE_CLUTTER, mode=p.GUI if force_gui else p.DIRECT)
contact_params = RetrievalGetter.contact_parameters(env)


def cost_to_go(state, goal):
    return env.state_distance_two_arg(state, goal)


def create_contact_object():
    return tracking.ContactUKF(None, contact_params)


def object_robot_penetration_score(object_id, robot_id, object_transform):
    """Compute the penetration between object and robot for a given transform of the object"""
    o_pos, o_orientation = p.getBasePositionAndOrientation(object_id)
    yaw = torch.atan2(object_transform[1, 0], object_transform[0, 0])
    t = np.r_[object_transform[:2, 2], o_pos[2]]
    # temporarily move object with transform
    p.resetBasePositionAndOrientation(object_id, t, p.getQuaternionFromEuler([0, 0, yaw]))

    p.performCollisionDetection()
    closest = p.getClosestPoints(object_id, robot_id, 100)
    d = min(c[ContactInfo.DISTANCE] for c in closest)

    p.resetBasePositionAndOrientation(object_id, o_pos, o_orientation)
    return -d


using_soft_contact = True
using_predetermined_control = True

dtype = torch.float32
if using_soft_contact:
    contact_set = tracking.ContactSetSoft(arm.ArmPointToConfig(env), contact_params)
else:
    contact_set = tracking.ContactSetHard(contact_params, contact_object_factory=create_contact_object)

# TODO have this seed be selectable
rand.seed(0)
if using_predetermined_control:
    predetermined_control = {}

    ctrl = [[0.7, -1]] * 5
    ctrl += [[0.4, 0.4], [.5, -1]] * 6
    ctrl += [[-0.2, 1]] * 4
    ctrl += [[0.3, -0.3], [0.4, 1]] * 4
    ctrl += [[1., -1]] * 3
    ctrl += [[1., 0.6], [-0.7, 0.5]] * 4
    ctrl += [[0., 1]] * 5
    ctrl += [[1., 0]] * 4
    ctrl += [[0.4, -1.], [0.4, 0.5]] * 4
    noise = (np.random.rand(len(ctrl), 2) - 0.5) * 0.5
    ctrl = np.add(ctrl, noise)
    predetermined_control[Levels.SIMPLE_CLUTTER] = ctrl

    ctrl = RetrievalPredeterminedController(env.contact_detector, contact_set,
                                            predetermined_control[env.level])
else:
    ds, pm = RetrievalGetter.prior(env, use_tsf=UseTsf.NO_TRANSFORM)
    u_min, u_max = env.get_control_bounds()
    ctrl = RetrievalController(env.contact_detector, env.nu, pm.dyn_net, cost_to_go, contact_set, u_min, u_max,
                               walk_length=6)

obs = env.reset()
z = env._observe_ee(return_z=True)[-1]

model_points = sample_model_points(env.target_object_id, num_points=50, force_z=z, seed=0, name="cheezit")
mp = model_points[:, :2].cpu().numpy()
mph = model_points.clone().to(dtype=dtype)
mph[:, -1] = 1

ctrl.set_goal(env.goal[:2])
info = None
simTime = 0
best_tsf_guess = None
while True:
    best_distance = None
    simTime += 1
    env.draw_user_text("{}".format(simTime), xy=(0.5, 0.7, -1))

    action = ctrl.command(obs, info)
    env.visualize_contact_set(contact_set)
    if env.contact_detector.in_contact():
        pts = contact_set.get_posterior_points() if using_soft_contact else None
        to_iter = contact_set.get_hard_assignment(
            contact_set.p.hard_assignment_threshold) if using_soft_contact else contact_set
        dist_per_est_obj = []
        for c in to_iter:
            this_pts = pts[c] if using_soft_contact else c.points
            T, distances, _ = icp.icp_3(this_pts, model_points[:, :2], given_init_pose=best_tsf_guess, batch=30)
            T = T.inverse()
            penetration = [object_robot_penetration_score(env.target_object_id, env.robot_id, T[b]) for b in
                           range(T.shape[0])]
            score = np.abs(penetration)
            best_tsf_index = np.argmin(score)


            yaws = rot_2d_mat_to_angle(T)
            translations = T[:, :2, 2]
            best_tsf_distances = (translations.var(dim=0).sum()).item()
            # best_tsf_distances = distances[best_tsf_index].mean().item()

            dist_per_est_obj.append(best_tsf_distances)
            if best_distance is None or best_tsf_distances < best_distance:
                best_distance = best_tsf_distances
                best_tsf_guess = T[best_tsf_index].inverse()


            # for b in range(T.shape[0]):
            #     transformed_model_points = mph @ T[b].transpose(-1, -2)
            #     for i, pt in enumerate(transformed_model_points):
            #         if i % 2 == 0:
            #             pt = [pt[0], pt[1], z]
            #             env._dd.draw_point(f"tmpt{b}-{i}", pt, color=(0, 1, b / T.shape[0]), length=0.003)

        logger.info(f"err each obj {np.round(dist_per_est_obj, 4)}")
        transformed_model_points = mph @ best_tsf_guess.inverse().transpose(-1, -2)
        for i, pt in enumerate(transformed_model_points):
            if i % 2 == 0:
                continue
            pt = [pt[0], pt[1], z]
            env._dd.draw_point(f"tmptbest-{i}", pt, color=(0, 0, 1), length=0.008)

    if torch.is_tensor(action):
        action = action.cpu()

    action = np.array(action).flatten()
    obs, rew, done, info = env.step(action)
