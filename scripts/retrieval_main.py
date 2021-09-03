from cottun.retrieval_controller import RetrievalController, RetrievalPredeterminedController
from tampc.util import UseTsf

try:
    import sys

    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except:
    pass

import time
import math
import torch
import pybullet as p
import numpy as np
import logging
import os
from datetime import datetime

from arm_pytorch_utilities import rand

from tampc import cfg
from cottun import tracking
from tampc.env import arm
from tampc.env.arm import Levels
from tampc.env_getters.arm import RetrievalGetter
from tampc.env.pybullet_env import ContactInfo, state_action_color_pairs, closest_point_on_surface
from cottun import icp

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
    z = env._observe_ee(return_z=True)[-1]
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


from arm_pytorch_utilities.math_utils import angular_diff


def pose_error(target_pose, guess_pose):
    # mirrored, so being off by 180 degrees is fine
    yaw_error = min(abs(angular_diff(target_pose[-1], guess_pose[-1])),
                    abs(angular_diff(target_pose[-1] + np.pi, guess_pose[-1])))
    pos_error = np.linalg.norm(np.subtract(target_pose[:2], guess_pose[:2]))
    return pos_error, yaw_error


def run_retrieval(env, seed=0, using_soft_contact=True, using_predetermined_control=True):
    contact_params = RetrievalGetter.contact_parameters(env)

    def cost_to_go(state, goal):
        return env.state_distance_two_arg(state, goal)

    def create_contact_object():
        return tracking.ContactUKF(None, contact_params)

    dtype = torch.float32

    if using_soft_contact:
        contact_set = tracking.ContactSetSoft(arm.ArmPointToConfig(env), contact_params)
    else:
        contact_set = tracking.ContactSetHard(contact_params, contact_object_factory=create_contact_object)

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

        rand.seed(seed)
        for k, v in predetermined_control.items():
            predetermined_control[k] = np.add(v, (np.random.rand(len(v), 2) - 0.5) * 0.02)

        ctrl = RetrievalPredeterminedController(env.contact_detector, contact_set,
                                                predetermined_control[env.level])
    else:
        ds, pm = RetrievalGetter.prior(env, use_tsf=UseTsf.NO_TRANSFORM)
        u_min, u_max = env.get_control_bounds()
        rand.seed(seed)
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
    pose_error_per_step = {}

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

                # pick object with lowest variance in its translation estimate
                translations = T[:, :2, 2]
                best_tsf_distances = (translations.var(dim=0).sum()).item()

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
            best_T = best_tsf_guess.inverse()

            target_pose = p.getBasePositionAndOrientation(env.target_object_id)
            yaw = p.getEulerFromQuaternion(target_pose[1])[-1]
            target_pose = [target_pose[0][0], target_pose[0][1], yaw]

            guess_pose = [best_T[0, 2].item(), best_T[1, 2].item(), rot_2d_mat_to_angle(best_T.view(1, 3, 3)).item()]
            pos_err, yaw_err = pose_error(target_pose, guess_pose)

            pose_error_per_step[simTime] = pos_err + 0.3 * yaw_err
            logger.info(f"pose error {simTime}: {pos_err} {yaw_err} {pose_error_per_step[simTime]}")
            transformed_model_points = mph @ best_T.transpose(-1, -2)
            for i, pt in enumerate(transformed_model_points):
                if i % 2 == 0:
                    continue
                pt = [pt[0], pt[1], z]
                env._dd.draw_point(f"tmptbest-{i}", pt, color=(0, 0, 1), length=0.008)

        if torch.is_tensor(action):
            action = action.cpu()

        action = np.array(action).flatten()
        obs, rew, done, info = env.step(action)


def main():
    force_gui = True
    env = RetrievalGetter.env(level=Levels.SIMPLE_CLUTTER, mode=p.GUI if force_gui else p.DIRECT)
    run_retrieval(env, seed=1)


if __name__ == "__main__":
    main()
