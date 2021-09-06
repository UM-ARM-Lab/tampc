import abc
import copy
import typing
import time
import math
import torch
import pybullet as p
import numpy as np
import logging
import os
from datetime import datetime

from sklearn.cluster import Birch, DBSCAN, KMeans

from cottun.cluster_baseline import process_labels_with_noise, OnlineAgglomorativeClustering, OnlineSklearnFixedClusters
from cottun.defines import NO_CONTACT_ID
from cottun.evaluation import compute_contact_error, clustering_metrics
from cottun.retrieval_controller import RetrievalPredeterminedController, rot_2d_mat_to_angle, \
    sample_model_points, pose_error, OursRetrievalPredeterminedController
from tampc.controller import controller
from tampc.env.env import InfoKeys

from arm_pytorch_utilities import rand, tensor_utils, math_utils

from tampc import cfg
from cottun import tracking, detection
from tampc.env import arm
from tampc.env.arm import Levels
from tampc.env_getters.arm import RetrievalGetter
from tampc.env.pybullet_env import ContactInfo, state_action_color_pairs
from cottun import icp

ch = logging.StreamHandler()
fh = logging.FileHandler(os.path.join(cfg.ROOT_DIR, "logs", "{}.log".format(datetime.now())))

logging.basicConfig(level=logging.INFO,
                    format='[%(levelname)s %(asctime)s %(pathname)s:%(lineno)d] %(message)s',
                    datefmt='%m-%d %H:%M:%S', handlers=[ch, fh])

logging.getLogger('matplotlib.font_manager').disabled = True

logger = logging.getLogger(__name__)


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
        env.vis.draw_point(f"mpt.{i}", pt, color=(0, 0, 1), length=0.003)

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
        env.vis.draw_point(f"tmpt.{i}", pt, color=(0, 1, 0), length=0.003)

    while True:
        env.step([0, 0])
        time.sleep(0.2)


def object_robot_penetration_score(pt_to_config, config, object_transform, model_pts):
    """Compute the penetration between object and robot for a given transform of the object"""
    # transform model points by object transform
    transformed_model_points = model_pts @ object_transform.transpose(-1, -2)
    d = pt_to_config(config[:, :2], transformed_model_points[:, :2])
    d = d.min().item()
    return -d


class TrackingMethod:
    """Common interface for each tracking method including ours and baselines"""

    @abc.abstractmethod
    def __iter__(self):
        """Iterating over this provides a set of contact points corresponding to an object"""

    @abc.abstractmethod
    def create_predetermined_controller(self, controls):
        """Return a predetermined controller that updates the method when querying for a command"""

    @abc.abstractmethod
    def visualize_contact_points(self, env):
        """Render the tracked contact points in the given environment"""

    @abc.abstractmethod
    def get_labelled_moved_points(self, labels):
        """Return the final position of the tracked points as well as their object label"""


class SoftTrackingIterator:
    def __init__(self, pts, to_iter):
        self.pts = pts
        self.to_iter = to_iter

    def __next__(self):
        indices = next(self.to_iter)
        return self.pts[indices]


class OurTrackingMethod(TrackingMethod):
    def __init__(self, env):
        self.env = env
        self.ctrl = None

    @property
    @abc.abstractmethod
    def contact_set(self) -> tracking.ContactSet:
        """Return some contact set"""

    def visualize_contact_points(self, env):
        env.visualize_contact_set(self.contact_set)

    def create_predetermined_controller(self, controls):
        self.ctrl = OursRetrievalPredeterminedController(self.env.contact_detector, self.contact_set, controls)
        return self.ctrl


class OurSoftTrackingMethod(OurTrackingMethod):
    def __init__(self, env):
        self.contact_params = RetrievalGetter.contact_parameters(env)
        self._contact_set = tracking.ContactSetSoft(arm.ArmPointToConfig(env), self.contact_params)
        super(OurSoftTrackingMethod, self).__init__(env)

    @property
    def contact_set(self) -> tracking.ContactSetSoft:
        return self._contact_set

    def __iter__(self):
        pts = self.contact_set.get_posterior_points()
        to_iter = self.contact_set.get_hard_assignment(self.contact_set.p.hard_assignment_threshold)
        return SoftTrackingIterator(pts, iter(to_iter))

    def get_labelled_moved_points(self, labels):
        groups = self.contact_set.get_hard_assignment(self.contact_params.hard_assignment_threshold)
        contact_indices = torch.tensor(self.ctrl.contact_indices, device=groups[0].device)
        # self.ctrl.contact_indices = []

        for group_id, group in enumerate(groups):
            labels[contact_indices[group].cpu().numpy()] = group_id + 1

        contact_pts = self.contact_set.get_posterior_points()
        return labels, contact_pts


class HardTrackingIterator:
    def __init__(self, contact_objs):
        self.contact_objs = contact_objs

    def __next__(self):
        object: tracking.ContactObject = next(self.contact_objs)
        return object.points


class OurHardTrackingMethod(OurTrackingMethod):
    def __init__(self, env):
        self.contact_params = RetrievalGetter.contact_parameters(env)
        self._contact_set = tracking.ContactSetHard(self.contact_params,
                                                    contact_object_factory=self.create_contact_object)
        super(OurHardTrackingMethod, self).__init__(env)

    @property
    def contact_set(self) -> tracking.ContactSetHard:
        return self._contact_set

    def __iter__(self):
        return HardTrackingIterator(iter(self.contact_set))

    def create_contact_object(self):
        return tracking.ContactUKF(None, self.contact_params)


class SklearnPredeterminedController(RetrievalPredeterminedController):

    def __init__(self, online_method, contact_detector: detection.ContactDetector, controls, nu=None):
        super().__init__(controls, nu=nu)
        self.online_method = online_method
        self.contact_detector = contact_detector
        self.in_contact = []

    def update(self, obs, info):
        contact_point = self.contact_detector.get_last_contact_location()
        if contact_point is not None:
            self.in_contact.append(True)
            contact_point = contact_point.cpu().numpy()
            dobj = info[InfoKeys.DEE_IN_CONTACT]
            self.online_method.update(contact_point - dobj, self.u_history[-1], dobj)
        else:
            self.in_contact.append(False)


class SklearnTrackingMethod(TrackingMethod):
    def __init__(self, env, online_class, method, inertia_ratio=0.5, **kwargs):
        self.env = env
        self.online_method = online_class(method(**kwargs), inertia_ratio=inertia_ratio)
        self.ctrl: typing.Optional[SklearnPredeterminedController] = None

    def __iter__(self):
        moved_pts = self.online_method.moved_data()
        return iter(moved_pts)

    def create_predetermined_controller(self, controls):
        self.ctrl = SklearnPredeterminedController(self.online_method, self.env.contact_detector, controls, nu=2)
        return self.ctrl

    def visualize_contact_points(self, env):
        # valid = self.ctrl.in_contact
        # labels = np.ones(len(valid)) * NO_CONTACT_ID
        moved_pt_labels = process_labels_with_noise(self.online_method.final_labels())
        moved_pts = self.online_method.moved_data()
        # labels[valid] = moved_pt_labels
        for i, obj_id in enumerate(np.unique(moved_pt_labels)):
            if obj_id == NO_CONTACT_ID:
                continue

            indices = moved_pt_labels == obj_id
            color, u_color = state_action_color_pairs[i % len(state_action_color_pairs)]
            base_name = str(i)
            self.env.visualize_state_actions(base_name, moved_pts[indices], None, color, u_color, 0)

    def get_labelled_moved_points(self, labels):
        labels[1:][self.ctrl.in_contact] = process_labels_with_noise(self.online_method.final_labels())
        moved_pts = self.online_method.moved_data()
        return labels, moved_pts


def run_retrieval(env, method: TrackingMethod, seed=0, ctrl_noise_max=0.005):
    dtype = torch.float32

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
    rand.seed(0)
    noise = (np.random.rand(len(ctrl), 2) - 0.5) * 0.5
    ctrl = np.add(ctrl, noise)
    predetermined_control[Levels.SIMPLE_CLUTTER] = ctrl

    ctrl = [[0.9, -0.3]] * 2
    ctrl += [[1.0, 0.], [-.2, -0.6]] * 6
    ctrl += [[0.1, -0.9], [0.8, -0.6]] * 1
    ctrl += [[0.1, 0.8], [0.1, 0.9], [0.1, -1.0], [0.1, -1.0], [0.2, -1.0]]
    ctrl += [[0.1, 0.8], [0.1, 0.9], [0.3, 0.8], [0.4, 0.6]]
    ctrl += [[0.1, -0.8], [0.1, -0.9], [0.3, -0.8]]
    ctrl += [[-0.2, 0.8], [0.1, 0.9], [0.3, 0.8]]
    ctrl += [[-0., -0.8], [0.1, -0.7]]
    ctrl += [[0.4, -0.5], [0.2, -1.0]]
    ctrl += [[-0.2, -0.5], [0.2, -0.6]]
    ctrl += [[0.2, 0.4], [0.2, -1.0]]
    ctrl += [[0.4, 0.4], [0.4, -0.4]] * 3
    ctrl += [[-0.5, 1.0]] * 3
    ctrl += [[0.5, 0.], [0, 0.4]] * 3
    predetermined_control[Levels.TIGHT_CLUTTER] = ctrl

    rand.seed(seed)
    for k, v in predetermined_control.items():
        predetermined_control[k] = np.add(v, (np.random.rand(len(v), 2) - 0.5) * ctrl_noise_max)

    ctrl = method.create_predetermined_controller(predetermined_control[env.level])

    obs = env.reset()
    z = env._observe_ee(return_z=True)[-1]

    model_points = sample_model_points(env.target_object_id, num_points=50, force_z=z, seed=0, name="cheezit")
    mph = model_points.clone().to(dtype=dtype)
    # make homogeneous [x, y, 1]
    mph[:, -1] = 1

    ctrl.set_goal(env.goal[:2])
    info = None
    simTime = 0
    best_tsf_guess = None
    guess_pose = None
    pose_error_per_step = {}

    pt_to_config = arm.ArmPointToConfig(env)

    contact_id = []

    while not ctrl.done():
        best_distance = None
        simTime += 1
        env.draw_user_text("{}".format(simTime), xy=(0.5, 0.7, -1))

        action = ctrl.command(obs, info)
        method.visualize_contact_points(env)
        if env.contact_detector.in_contact():
            contact_id.append(info[InfoKeys.CONTACT_ID])
            all_configs = torch.tensor(ctrl.x_history, dtype=dtype, device=mph.device).view(-1, env.nx)
            dist_per_est_obj = []
            for this_pts in method:
                this_pts = tensor_utils.ensure_tensor(model_points.device, dtype, this_pts)
                T, distances, _ = icp.icp_3(this_pts.view(-1, 2), model_points[:, :2],
                                            given_init_pose=best_tsf_guess, batch=30)
                T = T.inverse()
                penetration = [object_robot_penetration_score(pt_to_config, all_configs, T[b], mph) for b in
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

            logger.debug(f"err each obj {np.round(dist_per_est_obj, 4)}")
            best_T = best_tsf_guess.inverse()

            target_pose = p.getBasePositionAndOrientation(env.target_object_id)
            yaw = p.getEulerFromQuaternion(target_pose[1])[-1]
            target_pose = [target_pose[0][0], target_pose[0][1], yaw]

            guess_pose = [best_T[0, 2].item(), best_T[1, 2].item(), rot_2d_mat_to_angle(best_T.view(1, 3, 3)).item()]
            pos_err, yaw_err = pose_error(target_pose, guess_pose)

            pose_error_per_step[simTime] = pos_err + 0.3 * yaw_err
            logger.debug(f"pose error {simTime}: {pos_err} {yaw_err} {pose_error_per_step[simTime]}")
            transformed_model_points = mph @ best_T.transpose(-1, -2)
            for i, pt in enumerate(transformed_model_points):
                if i % 2 == 0:
                    continue
                pt = [pt[0], pt[1], z]
                env.vis.draw_point(f"tmptbest.{i}", pt, color=(0, 0, 1), length=0.008)
        else:
            contact_id.append(NO_CONTACT_ID)

        if torch.is_tensor(action):
            action = action.cpu()

        action = np.array(action).flatten()
        obs, rew, done, info = env.step(action)

    # evaluate FMI and contact error here
    labels, moved_points = method.get_labelled_moved_points(np.ones(len(contact_id)) * NO_CONTACT_ID)
    contact_id = np.array(contact_id)

    in_label_contact = contact_id != NO_CONTACT_ID

    m = clustering_metrics(contact_id[in_label_contact], labels[in_label_contact])
    contact_error = compute_contact_error(None, moved_points, env=env, visualize=False)
    cme = np.mean(np.abs(contact_error))

    # attempt grasp
    predetermined_grasp_offset = {
        Levels.TIGHT_CLUTTER: [0., -0.25]
    }
    grasp_offset = predetermined_grasp_offset[env.level]
    # object is symmetric so pose can be off by 180
    yaw = guess_pose[2]
    if yaw > np.pi / 2:
        yaw -= np.pi
    elif yaw < -np.pi / 2:
        yaw += np.pi
    grasp_offset = math_utils.rotate_wrt_origin(grasp_offset, yaw)
    target_pos = [guess_pose[0] + grasp_offset[0], guess_pose[1] + grasp_offset[1]]
    env.vis.draw_point("pre_grasp", [target_pos[0], target_pos[1], z], color=(1, 0, 0))
    # get to target pos
    diff = np.subtract(target_pos, obs)
    start = time.time()
    while np.linalg.norm(diff) > 0.01 and time.time() - start < 5:
        obs, _, _, _ = env.step(diff / env.MAX_PUSH_DIST)
        diff = np.subtract(target_pos, obs)
    # rotate in place
    prev_ee_orientation = copy.deepcopy(env.endEffectorOrientation)
    env.endEffectorOrientation = p.getQuaternionFromEuler([0, np.pi / 2, yaw + np.pi / 2])
    env.sim_step_wait = 0.01
    env.step([0, 0])
    env.open_gripper()
    env.step([0, 0])
    env.sim_step_wait = None
    # go for the grasp

    move_times = 4
    move_dir = -np.array(grasp_offset)
    while move_times > 0:
        act_mag = move_times if move_times <= 1 else 1
        move_times -= 1
        u = move_dir / np.linalg.norm(move_dir) * act_mag
        obs, _, _, _ = env.step(u)
    env.sim_step_wait = 0.01
    env.close_gripper()
    env.step([0, 0])
    env.sim_step_wait = None

    env.endEffectorOrientation = prev_ee_orientation
    return m, cme


def main(env, method_name, seed=0):
    methods_to_run = {
        'ours': OurSoftTrackingMethod(env),
        'online-birch': SklearnTrackingMethod(env, OnlineAgglomorativeClustering, Birch, n_clusters=None,
                                              inertia_ratio=0.2,
                                              threshold=0.07),
        'online-dbscan': SklearnTrackingMethod(env, OnlineAgglomorativeClustering, DBSCAN, eps=0.05, min_samples=1),
        'online-kmeans': SklearnTrackingMethod(env, OnlineSklearnFixedClusters, KMeans, inertia_ratio=0.2, n_clusters=1,
                                               random_state=0)
    }
    env.draw_user_text(f"{method_name} seed {seed}", xy=[-0.1, 0.28, -0.5])
    return run_retrieval(env, methods_to_run[method_name], seed=seed)


if __name__ == "__main__":
    env = RetrievalGetter.env(level=Levels.TIGHT_CLUTTER, mode=p.GUI)

    method_name = "online-birch"
    fmis = []
    cmes = []
    for seed in range(10):
        m, cme = main(env, method_name, seed=seed)
        fmi = m[0]
        fmis.append(fmi)
        cmes.append(cme)
        logger.info(f"{method_name} fmi {fmi} cme {cme}")
        env.vis.clear_visualizations()
        env.reset()
    logger.info(f"{method_name} mean fmi {np.mean(fmis)} median fmi {np.median(fmis)} std fmi {np.std(fmis)} {fmis}\n"
                f"mean cme {np.mean(cmes)} median cme {np.median(cmes)} std cme {np.std(cmes)} {cmes}")
