import abc
from typing import Type

import torch
import pickle
import logging
import os
from datetime import datetime
import scipy.io
import numpy as np
import os.path
import pybullet as p
import re

import matplotlib.pyplot as plt
import typing

from sklearn.cluster import KMeans, DBSCAN, Birch

from stucco.defines import NO_CONTACT_ID, RunKey, CONTACT_RES_FILE, RUN_AMBIGUITY, CONTACT_ID, CONTACT_POINT_CACHE
from stucco.evaluation import dict_to_namespace_str, plot_cluster_res, load_runs_results, get_file_metainfo, \
    clustering_metrics, compute_contact_error
from stucco.detection_impl import ContactDetectorPlanarPybulletGripper

from arm_pytorch_utilities.optim import get_device

from stucco import cfg
from stucco import tracking
from stucco.env import arm, pybullet_env as env_base
from stucco.env.env import InfoKeys
from stucco.env_getters.arm import ArmGetter

from stucco.cluster_baseline import process_labels_with_noise, OnlineSklearnFixedClusters, OnlineAgglomorativeClustering

ch = logging.StreamHandler()
fh = logging.FileHandler(os.path.join(cfg.ROOT_DIR, "logs", "{}.log".format(datetime.now())))

logging.basicConfig(level=logging.INFO,
                    format='[%(levelname)s %(asctime)s %(pathname)s:%(lineno)d] %(message)s',
                    datefmt='%m-%d %H:%M:%S', handlers=[ch, fh])

logging.getLogger('matplotlib.font_manager').disabled = True

logger = logging.getLogger(__name__)


class OurMethodFactory:
    def __init__(self, **kwargs):
        self.env_kwargs = kwargs
        self.env_class = None
        self.p: typing.Optional[tracking.ContactParameters] = None

    @abc.abstractmethod
    def create_contact_set(self, contact_params) -> tracking.ContactSet:
        """Create a contact set"""

    @abc.abstractmethod
    def get_contact_point_results(self, contact_set) -> typing.Tuple[torch.tensor, torch.tensor]:
        """After running an experiment, retrieve the contact points and weights from a contact set"""

    def get_final_labels(self, labels, contact_set):
        return labels

    def update_labels_single_res(self, labels, i, latest_obj_id, *update_return):
        return latest_obj_id

    def __call__(self, X, U, reactions, env_class, info, contact_detector, contact_pts):
        self.env_class = env_class
        # TODO select getter based on env class
        contact_params = ArmGetter.contact_parameters(env_class, **self.env_kwargs)
        self.p = contact_params
        d = get_device()
        dtype = torch.float32
        contact_detector.to(device=d, dtype=dtype)

        contact_set = self.create_contact_set(contact_params)
        labels = np.zeros(len(X) - 1)
        x = torch.from_numpy(X).to(device=d, dtype=dtype)
        u = torch.from_numpy(U).to(device=d, dtype=dtype)
        r = torch.from_numpy(reactions).to(device=d, dtype=dtype)
        if not torch.is_tensor(info[InfoKeys.DEE_IN_CONTACT]):
            info[InfoKeys.DEE_IN_CONTACT] = torch.from_numpy(info[InfoKeys.DEE_IN_CONTACT]).to(device=d, dtype=dtype)
        obj_id = 0
        for i in range(len(X) - 1):
            this_info = {
                InfoKeys.OBJ_POSES: {obj_id: obj_pose[i] for obj_id, obj_pose in info[InfoKeys.OBJ_POSES].items()},
            }
            for key in [InfoKeys.DEE_IN_CONTACT, InfoKeys.HIGH_FREQ_REACTION_F, InfoKeys.HIGH_FREQ_REACTION_T,
                        InfoKeys.HIGH_FREQ_EE_POSE]:
                this_info[key] = info[key][i]
            # isolate to get contact point and feed that in instead of contact configuration
            ee_force_torque = np.r_[
                this_info[InfoKeys.HIGH_FREQ_REACTION_F][-1], this_info[InfoKeys.HIGH_FREQ_REACTION_T][-1]]
            pose = (this_info[InfoKeys.HIGH_FREQ_EE_POSE][-1][:3], this_info[InfoKeys.HIGH_FREQ_EE_POSE][-1][3:])
            contact_detector.clear()
            contact_detector.observe_residual(ee_force_torque, pose)
            c, cc = contact_set.update(x[i], u[i], env_class.state_difference(x[i + 1], x[i]).reshape(-1),
                                       contact_detector, r[i + 1], this_info)

            if c is None:
                labels[i] = NO_CONTACT_ID
            else:
                obj_id = self.update_labels_single_res(labels, i, obj_id, c, cc)
        labels = self.get_final_labels(labels, contact_set)
        param_values = str(contact_params).split('(')
        start = param_values[0]
        param_values = param_values[1].split(',')
        # remove the functional values
        param_values = [param for param in param_values if '<' not in param]
        param_str_values = f"{start}({','.join(param_values)}"
        # pass where we think contact will be made
        contact_pts, pt_weights = self.get_contact_point_results(contact_set)
        return labels, param_str_values, contact_pts.cpu().numpy(), pt_weights.cpu().numpy()


class OurMethodHard(OurMethodFactory):
    def __init__(self, contact_object_class: Type[tracking.ContactObject] = tracking.ContactUKF, **kwargs):
        self.contact_object_class = contact_object_class
        super(OurMethodHard, self).__init__(**kwargs)

    def update_labels_single_res(self, labels, i, latest_obj_id, *update_return):
        c, cc = update_return
        this_obj_id = getattr(c, 'id', None)
        if this_obj_id is None:
            setattr(c, 'id', latest_obj_id)
            latest_obj_id += 1
        labels[i] = c.id
        # merged, have to set all labels previously assigned to new obj
        if len(cc) > 1:
            for other_c in cc:
                labels[labels == other_c.id] = c.id
        return this_obj_id

    def create_contact_set(self, contact_params) -> tracking.ContactSet:
        def create_contact_object():
            return self.contact_object_class(None, contact_params)

        return tracking.ContactSetHard(contact_params, contact_object_factory=create_contact_object,
                                       device=get_device())

    def get_contact_point_results(self, contact_set: tracking.ContactSetHard) -> typing.Tuple[
        torch.tensor, torch.tensor]:
        contact_pts = torch.cat([cc.points for cc in contact_set], dim=0)
        pt_weights = torch.cat([cc.weight.repeat(len(cc.points)) for cc in contact_set], dim=0)
        return contact_pts, pt_weights


class OurMethodSoft(OurMethodFactory):
    def __init__(self, visualize=False, **kwargs):
        super(OurMethodSoft, self).__init__(**kwargs)
        self.env = None
        self.visualize = visualize
        self.contact_indices = []

    def _pt_to_config_dist(self, configs, pts):
        if self.env is None:
            self.env = self.env_class(mode=p.GUI if self.visualize else p.DIRECT)
            self._dist_calc = arm.ArmPointToConfig(self.env)
        return self._dist_calc(configs, pts)

    def create_contact_set(self, contact_params) -> tracking.ContactSet:
        return tracking.ContactSetSoft(self._pt_to_config_dist, contact_params, device=get_device())

    def update_labels_single_res(self, labels, i, latest_obj_id, *update_return):
        self.contact_indices.append(i)
        return latest_obj_id

    def get_final_labels(self, labels, contact_set: tracking.ContactSetSoft):
        groups = contact_set.get_hard_assignment(self.p.hard_assignment_threshold)
        contact_indices = torch.tensor(self.contact_indices, device=groups[0].device)
        self.contact_indices = []
        for group_id, group in enumerate(groups):
            labels[contact_indices[group].cpu().numpy()] = group_id + 1
        if self.env is not None:
            self.env.close()
            self.env = None
        return labels

    def get_contact_point_results(self, contact_set: tracking.ContactSetSoft) -> typing.Tuple[
        torch.tensor, torch.tensor]:
        contact_pts = contact_set.get_posterior_points()
        # TODO consider some way of weighing the individual points rather than particles?
        pt_weights = torch.ones(contact_pts.shape[0], device=contact_pts.device, dtype=contact_pts.dtype)
        return contact_pts, pt_weights


def sklearn_method_factory(method, **kwargs):
    def sklearn_method(X, U, reactions, env_class, info, contact_detector, contact_pts):
        # simple baselines
        # cluster in [contact point] space
        xx = contact_pts[:, :2]
        valid = np.linalg.norm(contact_pts, axis=1) > 0.
        xx = xx[valid]
        res = method(**kwargs).fit(xx)
        # return res.labels_, dict_to_namespace_str(kwargs)
        labels = np.ones(len(valid)) * NO_CONTACT_ID
        res.labels_ = process_labels_with_noise(res.labels_)
        labels[valid] = res.labels_
        return labels, dict_to_namespace_str(kwargs), xx, np.ones(len(xx))

    return sklearn_method


def online_sklearn_method_factory(online_class, method, inertia_ratio=0.5, **kwargs):
    def sklearn_method(X, U, reactions, env_class, info, contact_detector, contact_pts):
        online_method = online_class(method(**kwargs), inertia_ratio=inertia_ratio)
        valid = np.linalg.norm(contact_pts, axis=1) > 0.
        dobj = info[InfoKeys.DEE_IN_CONTACT]
        for i in range(len(X) - 1):
            if not valid[i]:
                continue
            # intermediate labels in case we want plotting of movement
            labels = online_method.update(contact_pts[i] - dobj[i], U[i], dobj[i])
        labels = np.ones(len(valid)) * NO_CONTACT_ID
        labels[valid] = process_labels_with_noise(online_method.final_labels())
        moved_pts = online_method.moved_data()
        return labels, dict_to_namespace_str(kwargs), moved_pts, np.ones(len(moved_pts))

    return sklearn_method


def get_contact_point_history(data, filename):
    """Return the contact points; for each pts[i], report the point of contact before u[i]"""
    contact_detector = ContactDetectorPlanarPybulletGripper("floating_gripper", np.diag([1, 1, 1, 50, 50, 50]), 1.)
    if os.path.exists(CONTACT_POINT_CACHE):
        with open(CONTACT_POINT_CACHE, 'rb') as f:
            cache = pickle.load(f)
            pts = cache.get(filename, None)
            if pts is not None:
                return contact_detector, pts
    else:
        cache = {}

    pts = []
    for i in range(len(data['X']) - 1):
        for j in range(len(data[InfoKeys.HIGH_FREQ_REACTION_F][i])):
            ee_force_torque = np.r_[
                data[InfoKeys.HIGH_FREQ_REACTION_F][i][j], data[InfoKeys.HIGH_FREQ_REACTION_T][i][j]]
            pose = (data[InfoKeys.HIGH_FREQ_EE_POSE][i][j][:3], data[InfoKeys.HIGH_FREQ_EE_POSE][i][j][3:])
            contact_detector.observe_residual(ee_force_torque, pose)
        pt = contact_detector.get_last_contact_location()
        if pt is None:
            pt = np.r_[0, 0, 0]
        else:
            # get the point before this action
            pt = pt.cpu().numpy() - data[InfoKeys.DEE_IN_CONTACT][i]
        pts.append(pt)

    pts = np.stack(pts)
    cache[filename] = pts
    with open(CONTACT_POINT_CACHE, 'wb') as f:
        pickle.dump(cache, f)

    return contact_detector, pts


def evaluate_methods_on_file(datafile, run_res, methods, show_in_place=False):
    try:
        env_cls, level, seed = get_file_metainfo(datafile)
    except (RuntimeError, RuntimeWarning) as e:
        logger.info(f"{full_filename} error: {e}")
        return None

    # load data
    d = scipy.io.loadmat(datafile)
    # mask = d['mask'].reshape(-1) != 0
    # use environment specific state difference function since not all states are R^n
    dX = env_cls.state_difference(d['X'][1:], d['X'][:-1])
    dX = dX
    X = d['X']
    U = d['U']

    reactions = d['reaction']
    contact_id = d[InfoKeys.CONTACT_ID].reshape(-1)
    # filter out steps that had inconsistent contact (low reaction force but detected contact ID)
    # note that contact ID is 1 index in front of reaction since reactions are what was felt during the last step
    contact_id[np.linalg.norm(reactions[1:], axis=1) < 0.1] = NO_CONTACT_ID
    # make same size as X
    contact_id = np.r_[contact_id, NO_CONTACT_ID]

    ids, counts = np.unique(contact_id, return_counts=True)
    unique_contact_counts = dict(zip(ids, counts))
    steps_taken = sum(unique_contact_counts.values())
    # sort by frequency
    unique_contact_counts = dict(sorted(unique_contact_counts.items(), key=lambda item: item[1], reverse=True))

    # reject if we haven't made sufficient contact
    if NO_CONTACT_ID in unique_contact_counts:
        freespace_ratio = unique_contact_counts[NO_CONTACT_ID] / steps_taken
    else:
        freespace_ratio = 1.
    if len(unique_contact_counts) < 2 or freespace_ratio > 0.95 or \
            sum([v for k, v in unique_contact_counts.items() if k != NO_CONTACT_ID]) < 2:
        logger.info(f"Too few contacts; spends {freespace_ratio} ratio in freespace")
        return None
    logger.info(f"{datafile} freespace ratio {freespace_ratio} unique contact IDs {unique_contact_counts}")

    obj_poses = {}
    pose_key = re.compile('obj(\d+)pose')
    for k in d.keys():
        m = pose_key.match(k)
        if m is not None:
            obj_id = int(m.group(1))
            # not saved for the time step, so adjust mask usage
            obj_poses[obj_id] = d[k]

    contact_id_key = RunKey(level=level, seed=seed, method=CONTACT_ID, params=None)
    run_res[contact_id_key] = contact_id

    ambiguity_key = RunKey(level=level, seed=seed, method=RUN_AMBIGUITY, params=None)
    ambiguity = compute_run_ambiguity(d, unique_contact_counts.keys())
    run_res[ambiguity_key] = ambiguity

    if show_in_place:
        env = env_cls(environment_level=level, mode=p.GUI)
        for i, unique_obj_id in enumerate(unique_contact_counts.keys()):
            if unique_obj_id == NO_CONTACT_ID:
                continue
            x = X[contact_id == unique_obj_id]
            u = U[contact_id == unique_obj_id]
            state_c, action_c = env_base.state_action_color_pairs[(i - 1) % len(env_base.state_action_color_pairs)]
            env.visualize_state_actions(str(i), x, u, state_c, action_c, 0.1)
        env.close()
        import time
        time.sleep(1)

    # simplified plot in 2D
    def cluster_id_to_str(cluster_id):
        return 'no contact' if cluster_id == NO_CONTACT_ID else 'contact {}'.format(cluster_id)

    save_loc = "/home/zhsh/Documents/results/cluster_res/"

    def save_and_close_fig(f, name):
        plt.savefig(os.path.join(save_loc, f"{level} {seed} {name}.png"))
        plt.close(f)

    # ground truth
    f = plot_cluster_res(contact_id, X, f"Task {level} {datafile.split('/')[-1]} ground truth",
                         label_function=cluster_id_to_str)
    save_and_close_fig(f, '')

    info = {k: d[k] for k in [InfoKeys.DEE_IN_CONTACT, InfoKeys.HIGH_FREQ_REACTION_F, InfoKeys.HIGH_FREQ_REACTION_T,
                              InfoKeys.HIGH_FREQ_EE_POSE]}
    # additional info to pass to methods for debugging
    info[InfoKeys.OBJ_POSES] = obj_poses

    contact_detector, contact_pts = get_contact_point_history(d, datafile)
    # those not in contact are marked as 0,0,0
    in_contact = np.linalg.norm(contact_pts, axis=1) > 0
    # some of those points in actual contact are not labelled, so evaluate some metrics on only these
    in_label_contact = contact_id[:-1] != NO_CONTACT_ID

    for method_name, method_list in methods.items():
        if not isinstance(method_list, (list, tuple)):
            method_list = [method_list]

        for method in method_list:
            labels, param_values, moved_points, pt_weights = method(X, U, reactions, env_cls, info, contact_detector,
                                                                    contact_pts)
            run_key = RunKey(level=level, seed=seed, method=method_name, params=param_values)
            m = clustering_metrics(contact_id[:-1][in_label_contact], labels[in_label_contact])
            contact_error = compute_contact_error(contact_pts[in_contact], moved_points, env_cls=env_cls, level=level,
                                                  obj_poses=obj_poses, visualize=False)
            cme = np.mean(np.abs(contact_error))
            # normalize weights
            pt_weights = pt_weights / np.sum(pt_weights)
            wcme = np.sum(np.abs(contact_error) @ pt_weights)
            run_res[run_key] = list(m) + [cme, wcme]

            f = plot_cluster_res(labels, X[:-1], f"Task {level} {datafile.split('/')[-1]} {method_name}")
            save_and_close_fig(f, f"{method_name} {param_values.replace('.', '_')}")

    # plt.show()

    return X, U, dX, contact_id, obj_poses, reactions


def compute_run_ambiguity(data, obj_ids, dist_threshold=0.15):
    """Ambiguity ranges from [0,1] where 1 means the objects may be indistinguishable"""
    # a way to measure ambiguity is distance of objects to robot (same distance is more ambiguous)
    # another way is cosine similarity between vector of [robot->obj A], [robot->obj B]
    # should be able to combine those
    obj_distances = {obj_id: data[f"obj{obj_id}distance"] for obj_id in obj_ids if not obj_id == NO_CONTACT_ID}
    obj_distances = np.stack([np.concatenate(d) for d in obj_distances.values()]).T
    # no ambiguity if there's only 1 object
    if obj_distances.shape[1] < 2:
        return np.zeros(obj_distances.shape[0] + 1)
    # based on second closest distance
    partitioned_distances = np.partition(obj_distances, 1, axis=1)
    ambiguity = np.clip(dist_threshold - partitioned_distances[:, 1], 0, None) / dist_threshold
    # currently this is the ambiguity at the end of the step (start of next step)
    # convert to ambiguity at the start of the step by assuming first step has no ambiguity
    ambiguity = np.insert(ambiguity, 0, 0)
    return ambiguity


if __name__ == "__main__":
    runs = load_runs_results()

    dirs = ['arm/gripper10', 'arm/gripper11', 'arm/gripper12', 'arm/gripper13']
    methods_to_run = {
        # 'ours': [OurMethodSoft(length=0.02, hard_assignment_threshold=0.2),
        #          OurMethodSoft(length=0.02, hard_assignment_threshold=0.3),
        #          OurMethodSoft(length=0.02, hard_assignment_threshold=0.4),
        #          OurMethodSoft(length=0.02, hard_assignment_threshold=0.5),
        #          ],
        # 'ours UKF': OurMethodHard(length=0.1),
        # 'ours UKF convexity merge constraint': OurMethodHard(length=0.1),
        # 'ours PF': OurMethodHard(contact_object_class=tracking.ContactPF, length=0.1),
        # 'kmeans': sklearn_method_factory(KMeansWithAutoK),
        # 'dbscan': sklearn_method_factory(DBSCAN, eps=1.0, min_samples=10),
        # 'birch': sklearn_method_factory(Birch, n_clusters=None, threshold=1.5),
        'online-kmeans': online_sklearn_method_factory(OnlineSklearnFixedClusters, KMeans, inertia_ratio=0.2,
                                                       n_clusters=1, random_state=0),
        'online-dbscan': online_sklearn_method_factory(OnlineAgglomorativeClustering, DBSCAN, eps=0.05, min_samples=1),
        'online-birch': online_sklearn_method_factory(OnlineAgglomorativeClustering, Birch, n_clusters=None,
                                                      threshold=0.08)
    }

    # full_filename = os.path.join(cfg.DATA_DIR, 'arm/gripper13/25.mat')
    # evaluate_methods_on_file(full_filename, runs, methods_to_run)

    for res_dir in dirs:
        full_dir = os.path.join(cfg.DATA_DIR, res_dir)

        files = os.listdir(full_dir)
        files = sorted(files)

        for file in files:
            full_filename = '{}/{}'.format(full_dir, file)
            if os.path.isdir(full_filename):
                continue
            evaluate_methods_on_file(full_filename, runs, methods_to_run)

    with open(CONTACT_RES_FILE, 'wb') as f:
        pickle.dump(runs, f)
        logger.info("saved runs to %s", CONTACT_RES_FILE)
