from typing import Type

import torch
import time
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
from cottun.defines import NO_CONTACT_ID, RunKey, CONTACT_RES_FILE, RUN_AMBIGUITY, CONTACT_ID
from cottun.script_utils import extract_env_and_level_from_string, dict_to_namespace_str, record_metric, \
    plot_cluster_res, load_runs_results, get_file_metainfo, clustering_metrics
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.cluster import Birch

from arm_pytorch_utilities.optim import get_device

from tampc import cfg
from cottun import contact
from tampc.env import pybullet_env as env_base, arm
from tampc.env.env import InfoKeys
from tampc.env_getters.arm import ArmGetter
from tampc.env.pybullet_env import ContactInfo

from cottun.cluster_baseline import KMeansWithAutoK
from cottun.cluster_baseline import OnlineSklearnFixedClusters
from cottun.cluster_baseline import OnlineAgglomorativeClustering

ch = logging.StreamHandler()
fh = logging.FileHandler(os.path.join(cfg.ROOT_DIR, "logs", "{}.log".format(datetime.now())))

logging.basicConfig(level=logging.INFO,
                    format='[%(levelname)s %(asctime)s %(pathname)s:%(lineno)d] %(message)s',
                    datefmt='%m-%d %H:%M:%S', handlers=[ch, fh])

logging.getLogger('matplotlib.font_manager').disabled = True

logger = logging.getLogger(__name__)


def our_method_factory(contact_object_class: Type[contact.ContactObject] = contact.ContactUKF, **kwargs):
    def our_method(X, U, reactions, env_class, info):
        # TODO select getter based on env class
        contact_params = ArmGetter.contact_parameters(env_class, **kwargs)
        d = get_device()
        dtype = torch.float32

        def create_contact_object():
            return contact_object_class(None, contact_params)

        contact_set = contact.ContactSetHard(contact_params, contact_object_factory=create_contact_object)
        labels = np.zeros(len(X) - 1)
        x = torch.from_numpy(X).to(device=d, dtype=dtype)
        u = torch.from_numpy(U).to(device=d, dtype=dtype)
        r = torch.from_numpy(reactions).to(device=d, dtype=dtype)
        info[InfoKeys.DEE_IN_CONTACT] = torch.from_numpy(info[InfoKeys.DEE_IN_CONTACT]).to(device=d, dtype=dtype)
        obj_id = 0
        for i in range(len(X) - 1):
            this_info = {
                InfoKeys.OBJ_POSES: {obj_id: obj_pose[i] for obj_id, obj_pose in info[InfoKeys.OBJ_POSES].items()},
                InfoKeys.DEE_IN_CONTACT: info[InfoKeys.DEE_IN_CONTACT][i]
            }
            c, cc = contact_set.update(x[i], u[i], env_class.state_difference(x[i + 1], x[i]).reshape(-1), r[i + 1],
                                       this_info)
            if c is None:
                labels[i] = NO_CONTACT_ID
            else:
                this_obj_id = getattr(c, 'id', None)
                if this_obj_id is None:
                    setattr(c, 'id', obj_id)
                    obj_id += 1
                labels[i] = c.id
                # merged, have to set all labels previously assigned to new obj
                if len(cc) > 1:
                    for other_c in cc:
                        labels[labels == other_c.id] = c.id
        param_values = str(contact_params).split('(')
        start = param_values[0]
        param_values = param_values[1].split(',')
        # remove the functional values
        param_values = [param for param in param_values if '<' not in param]
        param_str_values = f"{start}({','.join(param_values)}"
        # pass where we think contact will be made
        contact_pts = torch.cat([cc.points for cc in contact_set], dim=0)
        pt_weights = torch.cat([cc.weight.repeat(len(cc.points)) for cc in contact_set], dim=0)
        return labels, param_str_values, contact_pts.cpu().numpy(), pt_weights.cpu().numpy()

    return our_method


def our_soft_method_factory(**kwargs):
    def our_method(X, U, reactions, env_class, info):
        # TODO select getter based on env class
        contact_params = ArmGetter.contact_parameters(env_class, **kwargs)
        d = get_device()
        dtype = torch.float32

        contact_set = contact.ContactSetSoft(contact_params)
        labels = np.zeros(len(X) - 1)
        x = torch.from_numpy(X).to(device=d, dtype=dtype)
        u = torch.from_numpy(U).to(device=d, dtype=dtype)
        r = torch.from_numpy(reactions).to(device=d, dtype=dtype)
        info[InfoKeys.DEE_IN_CONTACT] = torch.from_numpy(info[InfoKeys.DEE_IN_CONTACT]).to(device=d, dtype=dtype)
        obj_id = 0
        for i in range(len(X) - 1):
            this_info = {
                InfoKeys.OBJ_POSES: {obj_id: obj_pose[i] for obj_id, obj_pose in info[InfoKeys.OBJ_POSES].items()},
                InfoKeys.DEE_IN_CONTACT: info[InfoKeys.DEE_IN_CONTACT][i]
            }
            c, cc = contact_set.update(x[i], u[i], env_class.state_difference(x[i + 1], x[i]).reshape(-1), r[i + 1],
                                       this_info)
            if c is None:
                labels[i] = NO_CONTACT_ID
            else:
                this_obj_id = getattr(c, 'id', None)
                if this_obj_id is None:
                    setattr(c, 'id', obj_id)
                    obj_id += 1
                labels[i] = c.id
                # merged, have to set all labels previously assigned to new obj
                if len(cc) > 1:
                    for other_c in cc:
                        labels[labels == other_c.id] = c.id
        param_values = str(contact_params).split('(')
        start = param_values[0]
        param_values = param_values[1].split(',')
        # remove the functional values
        param_values = [param for param in param_values if '<' not in param]
        param_str_values = f"{start}({','.join(param_values)}"
        # pass where we think contact will be made
        contact_pts = contact_set.pts
        # TODO compute weights
        pt_weights = torch.ones(contact_pts.shape[0])
        return labels, param_str_values, contact_pts.cpu().numpy(), pt_weights.cpu().numpy()

    return our_method


def process_labels_with_noise(labels):
    noise_label = max(labels) + 1
    for i in range(len(labels)):
        # some methods use -1 to indicate noise; in this case we have to assign a cluster so we use a single element
        if labels[i] == -1:
            labels[i] = noise_label
            noise_label += 1
    return labels


def sklearn_method_factory(method, **kwargs):
    def sklearn_method(X, U, reactions, env_class, info):
        # simple baselines
        # cluster in [pos, next reaction force, action] space
        xx = np.concatenate([X[:-1, :2], reactions[1:], U[:-1]], axis=1)
        valid = np.linalg.norm(reactions[1:], axis=1) > 0.1
        xx = xx[valid]
        # give it some help by telling it the number of clusters as unique contacts IDs
        res = method(**kwargs).fit(xx)
        # return res.labels_, dict_to_namespace_str(kwargs)
        labels = np.ones(len(valid)) * NO_CONTACT_ID
        res.labels_ = process_labels_with_noise(res.labels_)
        labels[valid] = res.labels_
        return labels, dict_to_namespace_str(kwargs), xx, np.ones(len(xx))

    return sklearn_method


def online_sklearn_method_factory(online_class, method, inertia_ratio=0.5, **kwargs):
    def sklearn_method(X, U, reactions, env_class, info):
        online_method = online_class(method(**kwargs), inertia_ratio=inertia_ratio)
        valid = np.linalg.norm(reactions[1:], axis=1) > 0.1
        for i in range(len(X) - 1):
            if not valid[i]:
                continue
            # intermediate labels in case we want plotting of movement
            labels = online_method.update(X[i], U[i], info[InfoKeys.DEE_IN_CONTACT][i],
                                          reactions[i + 1])
        labels = np.ones(len(valid)) * NO_CONTACT_ID
        labels[valid] = process_labels_with_noise(online_method.final_labels())
        moved_pts = online_method.moved_data()
        return labels, dict_to_namespace_str(kwargs), moved_pts, np.ones(len(moved_pts))

    return sklearn_method


def compute_contact_error(before_moving_pts, moved_pts, env_cls: Type[arm.ArmEnv], level, obj_poses,
                          visualize=False):
    contact_error = []
    if moved_pts is not None:
        # set the gripper away from other objects so that physics don't deform the fingers
        env = env_cls(init=(100, 100), environment_level=level, mode=p.GUI if visualize else p.DIRECT, log_video=True)
        env.extrude_objects_in_z = True
        # to make object IDs consistent (after a reset the object IDs may not be in previously created order)
        env.reset()
        for obj_id, poses in obj_poses.items():
            pos = poses[-1, :3]
            orientation = poses[-1, 3:]
            p.resetBasePositionAndOrientation(obj_id, pos, orientation)

        if visualize:
            # visualize all the moved points
            state_c, action_c = env_base.state_action_color_pairs[0]
            env.visualize_state_actions("movedpts", moved_pts, None, state_c, action_c, 0.1)
            state_c, action_c = env_base.state_action_color_pairs[1]
            env.visualize_state_actions("premovepts", before_moving_pts, None, state_c, action_c, 0.1)
            env._dd.clear_visualization_after("movedpts", 0)
            env._dd.clear_visualization_after("premovepts", 0)

        for point in moved_pts:
            env.set_state(np.r_[point, 0, 0])
            p.performCollisionDetection()

            distances = []
            for obj_id in env.movable + env.immovable:
                c = p.getClosestPoints(obj_id, env.robot_id, 100000)

                if visualize:
                    # visualize the points on the robot and object
                    for cc in c:
                        env._dd.draw_point("contactA", cc[ContactInfo.POS_A], color=(1, 0, 0))
                        env._dd.draw_point("contactB", cc[ContactInfo.POS_B], color=(1, 0, 0))
                        env._dd.draw_2d_line("contact between", cc[ContactInfo.POS_A],
                                             np.subtract(cc[ContactInfo.POS_B], cc[ContactInfo.POS_A]), scale=1,
                                             color=(0, 1, 0))
                        env.draw_user_text(str(round(cc[ContactInfo.DISTANCE], 3)), xy=(0.3, 0.5, -1))

                # for multi-link bodies, will return 1 per combination; store the min
                distances.append(min(cc[ContactInfo.DISTANCE] for cc in c))
            contact_error.append(min(distances))
        logger.info(f"largest penetration: {round(min(contact_error), 4)}")
        env.close()
    return contact_error


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

    in_contact = contact_id[:-1] != NO_CONTACT_ID
    for method_name, method in methods.items():
        # additional info to pass to methods for debugging
        info = {InfoKeys.OBJ_POSES: obj_poses, InfoKeys.DEE_IN_CONTACT: d[InfoKeys.DEE_IN_CONTACT]}

        labels, param_values, moved_points, pt_weights = method(X, U, reactions, env_cls, info)
        run_key = RunKey(level=level, seed=seed, method=method_name, params=param_values)
        m = clustering_metrics(contact_id[:-1][in_contact], labels[in_contact])
        contact_error = compute_contact_error(X[:-1][in_contact], moved_points, env_cls, level,
                                              obj_poses)
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
        # 'ours soft': our_soft_method_factory(length=0.1),
        'ours UKF': our_method_factory(length=0.1),
        # 'ours UKF convexity merge constraint': our_method_factory(length=0.1),
        # 'ours PF': our_method_factory(contact_object_class=contact.ContactPF, length=0.1),
        # 'kmeans': sklearn_method_factory(KMeansWithAutoK),
        # 'dbscan': sklearn_method_factory(DBSCAN, eps=1.0, min_samples=10),
        # 'birch': sklearn_method_factory(Birch, n_clusters=None, threshold=1.5),
        # 'online-kmeans': online_sklearn_method_factory(OnlineSklearnFixedClusters, KMeans, n_clusters=1,
        #                                                random_state=0),
        # 'online-dbscan': online_sklearn_method_factory(OnlineAgglomorativeClustering, DBSCAN, eps=1.0, min_samples=5),
        # 'online-birch': online_sklearn_method_factory(OnlineAgglomorativeClustering, Birch, n_clusters=None,
        #                                               threshold=1.5)
    }

    # full_filename = os.path.join(cfg.DATA_DIR, 'arm/gripper12/17.mat')
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
