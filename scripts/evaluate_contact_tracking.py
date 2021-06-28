import torch
import logging
import os
from datetime import datetime
import scipy.io
import numpy as np
import os.path
import re
import pybullet as p
import typing

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn import metrics

from arm_pytorch_utilities.optim import get_device
from arm_pytorch_utilities import preprocess

from tampc import cfg, contact
from tampc.env import pybullet_env as env_base
from tampc.dynamics import hybrid_model
from tampc.env import arm
from tampc.dynamics.hybrid_model import OnlineAdapt
from tampc.util import no_tsf_preprocessor, UseTsf
from tampc.controller.online_controller import StateToPositionTransformer
from tampc.env_getters.arm import ArmGetter

ch = logging.StreamHandler()
fh = logging.FileHandler(os.path.join(cfg.ROOT_DIR, "logs", "{}.log".format(datetime.now())))

logging.basicConfig(level=logging.INFO,
                    format='[%(levelname)s %(asctime)s %(pathname)s:%(lineno)d] %(message)s',
                    datefmt='%m-%d %H:%M:%S', handlers=[ch, fh])

logging.getLogger('matplotlib.font_manager').disabled = True

logger = logging.getLogger(__name__)

# by convention -1 refers to not in contact
NO_CONTACT_ID = -1

prefix_to_environment = {'arm/gripper': arm.FloatingGripperEnv}

all_res = {}


def extract_env_and_level_from_string(string) -> typing.Optional[
    typing.Tuple[typing.Type[env_base.PybulletEnv], int, int]]:
    for name, env_cls in prefix_to_environment.items():
        m = re.search(r"{}\d+".format(name), string)
        if m is not None:
            # find level, which is number succeeding this string match
            level = int(m.group()[len(name):])
            seed = int(os.path.splitext(os.path.basename(string))[0])
            return env_cls, level, seed

    return None


def load_file(datafile, show_in_place=False):
    if not os.path.exists(datafile):
        raise RuntimeError(f"File doesn't exist")

    ret = extract_env_and_level_from_string(datafile)
    if ret is None:
        raise RuntimeError(f"Path not properly formatted to extract environment and level")
    env_cls, level, seed = ret

    # load data
    d = scipy.io.loadmat(datafile)
    mask = d['mask'].reshape(-1) != 0
    # use environment specific state difference function since not all states are R^n
    dX = env_cls.state_difference(d['X'][1:], d['X'][:-1])
    dX = dX[mask[:-1]]
    X = d['X'][mask]
    U = d['U'][mask]

    contact_id = d['contact_id'][mask].reshape(-1)
    ids, counts = np.unique(contact_id, return_counts=True)
    unique_contact_counts = dict(zip(ids, counts))
    steps_taken = sum(unique_contact_counts.values())
    # sort by frequency
    unique_contact_counts = dict(sorted(unique_contact_counts.items(), key=lambda item: item[1], reverse=True))

    # reject if we haven't made sufficient contact
    freespace_ratio = unique_contact_counts[NO_CONTACT_ID] / steps_taken
    if len(unique_contact_counts) < 2 or freespace_ratio > 0.95:
        raise RuntimeWarning(f"Too few contacts; spends {freespace_ratio} ratio in freespace")
    logger.info(f"{datafile} freespace ratio {freespace_ratio} unique contact IDs {unique_contact_counts}")

    obj_poses = {}
    for unique_obj_id in unique_contact_counts.keys():
        if unique_obj_id == NO_CONTACT_ID:
            continue
        # not saved for the time step, so adjust mask usage
        obj_poses[unique_obj_id] = d[f"obj{unique_obj_id}pose"][mask[1:]]

    reactions = d['reaction'][mask]
    in_contact = contact_id != NO_CONTACT_ID

    # TODO compute inherent ambiguity of each contact so as to not penalize misclassifications on ambiguous contacts
    # a way to measure ambiguity is distance of objects to robot (same distance is more ambiguous)
    # another way is cosine similarity between vector of [robot->obj A], [robot->obj B]
    # should be able to combine those

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

    # our method
    env = ArmGetter.env(level=level, mode=p.DIRECT)
    rep_name = None
    use_tsf = UseTsf.NO_TRANSFORM
    ds, pm = ArmGetter.prior(env, use_tsf, rep_name=rep_name)

    dss = [ds]
    ensemble = []
    for s in range(10):
        _, pp = ArmGetter.prior(env, use_tsf, rep_name=rep_name, seed=s)
        ensemble.append(pp.dyn_net)

    hybrid_dynamics = hybrid_model.HybridDynamicsModel(dss, pm, env.state_difference, env.state_distance_two_arg,
                                                       [use_tsf.name],
                                                       device=get_device(),
                                                       preprocessor=no_tsf_preprocessor(),
                                                       nominal_model_kwargs={'online_adapt': OnlineAdapt.NONE},
                                                       ensemble=ensemble,
                                                       project_by_default=True)

    contact_params = ArmGetter.contact_parameters(env)
    contact_preprocessing = preprocess.PytorchTransformer(
        StateToPositionTransformer(contact_params.state_to_pos, contact_params.pos_to_state,
                                   contact_params.max_pos_move_per_action, 2),
        StateToPositionTransformer(contact_params.state_to_pos, contact_params.pos_to_state,
                                   contact_params.max_pos_move_per_action, 0))

    def create_contact_object():
        return contact.ContactObject(hybrid_dynamics.create_empty_local_model(use_prior=True,
                                                                              preprocessor=contact_preprocessing,
                                                                              nom_projection=False), contact_params)

    contact_set = contact.ContactSet(contact_params, contact_object_factory=create_contact_object)
    labels = np.zeros(len(contact_id) - 1)
    x = torch.from_numpy(X).to(device=pm.dyn_net.device, dtype=pm.dyn_net.dtype)
    u = torch.from_numpy(U).to(device=pm.dyn_net.device, dtype=pm.dyn_net.dtype)
    r = torch.from_numpy(reactions).to(device=pm.dyn_net.device, dtype=pm.dyn_net.dtype)
    for i in range(len(X) - 1):
        c = contact_set.update(x[i], u[i], env.state_difference(x[i + 1], x[i]).reshape(-1), r[i + 1])
        labels[i] = -1 if c is None else hash(c)
    env.close()

    # TODO cluster iteratively, using our tracking to move the points to get fairer baselines
    # simple baselines
    # cluster in [pos, next reaction force, action] space
    xx = np.concatenate([X[:-1, :2], reactions[1:], U[:-1]], axis=1)
    # give it some help by telling it the number of clusters as unique contacts IDs
    kmeans = KMeans(n_clusters=len(unique_contact_counts), random_state=0).fit(xx)
    dbscan = DBSCAN(eps=0.5, min_samples=10).fit(xx)

    # we care about homogenity more than completeness - multiple objects in a single cluster is more dangerous
    h, c, v = clustering_metrics(contact_id[:-1], kmeans.labels_, beta=0.5)
    logger.info(f"kmeans h {h} c {c} v {v}")
    h, c, v = clustering_metrics(contact_id[:-1], dbscan.labels_, beta=0.5)
    logger.info(f"dbscan h {h} c {c} v {v}")
    h, c, v = clustering_metrics(contact_id[:-1], labels, beta=0.5)
    logger.info(f"ours h {h} c {c} v {v}")
    all_res[(level, seed)] = h, c, v

    # simplified plot in 2D
    def cluster_id_to_str(cluster_id):
        return 'no contact' if cluster_id == NO_CONTACT_ID else 'contact {}'.format(cluster_id)

    save_loc = "/home/zhsh/Documents/results/2021-06-22/6 clustering res/"

    def save_and_close_fig(f, name):
        plt.savefig(os.path.join(save_loc, f"{level} {seed} {name}"))
        plt.close(f)

    # # ground truth
    f = plot_cluster_res(contact_id, X, f"Task {level} {datafile.split('/')[-1]} ground truth",
                         label_function=cluster_id_to_str)
    save_and_close_fig(f, 'gt')

    # comparison of methods
    f = plot_cluster_res(kmeans.labels_, xx, f"Task {level} {datafile.split('/')[-1]} K-means")
    save_and_close_fig(f, 'kmeans')
    f = plot_cluster_res(dbscan.labels_, xx, f"Task {level} {datafile.split('/')[-1]} DBSCAN")
    save_and_close_fig(f, 'dbscan')
    f = plot_cluster_res(labels, xx, f"Task {level} {datafile.split('/')[-1]} ours")
    save_and_close_fig(f, 'ours')
    #
    # plt.show()

    return X, U, dX, contact_id, obj_poses, reactions


def clustering_metrics(labels_true, labels_pred, beta=1.):
    # beta < 1 means more weight for homogenity
    return metrics.homogeneity_score(labels_true, labels_pred), \
           metrics.completeness_score(labels_true, labels_pred), \
           metrics.v_measure_score(labels_true, labels_pred, beta=beta)


def plot_cluster_res(labels, xx, name, label_function=None):
    f = plt.figure()
    f.suptitle(name)
    ax = plt.gca()
    ids, counts = np.unique(labels, return_counts=True)
    sklearn_cluster_counts = dict(zip(ids, counts))
    sklearn_cluster_counts = dict(sorted(sklearn_cluster_counts.items(), key=lambda item: item[1], reverse=True))
    for i, cluster_id in enumerate(sklearn_cluster_counts.keys()):
        x = xx[labels == cluster_id]
        pos = x[:, :2]
        if label_function is not None:
            ax.scatter(pos[:, 0], pos[:, 1], label=label_function(cluster_id))
        else:
            ax.scatter(pos[:, 0], pos[:, 1])
    set_position_axis_bounds(ax, 0.7)
    ax.legend()
    return f


def set_position_axis_bounds(ax, bound=0.7):
    ax.set_xlim(-bound, bound)
    ax.set_ylim(-bound, bound)
    ax.set_xlabel('x')
    ax.set_xlabel('y')


data = {}

dirs = ['arm/gripper10', 'arm/gripper11', 'arm/gripper12', 'arm/gripper13']

for res_dir in dirs:
    # full_dir = os.path.join(cfg.DATA_DIR, 'arm/gripper10')
    full_dir = os.path.join(cfg.DATA_DIR, res_dir)

    files = os.listdir(full_dir)
    files = sorted(files)

    for file in files:
        full_filename = '{}/{}'.format(full_dir, file)
        # # some interesting ones filtered
        # if file not in ['16.mat', '18.mat', '22.mat']:
        #     continue
        if os.path.isdir(full_filename):
            continue
        try:
            data[full_filename] = load_file(full_filename)
        except (RuntimeError, RuntimeWarning) as e:
            logger.info(f"{full_filename} error: {e}")
            continue

for k, v in all_res.items():
    pretty_v = [round(metric, 2) for metric in v]
    logger.info(f"{k} : {pretty_v}")
