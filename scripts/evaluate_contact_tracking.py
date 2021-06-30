import torch
import pickle
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
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.cluster import Birch
from sklearn import metrics

from arm_pytorch_utilities.optim import get_device

from tampc import cfg
from cottun import contact
from tampc.env import pybullet_env as env_base
from tampc.env import arm
from tampc.env_getters.arm import ArmGetter

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

# by convention -1 refers to not in contact
NO_CONTACT_ID = -1

prefix_to_environment = {'arm/gripper': arm.FloatingGripperEnv}


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


def our_method(X, U, reactions, env_class):
    # TODO select getter based on env class
    contact_params = ArmGetter.contact_parameters(env_class)
    d = get_device()
    dtype = torch.float32

    def create_contact_object():
        return contact.ContactObject(None, contact_params)

    contact_set = contact.ContactSet(contact_params, contact_object_factory=create_contact_object)
    labels = np.zeros(len(X) - 1)
    x = torch.from_numpy(X).to(device=d, dtype=dtype)
    u = torch.from_numpy(U).to(device=d, dtype=dtype)
    r = torch.from_numpy(reactions).to(device=d, dtype=dtype)
    for i in range(len(X) - 1):
        c = contact_set.update(x[i], u[i], env_class.state_difference(x[i + 1], x[i]).reshape(-1), r[i + 1])
        labels[i] = -1 if c is None else hash(c)
    param_values = str(contact_params).split('(')
    start = param_values[0]
    param_values = param_values[1].split(',')
    # remove the functional values
    param_values = [param for param in param_values if '<' not in param]
    param_str_values = f"{start}({','.join(param_values)}"
    return labels, param_str_values


def dict_to_namespace_str(d):
    return str(d).replace(': ', '=').replace('\'', '').strip('{}')


def sklearn_method_factory(method, **kwargs):
    def sklearn_method(X, U, reactions, env_class):
        # TODO cluster iteratively, using our tracking to move the points to get fairer baselines
        # simple baselines
        # cluster in [pos, next reaction force, action] space
        xx = np.concatenate([X[:-1, :2], reactions[1:], U[:-1]], axis=1)
        # give it some help by telling it the number of clusters as unique contacts IDs
        res = method(**kwargs).fit(xx)
        return res.labels_, dict_to_namespace_str(kwargs)

    return sklearn_method


def online_sklearn_method_factory(online_class, method, inertia_ratio=0.5, **kwargs):
    def sklearn_method(X, U, reactions, env_class):
        online_method = online_class(method(**kwargs), inertia_ratio=inertia_ratio)
        for i in range(len(X) - 1):
            # intermediate labels in case we want plotting of movement
            labels = online_method.update(X[i], U[i], env_class.state_difference(X[i + 1], X[i]).reshape(-1),
                                          reactions[i + 1])
        return online_method.final_labels(), dict_to_namespace_str(kwargs)

    return sklearn_method


def load_file(datafile, run_res, methods, show_in_place=False):
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
    if NO_CONTACT_ID in unique_contact_counts:
        freespace_ratio = unique_contact_counts[NO_CONTACT_ID] / steps_taken
    else:
        freespace_ratio = 1.
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

    for method_name, method in methods.items():
        labels, param_values = method(X, U, reactions, env_cls)
        record_metric(method_name, contact_id[:-1], labels, level, seed, param_values, run_res)

        f = plot_cluster_res(labels, X[:-1], f"Task {level} {datafile.split('/')[-1]} {method_name}")
        save_and_close_fig(f, f"{method_name} {param_values.replace('.', '_')}")

    # plt.show()

    return X, U, dX, contact_id, obj_poses, reactions


def clustering_metrics(labels_true, labels_pred, beta=1.):
    # beta < 1 means more weight for homogenity
    return metrics.homogeneity_score(labels_true, labels_pred), \
           metrics.completeness_score(labels_true, labels_pred), \
           metrics.v_measure_score(labels_true, labels_pred, beta=beta)


def record_metric(method_name, labels_true, labels_pred, level, seed, param_values, run_res):
    # we care about homogenity more than completeness - multiple objects in a single cluster is more dangerous
    h, c, v = clustering_metrics(labels_true, labels_pred, beta=0.5)
    logger.info(f"{method_name} h {h} c {c} v {v}")
    run_res[(level, seed, method_name, param_values)] = h, c, v
    return h, c, v


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


if __name__ == "__main__":
    fullname = os.path.join(cfg.DATA_DIR, 'contact_res.pkl')
    if os.path.exists(fullname):
        with open(fullname, 'rb') as f:
            runs = pickle.load(f)
            logger.info("loaded runs from %s", fullname)
    else:
        runs = {}

    dirs = ['arm/gripper10', 'arm/gripper11', 'arm/gripper12', 'arm/gripper13']
    methods_to_run = {
        'ours V1': our_method,
        'kmeans': sklearn_method_factory(KMeansWithAutoK),
        'dbscan': sklearn_method_factory(DBSCAN, eps=1.0, min_samples=10),
        'birch': sklearn_method_factory(Birch, n_clusters=None, threshold=1.5),
        'online-kmeans': online_sklearn_method_factory(OnlineSklearnFixedClusters, KMeans, n_clusters=1,
                                                       random_state=0),
        'online-dbscan': online_sklearn_method_factory(OnlineAgglomorativeClustering, DBSCAN, eps=1.0, min_samples=5),
        'online-birch': online_sklearn_method_factory(OnlineAgglomorativeClustering, Birch, n_clusters=None,
                                                      threshold=1.5)
    }

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
                load_file(full_filename, runs, methods_to_run)
            except (RuntimeError, RuntimeWarning) as e:
                logger.info(f"{full_filename} error: {e}")
                continue

    for k, v in runs.items():
        pretty_v = [round(metric, 2) for metric in v]
        logger.info(f"{k} : {pretty_v}")

    # plot results for all methods and runs
    plot_median = True
    f = plt.figure()
    ax = plt.gca()
    ax.set_xlabel('homogenity')
    ax.set_ylabel('completeness')
    ax.set_xlim(0, 1.1)
    ax.set_ylim(0, 1.1)
    for method in methods_to_run.keys():
        this_method_res = [v for k, v in runs.items() if method == k[2]]
        h, c, v = zip(*this_method_res)
        logger.info(f"{method} median {round(np.median(h), 2)} {round(np.median(c), 2)} {round(np.median(v), 2)}")

        if plot_median:
            # scatter for their median
            hm = np.median(h)
            cm = np.median(c)
            ax.errorbar(hm, cm, yerr=[[cm - np.percentile(c, 20)], [np.percentile(c, 80) - cm]],
                        xerr=[[hm - np.percentile(h, 20)], [np.percentile(h, 80) - hm]],
                        label=method, fmt='o')
        else:
            ax.scatter(h, c, alpha=0.4, label=method)

    ax.legend()
    plt.show()

    with open(fullname, 'wb') as f:
        pickle.dump(runs, f)
        logger.info("saved runs to %s", fullname)
