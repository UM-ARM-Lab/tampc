import os
import pickle
import re
import typing
import logging

import numpy as np
from matplotlib import pyplot as plt
from sklearn import metrics
from tampc import cfg
from tampc.env import pybullet_env as env_base, arm

logger = logging.getLogger(__name__)

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


def dict_to_namespace_str(d):
    return str(d).replace(': ', '=').replace('\'', '').strip('{}')


def clustering_metrics(labels_true, labels_pred, beta=1.):
    # beta < 1 means more weight for homogenity
    return metrics.homogeneity_score(labels_true, labels_pred), \
           metrics.completeness_score(labels_true, labels_pred), \
           metrics.v_measure_score(labels_true, labels_pred, beta=beta)


def record_metric(run_key, labels_true, labels_pred, run_res):
    # we care about homogenity more than completeness - multiple objects in a single cluster is more dangerous
    h, c, v = clustering_metrics(labels_true, labels_pred, beta=0.5)
    logger.info(f"{run_key.method} h {h} c {c} v {v}")
    run_res[run_key] = h, c, v
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


def load_runs_results():
    fullname = os.path.join(cfg.DATA_DIR, 'contact_res.pkl')
    if os.path.exists(fullname):
        with open(fullname, 'rb') as f:
            runs = pickle.load(f)
            logger.info("loaded runs from %s", fullname)
    else:
        runs = {}
    return runs
