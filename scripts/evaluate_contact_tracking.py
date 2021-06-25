from tampc import cfg
# from tampc.env import env as env_base
from tampc.env import pybullet_env as env_base
from tampc.env import arm
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

# by convention -1 refers to not in contact
NO_CONTACT_ID = -1

prefix_to_environment = {'arm/gripper': arm.FloatingGripperEnv}


def extract_env_and_level_from_string(string) -> typing.Optional[typing.Tuple[typing.Type[env_base.PybulletEnv], int]]:
    for name, env_cls in prefix_to_environment.items():
        m = re.search(r"{}\d+".format(name), string)
        if m is not None:
            # find level, which is number succeeding this string match
            level = int(m.group()[len(name):])
            return env_cls, level

    return None


def load_file(datafile):
    if not os.path.exists(datafile):
        raise RuntimeError(f"File doesn't exist")

    ret = extract_env_and_level_from_string(datafile)
    if ret is None:
        raise RuntimeError(f"Path not properly formatted to extract environment and level")
    env_cls, level = ret

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

    print(f"{datafile} freespace ratio {freespace_ratio} unique contact IDs {unique_contact_counts}")

    # env = env_cls(environment_level=level, mode=p.GUI)
    # for i, unique_obj_id in enumerate(unique_contact_counts.keys()):
    #     if unique_obj_id == NO_CONTACT_ID:
    #         continue
    #     x = X[contact_id == unique_obj_id]
    #     u = U[contact_id == unique_obj_id]
    #     state_c, action_c = env_base.state_action_color_pairs[(i - 1) % len(env_base.state_action_color_pairs)]
    #     env.visualize_state_actions(str(i), x, u, state_c, action_c, 0.1)
    # env.close()
    # import time
    # time.sleep(1)

    # simple baselines
    # cluster in [pos, next reaction force, action] space
    xx = np.concatenate([X[:-1, :2], reactions[1:], U[:-1]], axis=1)
    # give it some help by telling it the number of clusters as unique contacts IDs
    kmeans = KMeans(n_clusters=len(unique_contact_counts), random_state=0).fit(xx)
    dbscan = DBSCAN(eps=0.5, min_samples=10).fit(xx)

    # we care about homogenity more than completeness - multiple objects in a single cluster is more dangerous
    h, c, v = clustering_metrics(contact_id[:-1], kmeans.labels_, beta=0.5)
    print(f"kmeans h {h} c {c} v {v}")
    h, c, v = clustering_metrics(contact_id[:-1], dbscan.labels_, beta=0.5)
    print(f"dbscan h {h} c {c} v {v}")

    # simplified plot in 2D
    # ground truth
    f = plt.figure()
    f.suptitle(f"Task {level} {datafile.split('/')[-1]} ground truth")
    ax = plt.gca()
    for i, unique_obj_id in enumerate(unique_contact_counts.keys()):
        x = X[contact_id == unique_obj_id]
        pos = x[:, :2]
        ax.scatter(pos[:, 0], pos[:, 1],
                   label='no contact' if unique_obj_id == NO_CONTACT_ID else 'contact {}'.format(unique_obj_id))
    ax.legend()
    set_position_axis_bounds(ax, 0.7)

    plot_sklearn_cluster_res(kmeans, xx, f"Task {level} {datafile.split('/')[-1]} K-means")
    plot_sklearn_cluster_res(dbscan, xx, f"Task {level} {datafile.split('/')[-1]} DBSCAN")

    plt.show()

    return X, U, dX, contact_id, obj_poses, reactions


def clustering_metrics(labels_true, labels_pred, beta=1.):
    # beta < 1 means more weight for homogenity
    return metrics.homogeneity_score(labels_true, labels_pred), \
           metrics.completeness_score(labels_true, labels_pred), \
           metrics.v_measure_score(labels_true, labels_pred, beta=beta)


def plot_sklearn_cluster_res(method, xx, name):
    f = plt.figure()
    f.suptitle(name)
    ax = plt.gca()
    ids, counts = np.unique(method.labels_, return_counts=True)
    sklearn_cluster_counts = dict(zip(ids, counts))
    sklearn_cluster_counts = dict(sorted(sklearn_cluster_counts.items(), key=lambda item: item[1], reverse=True))
    for i, kmeans_cluster in enumerate(sklearn_cluster_counts.keys()):
        x = xx[method.labels_ == kmeans_cluster]
        pos = x[:, :2]
        ax.scatter(pos[:, 0], pos[:, 1], label=str(kmeans_cluster))
    set_position_axis_bounds(ax, 0.7)


def set_position_axis_bounds(ax, bound=0.7):
    ax.set_xlim(-bound, bound)
    ax.set_ylim(-bound, bound)
    ax.set_xlabel('x')
    ax.set_xlabel('y')


data = {}
full_dir = os.path.join(cfg.DATA_DIR, 'arm/gripper10')

files = os.listdir(full_dir)
files = sorted(files)

for file in files:
    full_filename = '{}/{}'.format(full_dir, file)
    # some interesting ones filtered
    if file not in ['16.mat', '18.mat', '22.mat']:
        continue
    if os.path.isdir(full_filename):
        continue
    try:
        data[full_filename] = load_file(full_filename)
    except (RuntimeError, RuntimeWarning) as e:
        print(f"{full_filename} error: {e}")
        continue
