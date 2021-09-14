import os
import pickle
import re
import typing
import logging
from typing import Type

import numpy as np
import pybullet as p
from matplotlib import pyplot as plt
from sklearn import metrics

from stucco import cfg
from stucco.env import arm, pybullet_env as env_base
from stucco.env.pybullet_env import ContactInfo

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


def get_file_metainfo(datafile):
    if not os.path.exists(datafile):
        raise RuntimeError(f"File doesn't exist")

    ret = extract_env_and_level_from_string(datafile)
    if ret is None:
        raise RuntimeError(f"Path not properly formatted to extract environment and level")
    env_cls, level, seed = ret
    return ret


def dict_to_namespace_str(d):
    return str(d).replace(': ', '=').replace('\'', '').strip('{}')


def clustering_metrics(labels_true, labels_pred):
    return metrics.fowlkes_mallows_score(labels_true, labels_pred),


def record_metric(run_key, labels_true, labels_pred, run_res):
    # we care about homogenity more than completeness - multiple objects in a single cluster is more dangerous
    ret = clustering_metrics(labels_true, labels_pred.astype(int))
    logger.info(f"{run_key.method} {ret}")
    run_res[run_key] = ret
    return ret


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
    # runs = {k: v for k, v in runs.items() if k.method != "ours"}
    # with open(fullname, 'wb') as f:
    #     pickle.dump(runs, f)
    return runs


def compute_contact_error(before_moving_pts, moved_pts,
                          # have either env or the env class and object poses
                          env_cls: Type[arm.ArmEnv]=None, level=None, obj_poses=None,
                          env=None,
                          visualize=False, contact_points_instead_of_contact_config=True):
    contact_error = []
    if moved_pts is not None:
        # if we're given an environment, use it directly and assume we want to compare against the current objects
        if env is None:
            # set the gripper away from other objects so that physics don't deform the fingers
            env = env_cls(init=(100, 100), environment_level=level, mode=p.GUI if visualize else p.DIRECT,
                          log_video=True)
            env.extrude_objects_in_z = True
            # to make object IDs consistent (after a reset the object IDs may not be in previously created order)
            env.reset()
            for obj_id, poses in obj_poses.items():
                pos = poses[-1, :3]
                orientation = poses[-1, 3:]
                p.resetBasePositionAndOrientation(obj_id, pos, orientation)

        test_obj_id = env.robot_id
        if contact_points_instead_of_contact_config:
            col_id = p.createCollisionShape(p.GEOM_SPHERE, radius=1e-8)
            vis_id = p.createVisualShape(p.GEOM_SPHERE, radius=0.003, rgbaColor=[0.1, 0.9, 0.3, 0.6])
            test_obj_id = p.createMultiBody(0, col_id, vis_id, basePosition=[0, 0, 0.1])

        if visualize:
            # visualize all the moved points
            state_c, action_c = env_base.state_action_color_pairs[0]
            env.visualize_state_actions("movedpts", moved_pts, None, state_c, action_c, 0.1)
            state_c, action_c = env_base.state_action_color_pairs[1]
            env.visualize_state_actions("premovepts", before_moving_pts, None, state_c, action_c, 0.1)
            env._dd.clear_visualization_after("movedpts", 0)
            env._dd.clear_visualization_after("premovepts", 0)

        for point in moved_pts:
            if contact_points_instead_of_contact_config:
                p.resetBasePositionAndOrientation(test_obj_id, [point[0], point[1], 0.1], [0, 0, 0, 1])
            else:
                env.set_state(np.r_[point, 0, 0])
            p.performCollisionDetection()

            distances = []
            for obj_id in env.movable + env.immovable:
                c = p.getClosestPoints(obj_id, test_obj_id, 100000)

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
        if contact_points_instead_of_contact_config:
            p.removeBody(test_obj_id)
        if env_cls is not None:
            env.close()
    return contact_error


def object_robot_penetration_score(pt_to_config, config, object_transform, model_pts):
    """Compute the penetration between object and robot for a given transform of the object"""
    # transform model points by object transform
    transformed_model_points = model_pts @ object_transform.transpose(-1, -2)
    d = pt_to_config(config[:, :2], transformed_model_points[:, :2])
    d = d.min().item()
    return -d