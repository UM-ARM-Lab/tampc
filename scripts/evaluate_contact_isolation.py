import logging
from datetime import datetime
import scipy.io
import numpy as np
import os.path
import pybullet as p
import re

from stucco.defines import NO_CONTACT_ID
from stucco.evaluation import get_file_metainfo

from stucco import cfg
from stucco.env.env import InfoKeys

ch = logging.StreamHandler()
fh = logging.FileHandler(os.path.join(cfg.ROOT_DIR, "logs", "{}.log".format(datetime.now())))

logging.basicConfig(level=logging.INFO,
                    format='[%(levelname)s %(asctime)s %(pathname)s:%(lineno)d] %(message)s',
                    datefmt='%m-%d %H:%M:%S', handlers=[ch, fh])

logging.getLogger('matplotlib.font_manager').disabled = True

logger = logging.getLogger(__name__)


def evaluate_on_file(datafile, show_in_place=False, log_video=False):
    try:
        env_cls, level, seed = get_file_metainfo(datafile)
    except (RuntimeError, RuntimeWarning) as e:
        logger.info(f"{full_filename} error: {e}")
        return None

    # load data
    d = scipy.io.loadmat(datafile)
    # use environment specific state difference function since not all states are R^n
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

    env = env_cls(environment_level=level, mode=p.GUI if show_in_place else p.DIRECT, log_video=log_video)
    # get same object ids as stored
    env.reset()

    all_dists = []
    # close gripper
    for i in [0, 1]:
        p.resetJointState(env.robot_id, i, 0)
    # go through previous data and see where the contact detector will isolate the contact to be
    for i in range(len(X)):
        if contact_id[i] == NO_CONTACT_ID:
            env.contact_detector.observation_history.clear()
            continue
        env.set_state(X[i], U[i])
        for obj_id, poses in obj_poses.items():
            pos = poses[i, :3]
            orientation = poses[i, 3:]
            p.resetBasePositionAndOrientation(obj_id, pos, orientation)
        p.performCollisionDetection()
        # env.draw_user_text("{}".format(i), xy=(0.5, 0.7, -1))

        dist_from_actual_contact = []
        r = d['r'][i]
        t = d['t'][i]
        po = d['p'][i]
        c = d[InfoKeys.HIGH_FREQ_CONTACT_POINT][i]
        # feed contact detector
        in_contact = False
        for j in range(len(r)):
            residual = np.r_[r[j], t[j]]
            pose = po[j][:3], po[j][3:]
            this_contact = env.contact_detector.observe_residual(residual, pose=pose)
            in_contact = in_contact or this_contact
            if this_contact:
                pt = env.contact_detector.get_last_contact_location(pose)
                if show_in_place:
                    env._dd.draw_point(f'c', pt, height=pt[2] + 0.001, color=(0, 0, 1))
                    env._dd.draw_point(f'true_c', c[j], height=pt[2] + 0.001, color=(0.7, 0.7, 0))
                    env._draw_reaction_force(r[j], InfoKeys.HIGH_FREQ_REACTION_F, (1, 0, 1))

                dist_from_actual_contact.append(np.linalg.norm(pt.cpu().numpy() - c[j]))
        assert in_contact
        all_dists.append(dist_from_actual_contact)

    env.close()
    return all_dists


if __name__ == "__main__":
    dirs = ['arm/gripper10', 'arm/gripper11', 'arm/gripper12', 'arm/gripper13']

    full_filename = os.path.join(cfg.DATA_DIR, 'arm/gripper10/14.mat')
    dists = evaluate_on_file(full_filename, show_in_place=True, log_video=True)
    per_step_dists = [np.mean(d) for d in dists]

    logger.info(
        f"{full_filename}\navg: {np.mean(per_step_dists)}\nmedian: {np.median(per_step_dists)}\n"
        f"max: {np.max(per_step_dists)} ({np.argmax(per_step_dists)})")
    exit(0)
    res = {}
    for res_dir in dirs:
        full_dir = os.path.join(cfg.DATA_DIR, res_dir)

        files = os.listdir(full_dir)
        files = sorted(files)

        for file in files:
            full_filename = '{}/{}'.format(full_dir, file)
            if os.path.isdir(full_filename):
                continue
            dists = evaluate_on_file(full_filename)
            if dists is None:
                continue
            per_step_dists = [np.mean(d) for d in dists]
            res[full_filename] = (np.mean(per_step_dists), np.median(per_step_dists), np.max(per_step_dists))

            logger.info(
                f"{full_filename}\navg: {res[full_filename][0]}\nmedian: {res[full_filename][1]}\n"
                f"max: {res[full_filename][2]} ({np.argmax(per_step_dists)})")

    sorted_runs = {k: v for k, v in sorted(res.items(), key=lambda item: item[1][0])}
    for k, v in sorted_runs.items():
        logger.info(f"{k} : {[round(metric, 2) for metric in v]}")
    logger.info(f"all : {[np.round(np.mean(metric), 2) for metric in zip(*sorted_runs.values())]}")
