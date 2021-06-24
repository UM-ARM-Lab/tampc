from tampc import cfg
from tampc.env import env
from tampc.env import arm
import scipy.io
import numpy as np
import os.path
import re
import pybullet as p
import typing

# by convention -1 refers to not in contact
NO_CONTACT_ID = -1

prefix_to_environment = {'arm/gripper': arm.FloatingGripperEnv}


def extract_env_and_level_from_string(string) -> typing.Optional[typing.Tuple[typing.Type[env.Env], int]]:
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

    contact_id = d['contact_id'][mask]
    ids, counts = np.unique(contact_id, return_counts=True)
    unique_contact_counts = dict(zip(ids, counts))
    steps_taken = sum(unique_contact_counts.values())

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
    return X, U, dX, contact_id, obj_poses, reactions


data = {}
full_dir = os.path.join(cfg.DATA_DIR, 'arm/gripper10')

files = os.listdir(full_dir)
files = sorted(files)

for file in files:
    full_filename = '{}/{}'.format(full_dir, file)
    if os.path.isdir(full_filename):
        continue
    try:
        data[full_filename] = load_file(full_filename)
    except (RuntimeError, RuntimeWarning) as e:
        print(f"{full_filename} error: {e}")
        continue
