try:
    import sys

    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except:
    pass

import time
import math
import random
import torch
import pybullet as p
import numpy as np
import matplotlib.pyplot as plt
import logging
import os
from datetime import datetime

from arm_pytorch_utilities import rand, load_data
from arm_pytorch_utilities.optim import get_device
from arm_pytorch_utilities import draw

from tampc import cfg
from cottun import tracking
from tampc.controller import controller
from tampc.transform import invariant
from tampc.dynamics import hybrid_model
from tampc.env import arm
from tampc.env.arm import task_map, Levels
from tampc.env_getters.arm import RetrievalGetter
from tampc.env.pybullet_env import ContactInfo, state_action_color_pairs
from cottun import icp

ch = logging.StreamHandler()
fh = logging.FileHandler(os.path.join(cfg.ROOT_DIR, "logs", "{}.log".format(datetime.now())))

logging.basicConfig(level=logging.DEBUG,
                    format='[%(levelname)s %(asctime)s %(pathname)s:%(lineno)d] %(message)s',
                    datefmt='%m-%d %H:%M:%S', handlers=[ch, fh])

logging.getLogger('matplotlib.font_manager').disabled = True

logger = logging.getLogger(__name__)

force_gui = True
env = RetrievalGetter.env(level=Levels.NO_CLUTTER, mode=p.GUI if force_gui else p.DIRECT)


# sample model points from object
def sample_model_points(object_id, num_points=100, reject_too_close=0.002, force_z=None):
    pos = p.getBasePositionAndOrientation(object_id)[0]
    tester = p.loadURDF(os.path.join(cfg.ROOT_DIR, "tester.urdf"), useFixedBase=False, basePosition=pos,
                        globalScaling=0.3)
    points = []
    sigma = 0.1
    while len(points) < num_points:
        tester_pos = np.add(pos, np.r_[np.random.randn(2) * sigma, 0])
        # sample an object at random points around this object and find closest point to it
        p.resetBasePositionAndOrientation(tester, tester_pos, p.getQuaternionFromEuler([0, 0, 0]))
        p.performCollisionDetection()
        closest = p.getClosestPoints(object_id, tester, 100)
        pt = closest[0][ContactInfo.POS_A]
        if force_z is not None:
            pt = (pt[0], pt[1], force_z)
        if len(points) > 0:
            d = np.subtract(points, pt)
            d = np.linalg.norm(d, axis=1)
            if np.any(d < reject_too_close):
                continue
        points.append(pt)

    p.removeBody(tester)
    points = np.stack(points)
    return points


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

model_points = sample_model_points(env.target_object_id, num_points=50, force_z=z)
for i, pt in enumerate(model_points):
    env._dd.draw_point(f"mpt{i}", pt, color=(0, 0, 1), length=0.003)

# perform ICP and visualize the transformed points
# history, transformed_contact_points = icp.icp(model_points[:, :2], contact_points,
#                                               point_pairs_threshold=len(contact_points), verbose=True)

# better to have few A than few B and then invert the transform
T, distances, i = icp.icp_2(contact_points, model_points[:, :2])
# transformed_contact_points = np.dot(np.c_[contact_points, np.ones((contact_points.shape[0], 1))], T.T)
# T, distances, i = icp.icp_2(model_points[:, :2], contact_points)
transformed_model_points = np.dot(np.c_[model_points[:, :2], np.ones((model_points.shape[0], 1))], np.linalg.inv(T).T)
for i, pt in enumerate(transformed_model_points):
    pt = [pt[0], pt[1], z]
    env._dd.draw_point(f"tmpt{i}", pt, color=(0, 1, 0), length=0.003)

while True:
    env.step([0, 0])
    time.sleep(0.2)
