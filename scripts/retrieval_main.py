try:
    import sys

    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except:
    pass

import time
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
from cottun import contact
from tampc.controller import controller
from tampc.transform import invariant
from tampc.dynamics import hybrid_model
from tampc.env import arm
from tampc.env.arm import task_map, Levels
from tampc.env_getters.arm import RetrievalGetter
from tampc.env.pybullet_env import ContactInfo

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
pts = sample_model_points(env.target_object_id, num_points=100, force_z=z)
for i, pt in enumerate(pts):
    env._dd.draw_point(f"pt{i}", pt, color=(1, 0, 0), length=0.003)

while True:
    env.step([0, 0])
    time.sleep(0.2)
