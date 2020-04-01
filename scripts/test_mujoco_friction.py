#!/usr/bin/env python3
"""
Example of how bodies interact with each other. For a body to be able to
move it needs to have joints. In this example, the "robot" is a red ball
with X and Y slide joints (and a Z slide joint that isn't controlled).
On the floor, there's a cylinder with X and Y slide joints, so it can
be pushed around with the robot. There's also a box without joints. Since
the box doesn't have joints, it's fixed and can't be pushed around.
"""

import logging
import math

import numpy as np
from matplotlib import pyplot as plt
from mujoco_py import load_model_from_xml, MjSim, MjViewer
from arm_pytorch_utilities import math_utils

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG,
                    format='[%(levelname)s %(asctime)s %(pathname)s:%(lineno)d] %(message)s',
                    datefmt='%m-%d %H:%M:%S')
logging.getLogger('matplotlib.font_manager').disabled = True

MODEL_XML = """
<?xml version="1.0" ?>
<mujoco>
    <option timestep="0.005" />
    <asset>
        <texture name="texplane" type="2d" builtin="checker" rgb1=".2 .3 .4" rgb2=".1 0.15 0.2" 
        width="512" height="512"/>  
        <material name='MatPlane' reflectance='0.5' texture="texplane" texrepeat="1 1" texuniform="true"/>
    </asset>
    <worldbody>
        <body name="robot" pos="0 0.1 0.025">
            <joint axis="1 0 0" damping="0.1" name="slide0" pos="0 0 0" type="slide"/>
            <joint axis="0 1 0" damping="0.1" name="slide1" pos="0 0 0" type="slide"/>
            <joint axis="0 0 1" damping="1" name="rot1" type="hinge"/>
            <joint axis="0 0 1" damping="1" name="slide2" pos="0 0 0" type="slide"/>
            <geom mass="1" size="0.15 0.15 0.05" type="box"/>
			<camera euler="0 0 0" fovy="40" name="rgb" pos="0 0 2.5"></camera>
        </body>
        <body name="floor" pos="0 0 -0.5">
           <geom name='arena' mass='1' size='0.5 0.5' material="MatPlane" type='cylinder' />
        </body>
       <geom name='floor' pos='0 0 0' size='0 0 .125' type='plane' material="MatPlane" condim='3'/>

    </worldbody>
    <actuator>
        <motor gear="1.0" joint="rot1"/>
    </actuator>
</mujoco>
"""


def rotate_wrt_origin(xy, theta):
    return (xy[0] * math.cos(theta) - xy[1] * math.sin(theta),
            xy[0] * math.sin(theta) + xy[1] * math.cos(theta))


def angular_diff(a, b):
    """Angle difference from b to a (a - b)"""
    d = a - b
    if d > math.pi:
        d -= 2 * math.pi
    elif d < -math.pi:
        d += 2 * math.pi
    return d


def get_dx(px, cx):
    dpos = cx[:2] - px[:2]
    dyaw = angular_diff(cx[2], px[2])
    dx = np.r_[dpos, dyaw]
    return dx


def dx_to_dz(px, dx):
    dz = np.zeros_like(dx)
    # dyaw is the same
    dz[2] = dx[2]
    dz[:2] = rotate_wrt_origin(dx[:2], -px[2])
    return dz


def _observe_block():
    x, y, yaw, z = sim.data.qpos
    return np.array((x, y, math_utils.angle_normalize(yaw)))


STATIC_VELOCITY_THRESHOLD = 1e-6


def _static_environment():
    v = sim.data.qvel
    if (np.linalg.norm(v) > STATIC_VELOCITY_THRESHOLD):
        return False
    return True


model = load_model_from_xml(MODEL_XML)
sim = MjSim(model)
viewer = MjViewer(sim)
t = 0

# get it lowered to the ground
for _ in range(40):
    sim.step()

N = 1500
yaws = np.zeros(N)
z_os = np.zeros((N, 3))

while not _static_environment():
    for _ in range(50):
        sim.step()
        viewer.render()
for simTime in range(N):
    px = _observe_block()
    yaws[simTime] = px[2]
    # apply control
    sim.data.ctrl[0] = 6
    # wait for quasi-staticness
    sim.step()
    sim.data.ctrl[0] = 0
    while not _static_environment():
        for _ in range(50):
            sim.step()
    viewer.render()
    cx = _observe_block()
    # difference in world frame
    dx = get_dx(px, cx)
    dz = dx_to_dz(px, dx)
    z_os[simTime] = dz
    logger.info("dx %s dz %s", dx, dz)

logger.info(z_os.std(0) / np.abs(np.mean(z_os, 0)))

plt.subplot(3, 1, 1)
v = z_os[:, 2]
plt.scatter(yaws, v)
plt.ylabel('dyaw')
plt.ylim(np.min(v), np.max(v))
plt.subplot(3, 1, 2)
v = z_os[:, 0]
plt.scatter(yaws, v)
plt.ylabel('dx_body')
plt.ylim(np.min(v), np.max(v))
plt.subplot(3, 1, 3)
v = z_os[:, 1]
plt.scatter(yaws, v)
plt.xlabel('yaw')
plt.ylabel('dy_body')
plt.ylim(np.min(v), np.max(v))
plt.show()
