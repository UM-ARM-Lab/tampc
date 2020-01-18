import abc
import random
import time
from datetime import datetime

import pybullet as p
import pybullet_data
import logging

logger = logging.getLogger(__name__)


class Mode:
    DIRECT = 0
    GUI = 1


class MyPybulletEnv:
    def __init__(self, mode=Mode.DIRECT, log_video=False):
        self.log_video = log_video
        self.mode = mode
        self.realtime = False
        self.sim_step_s = 1. / 240.
        self.randseed = None
        self._configure_physics_engine()

    def _configure_physics_engine(self):
        mode_dict = {Mode.GUI: p.GUI, Mode.DIRECT: p.DIRECT}

        # if the mode we gave is in the dict then use it, otherwise use the given mode value as is
        mode = mode_dict.get(self.mode) or self.mode

        self.physics_client = p.connect(mode)  # p.GUI for GUI or p.DIRECT for non-graphical version

        if self.log_video:
            p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4,
                                "{}_{}.mp4".format(datetime.now().strftime('%Y_%m_%d_%H_%M_%S'), self.randseed))

        # use data provided by PyBullet
        p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally

        # TODO not sure if I have to set timestep also for real time simulation; I think not
        if self.realtime:
            p.setRealTimeSimulation(True)
        else:
            p.setRealTimeSimulation(False)
            p.setTimeStep(self.sim_step_s)

    def seed(self, randseed=None):
        random.seed(time.time())
        if randseed is None:
            randseed = random.randint(0, 1000000)
        logger.info('random seed: %d', randseed)
        self.randseed = randseed
        random.seed(randseed)
        # potentially also randomize the starting configuration

    def close(self):
        p.disconnect(self.physics_client)

    @staticmethod
    @abc.abstractmethod
    def state_names():
        """Get list of names, one for each state corresponding to the index"""
        return []
