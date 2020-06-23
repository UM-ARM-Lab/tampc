import enum
import math
import torch
import pybullet as p
import numpy as np
import matplotlib.pyplot as plt
import logging
import os
from datetime import datetime

from arm_pytorch_utilities import rand, load_data
from arm_pytorch_utilities.optim import get_device
from arm_pytorch_utilities import preprocess

from meta_contact import cfg
from meta_contact.env import peg_in_hole
from meta_contact.controller import controller
from meta_contact.transform.peg_in_hole import CoordTransform, translation_generator
from meta_contact.transform import invariant
from meta_contact.dynamics import online_model, model, prior, hybrid_model

from arm_pytorch_utilities.model import make
from meta_contact.controller.online_controller import NominalTrajFrom
from tensorboardX import SummaryWriter

from meta_contact.dynamics.hybrid_model import OnlineAdapt, get_gating
from meta_contact.controller import online_controller
from meta_contact.controller.gating_function import AlwaysSelectNominal

ch = logging.StreamHandler()
fh = logging.FileHandler(os.path.join(cfg.ROOT_DIR, "logs", "{}.log".format(datetime.now())))

logging.basicConfig(level=logging.DEBUG,
                    format='[%(levelname)s %(asctime)s %(pathname)s:%(lineno)d] %(message)s',
                    datefmt='%m-%d %H:%M:%S', handlers=[ch, fh])

logging.getLogger('matplotlib.font_manager').disabled = True

logger = logging.getLogger(__name__)

env_dir = None


# --- SHARED GETTERS
def get_data_dir(level=0):
    return '{}{}.mat'.format(env_dir, level)


def get_env(mode=p.GUI, level=0, log_video=False):
    global env_dir
    init_peg = [-0.2, 0]
    hole_pos = [0.3, 0.3]

    env_opts = {
        'mode': mode,
        'hole': hole_pos,
        'init_peg': init_peg,
        'log_video': log_video,
        'environment_level': level,
    }
    env = peg_in_hole.PegFloatingGripperEnv(**env_opts)
    env_dir = 'peg/floating'
    return env


def get_ds(env, data_dir, **kwargs):
    d = get_device()
    config = load_data.DataConfig(predict_difference=True, predict_all_dims=True, expanded_input=False)
    ds = peg_in_hole.PegInHoleDataSource(env, data_dir=data_dir, config=config, device=d, **kwargs)
    return ds, config


def get_free_space_env_init(seed=1, **kwargs):
    d = get_device()
    env = get_env(kwargs.pop('mode', p.DIRECT), **kwargs)
    ds, config = get_ds(env, get_data_dir(0), validation_ratio=0.1)

    logger.info("initial random seed %d", rand.seed(seed))
    return d, env, config, ds


class UseTsf(enum.Enum):
    NO_TRANSFORM = 0
    COORD = 1
    SEP_DEC = 12
    EXTRACT = 13
    REX_EXTRACT = 14


def get_transform(env, ds, use_tsf):
    # add in invariant transform here
    d = get_device()
    if use_tsf is UseTsf.NO_TRANSFORM:
        return None
    elif use_tsf is UseTsf.COORD:
        return CoordTransform.factory(env, ds)
    else:
        raise RuntimeError("Unrecgonized transform {}".format(use_tsf))


def update_ds_with_transform(env, ds, use_tsf, evaluate_transform=True):
    invariant_tsf = get_transform(env, ds, use_tsf)

    if invariant_tsf:
        # load transform (only 1 function for learning transform reduces potential for different learning params)
        if use_tsf is not UseTsf.COORD and not invariant_tsf.load(invariant_tsf.get_last_checkpoint()):
            raise RuntimeError("Transform {} should be learned before using".format(invariant_tsf.name))

        if evaluate_transform:
            losses = invariant_tsf.evaluate_validation(None)
            logger.info("tsf on validation %s",
                        "  ".join(
                            ["{} {:.5f}".format(name, loss.mean().cpu().item()) if loss is not None else "" for
                             name, loss
                             in zip(invariant_tsf.loss_names(), losses)]))

        # wrap the transform as a data preprocessor
        preprocessor = preprocess.Compose(
            [invariant.InvariantTransformer(invariant_tsf),
             preprocess.PytorchTransformer(preprocess.RobustMinMaxScaler())])
    else:
        preprocessor = no_tsf_preprocessor()
    # update the datasource to use transformed data
    untransformed_config = ds.update_preprocessor(preprocessor)
    return untransformed_config, use_tsf.name, preprocessor


def no_tsf_preprocessor():
    return preprocess.PytorchTransformer(preprocess.RobustMinMaxScaler())


class OfflineDataCollection:
    @staticmethod
    def random_config(env):
        hole = (np.random.random((2,)) - 0.5)
        init_peg = (np.random.random((2,)) - 0.5)
        return hole, init_peg

    @staticmethod
    def freespace(trials=200, trial_length=50, mode=p.DIRECT):
        env = get_env(mode, 0)
        u_min, u_max = env.get_control_bounds()
        ctrl = controller.FullRandomController(env.nu, u_min, u_max)
        # use mode p.GUI to see what the trials look like
        save_dir = '{}{}'.format(env_dir, level)
        sim = peg_in_hole.PegInHole(env, ctrl, num_frames=trial_length, plot=False, save=True,
                                    stop_when_done=False, save_dir=save_dir)
        rand.seed(4)
        # randomly distribute data
        for _ in range(trials):
            seed = rand.seed()
            # start at fixed location
            hole, init_peg = OfflineDataCollection.random_config(env)
            env.set_task_config(hole=hole, init_peg=init_peg)
            ctrl = controller.FullRandomController(env.nu, u_min, u_max)
            sim.ctrl = ctrl
            sim.run(seed)

        if sim.save:
            load_data.merge_data_in_dir(cfg, save_dir, save_dir)
        plt.ioff()
        plt.show()


class PegNetwork(model.NetworkModelWrapper):
    def evaluate_validation(self):
        with torch.no_grad():
            XUv, _, _ = self.ds.original_validation_set()
            # try validation loss outside of our training region (by translating input)
            for dd in translation_generator():
                XU = torch.cat(
                    (XUv[:, :2] + torch.tensor(dd, device=XUv.device, dtype=XUv.dtype),
                     XUv[:, 2:]),
                    dim=1)
                if self.ds.preprocessor is not None:
                    XU = self.ds.preprocessor.transform_x(XU)
                vloss = self.user.compute_validation_loss(XU, self.Yv, self.ds)
                self.writer.add_scalar("loss/validation_{}_{}".format(dd[0], dd[1]), vloss.mean(),
                                       self.step)


class Learn:
    @staticmethod
    def model(use_tsf, seed=1, name="", train_epochs=600, batch_N=500):
        d, env, config, ds = get_free_space_env_init(seed)

        _, tsf_name, _ = update_ds_with_transform(env, ds, use_tsf)
        # tsf_name = "none_at_all"

        mw = PegNetwork(model.DeterministicUser(make.make_sequential_network(config).to(device=d)), ds,
                        name="peg_{}{}_{}".format(tsf_name, name, seed))
        mw.learn_model(train_epochs, batch_N=batch_N)


if __name__ == "__main__":
    level = 0
    ut = UseTsf.COORD

    # OfflineDataCollection.freespace(trials=200, trial_length=50, mode=p.GUI)

    for seed in range(1):
        Learn.model(ut, seed=seed, name="")