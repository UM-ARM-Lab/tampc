import os

import scipy.io

from arm_pytorch_utilities import preprocess, load_data
import sklearn.preprocessing as skpre
import torch
import matplotlib.pyplot as plt
from meta_contact.env import block_push as exp
from meta_contact import cfg, util
from meta_contact.controller import controller
import pybullet as p

import logging

from meta_contact.dynamics import model
from arm_pytorch_utilities.model import make

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format='[%(levelname)s %(asctime)s %(pathname)s:%(lineno)d] %(message)s',
                    datefmt='%m-%d %H:%M:%S')

if __name__ == "__main__":
    preprocessor = preprocess.SklearnPreprocessing(skpre.MinMaxScaler())
    # preprocessor = None
    config = load_data.DataConfig(predict_difference=True)
    ds = exp.PushDataSource(data_dir='pushing/touching.mat', preprocessor=preprocessor, validation_ratio=0.2,
                            config=config)

    m = model.MDNUser(make.make_sequential_network(config, make.make_mdn_end_block(num_components=3)))
    mw = model.NetworkModelWrapper(m, ds, name='combined')
    # learn prior model on data

    checkpoint = None
    # checkpoint = '/home/zhsh/catkin_ws/src/meta_contact/checkpoints/mdn_compare_standardized_not_affine.3315.tar'
    # checkpoint = '/home/zhsh/catkin_ws/src/meta_contact/checkpoints/mdn_compare_standardized.4845.tar'
    checkpoint = '/Users/johnsonzhong/Research/meta_contact/checkpoints/mdn.1200.tar'
    # checkpoint = '/Users/johnsonzhong/Research/meta_contact/checkpoints/mdn_quasistatic.2800.tar'
    # checkpoint = '/Users/johnsonzhong/Research/meta_contact/checkpoints/mdn_quasistatic_vanilla.2800.tar'
    # load data if we already have some, otherwise train from scratch
    if checkpoint and mw.load(checkpoint):
        logger.info("loaded checkpoint %s", checkpoint)
    else:
        mw.learn_model(100)

    # starting state
    actions = torch.tensor([0.03, 0] * 20)
    N = actions.shape[0] // 2

    ctrl = controller.PreDeterminedController(actions, 2)
    init_block_pos = [0 + 0.2, 0]
    init_block_yaw = 0
    init_pusher = [-0.3 + 0.2, 0]

    seed = 1
    data_file = os.path.join(cfg.DATA_DIR, "pushing", "{}.mat".format(seed))
    if os.path.isfile(data_file):
        d = scipy.io.loadmat(data_file)
    else:
        env = exp.PushAgainstWallEnv(mode=p.GUI, init_block=init_block_pos, init_yaw=init_block_yaw,
                                     init_pusher=init_pusher)
        sim = exp.InteractivePush(env, ctrl, num_frames=N, plot=False, save=True)
        sim.run(seed)

    # compare simulated results to what the model predicts
    start_index = 0
    sample = 5
    ds = exp.PushDataSource(data_dir=data_file, preprocessor=preprocessor, validation_ratio=0.01, config=config)
    X, Y, labels = ds.training_set()
    # labels = torch.from_numpy(d['contact'].astype(int)).flatten()

    axis_name = ['x robot (m)', 'y robot (m)', 'x block (m)', 'y block (m)', 'block rotation (rads)', 'dx', 'dy']
    util.plotter_map[m.__class__](mw.model, X, Y, labels, axis_name,
                                  'compared to sim', sample=sample, plot_states=False)

    plt.show()
    input()
