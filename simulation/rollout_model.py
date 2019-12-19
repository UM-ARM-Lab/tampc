import os
import scipy.io

from hybrid_sysid.experiment import preprocess
import sklearn.preprocessing as skpre
import numpy as np
import torch
import matplotlib.pyplot as plt
from arm_pytorch_utilities.draw import plot_mdn_prediction
from meta_contact.experiment import interactive_block_pushing as exp
from meta_contact import prior, cfg
from meta_contact.controller import controller
import pybullet as p

import logging

from meta_contact.model import make_mdn_model

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format='[%(levelname)s %(asctime)s %(pathname)s:%(lineno)d] %(message)s',
                    datefmt='%m-%d %H:%M:%S')

if __name__ == "__main__":
    preprocessor = preprocess.SklearnPreprocessing(skpre.MinMaxScaler())
    # preprocessor = None

    ds = exp.PushDataset(data_dir='pushing/touching.mat', preprocessor=preprocessor, validation_ratio=0.2)

    model = make_mdn_model(num_components=3)
    name = 'combined'
    prior = prior.Prior(model, name, ds, 1e-3, 1e-5)
    # learn prior model on data

    checkpoint = None
    # checkpoint = '/home/zhsh/catkin_ws/src/meta_contact/checkpoints/mdn_compare_standardized_not_affine.3315.tar'
    # checkpoint = '/home/zhsh/catkin_ws/src/meta_contact/checkpoints/mdn_compare_standardized.4845.tar'
    # checkpoint = '/home/zhsh/catkin_ws/src/meta_contact/checkpoints/mdn.5100.tar'
    checkpoint = '/Users/johnsonzhong/Research/meta_contact/checkpoints/mdn_quasistatic.2800.tar'
    # checkpoint = '/Users/johnsonzhong/Research/meta_contact/checkpoints/mdn_quasistatic_vanilla.2800.tar'
    # load data if we already have some, otherwise train from scratch
    if checkpoint and prior.load(checkpoint):
        logger.info("loaded checkpoint %s", checkpoint)
    else:
        prior.learn_model(100)

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
        sim = exp.InteractivePush(ctrl, num_frames=N, mode=p.GUI, plot=False, save=True)
        sim.set_task_config(init_block=init_block_pos, init_yaw=init_block_yaw, init_pusher=init_pusher)
        sim.run(seed)
        d = sim._export_data_dict()

    # compare simulated results to what the model predicts
    start_index = 0
    sample = False
    X = torch.from_numpy(np.column_stack((d['X'], d['U'])))
    Y = torch.from_numpy(d['Y'])
    labels = torch.from_numpy(d['contact'].astype(int)).flatten()

    if preprocessor:
        X, Y = preprocessor.transform_x(X), preprocessor.transform_y(Y)

    axis_name = ['x robot (m)', 'y robot (m)', 'x block (m)', 'y block (m)', 'block rotation (rads)', 'dx', 'dy']
    plot_mdn_prediction(prior.model, X, Y, labels, axis_name,
                        'compared to sim', sample=sample, plot_states=False)

    plt.show()
    input()
