from hybrid_sysid.experiment import preprocess
import sklearn.preprocessing as skpre
import numpy as np
import torch
import matplotlib.pyplot as plt
from arm_pytorch_utilities.draw import plot_mdn_prediction
from meta_contact.experiment import interactive_block_pushing as exp
from meta_contact import prior

import logging

from meta_contact.model import make_mdn_model

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format='[%(levelname)s %(asctime)s %(pathname)s:%(lineno)d] %(message)s',
                    datefmt='%m-%d %H:%M:%S')

if __name__ == "__main__":
    preprocessor = preprocess.SklearnPreprocessing(skpre.MinMaxScaler())
    # preprocessor = None

    # ds = exp.PushDataset(data_dir='pushing', preprocessor=preprocessor)
    # compare on trajectory
    ds = exp.PushDataset(data_dir='pushing/touching.mat', preprocessor=preprocessor, validation_ratio=0.2)

    model = make_mdn_model(num_components=3)
    name = 'mdn_quasistatic'
    prior = prior.Prior(model, name, ds, 1e-3, 1e-5)
    # learn prior model on data

    checkpoint = None
    # checkpoint = '/home/zhsh/catkin_ws/src/meta_contact/checkpoints/mdn_compare_standardized_not_affine.3315.tar'
    # checkpoint = '/home/zhsh/catkin_ws/src/meta_contact/checkpoints/mdn_compare_standardized.4845.tar'
    # checkpoint = '/home/zhsh/catkin_ws/src/meta_contact/checkpoints/mdn.5100.tar'
    checkpoint = '/Users/johnsonzhong/Research/meta_contact/checkpoints/mdn_quasistatic.2800.tar'
    # checkpoint = '/home/zhsh/catkin_ws/src/meta_contact/checkpoints/mdn_quasistatic_vanilla.2000.tar'
    # load data if we already have some, otherwise train from scratch
    if checkpoint and prior.load(checkpoint):
        logger.info("loaded checkpoint %s", checkpoint)
    else:
        prior.learn_model(100)

    # TODO use the model for roll outs instead of just 1 step prediction
    start_index = 0
    N = 300
    sample = True
    X = prior.XUv[start_index:N + start_index]
    Y = prior.Yv[start_index:N + start_index]
    labels = prior.labelsv[start_index:N + start_index]

    axis_name = ['x robot (m)', 'y robot (m)', 'x block (m)', 'y block (m)', 'block rotation (rads)']
    plot_mdn_prediction(prior.model, X, Y, labels, axis_name, 'validation', sample=sample)

    X = prior.XU[start_index:N + start_index]
    Y = prior.Y[start_index:N + start_index]
    labels = prior.labels[start_index:N + start_index]

    plot_mdn_prediction(prior.model, X, Y, labels, axis_name, 'training', sample=sample)

    plt.show()
    input()
