from arm_pytorch_utilities import preprocess, load_data
import sklearn.preprocessing as skpre
import numpy as np
import matplotlib.pyplot as plt
from arm_pytorch_utilities.draw import plot_mdn_prediction
from meta_contact.experiment import interactive_block_pushing as exp
from meta_contact import model
from arm_pytorch_utilities.model import make

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format='[%(levelname)s %(asctime)s %(pathname)s:%(lineno)d] %(message)s',
                    datefmt='%m-%d %H:%M:%S')

if __name__ == "__main__":
    preprocessor = preprocess.SklearnPreprocessing(skpre.MinMaxScaler())
    # preprocessor = None

    # ds = exp.PushDataset(data_dir='pushing', preprocessor=preprocessor)
    # compare on trajectory
    config = load_data.DataConfig(predict_difference=True)
    ds = exp.PushDataset(data_dir='pushing/touching.mat', preprocessor=preprocessor, validation_ratio=0.2,
                         config=config)

    m = model.MDNUser(make.make_mdn_model(num_components=3))
    name = 'mdn'
    mw = model.NetworkModelWrapper(m, name, ds, 1e-3, 1e-5)
    # learn prior model on data

    checkpoint = None
    # checkpoint = '/home/zhsh/catkin_ws/src/meta_contact/checkpoints/mdn_compare_standardized_not_affine.3315.tar'
    # checkpoint = '/home/zhsh/catkin_ws/src/meta_contact/checkpoints/mdn_compare_standardized.4845.tar'
    # checkpoint = '/home/zhsh/catkin_ws/src/meta_contact/checkpoints/mdn.5100.tar'
    # checkpoint = '/Users/johnsonzhong/Research/meta_contact/checkpoints/mdn_quasistatic_vanilla.2800.tar'
    # checkpoint = '/Users/johnsonzhong/Research/meta_contact/checkpoints/mdn_quasistatic.2800.tar'
    # checkpoint = '/Users/johnsonzhong/Research/meta_contact/checkpoints/mdn_quasistatic_lookahead.2800.tar'
    checkpoint = '/Users/johnsonzhong/Research/meta_contact/checkpoints/mdn.1200.tar'
    # checkpoint = '/Users/johnsonzhong/Research/meta_contact/checkpoints/dummy.2000.tar'
    # load data if we already have some, otherwise train from scratch
    if checkpoint and mw.load(checkpoint):
        logger.info("loaded checkpoint %s", checkpoint)
    else:
        mw.learn_model(300)

    # TODO use the model for roll outs instead of just 1 step prediction
    start_index = 0
    N = 50
    sample = 5
    X = mw.XUv[start_index:N + start_index]
    Y = mw.Yv[start_index:N + start_index]
    labels = mw.labelsv[start_index:N + start_index]

    axis_name = ['x robot (m)', 'y robot (m)', 'x block (m)', 'y block (m)', 'block rotation (rads)', 'dx', 'dy']
    plot_mdn_prediction(mw.model, X, Y, labels, axis_name, 'validation', sample=sample, plot_states=True)

    X = mw.XU[start_index:N + start_index]
    Y = mw.Y[start_index:N + start_index]
    labels = mw.labels[start_index:N + start_index]

    plot_mdn_prediction(mw.model, X, Y, labels, axis_name, 'training', sample=sample)

    plt.show()
    input()
