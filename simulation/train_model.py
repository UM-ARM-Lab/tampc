import logging

import matplotlib.pyplot as plt
import sklearn.preprocessing as skpre
from arm_pytorch_utilities import preprocess, load_data
from arm_pytorch_utilities.model import make
from meta_contact import util
from meta_contact.dynamics import model
from meta_contact.env import block_push as exp

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
    ds = exp.PushDataSource(exp.PushAgainstWallStickyEnv(), data_dir='pushing/touching.mat', preprocessor=preprocessor,
                            validation_ratio=0.2, config=config)

    # m = model.MDNUser(make.make_sequential_network(config, make.make_mdn_end_block(num_components=3)))
    m = model.DeterministicUser(make.make_sequential_network(config))
    mw = model.NetworkModelWrapper(m, ds, name='weird')
    # learn prior model on data

    # load data if we already have some, otherwise train from scratch
    if not mw.load(mw.get_last_checkpoint()):
        mw.learn_model(50)

    # TODO use the model for roll outs instead of just 1 step prediction
    start_index = 0
    N = 50
    sample = 5
    plotter = util.plotter_map[m.__class__]
    X = mw.XUv[start_index:N + start_index]
    Y = mw.Yv[start_index:N + start_index]
    labels = mw.labelsv[start_index:N + start_index]

    axis_name = ['x robot (m)', 'y robot (m)', 'x block (m)', 'y block (m)', 'block rotation (rads)', 'dx', 'dy']
    plotter(mw.model, X, Y, labels, axis_name, 'validation', sample=sample)

    X = mw.XU[start_index:N + start_index]
    Y = mw.Y[start_index:N + start_index]
    labels = mw.labels[start_index:N + start_index]

    plotter(mw.model, X, Y, labels, axis_name, 'training', sample=sample)

    plt.show()
    input()
