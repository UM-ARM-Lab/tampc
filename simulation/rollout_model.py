from hybrid_sysid.experiment import preprocess
import sklearn.preprocessing as skpre
import numpy as np
from hybrid_system_with_mixtures.mdn.model import MixtureDensityNetwork
import torch
import matplotlib.pyplot as plt
from hybrid_sysid.draw import highlight_value_ranges
from meta_contact.experiment import interactive_block_pushing as exp
from meta_contact import prior
from meta_contact.controller import controller
import pybullet as p

import logging

from meta_contact.model import make_mdn_model

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format='[%(levelname)s %(asctime)s %(pathname)s:%(lineno)d] %(message)s',
                    datefmt='%m-%d %H:%M:%S')


def plot_comparison(learned_model, X, Y, labels, axis_name, title, plot_states=False, sample=False):
    # freeze model
    for param in learned_model.parameters():
        param.requires_grad = False

    if plot_states:
        state_dim = X.shape[1]
        assert state_dim == len(axis_name)

        fig, axes = plt.subplots(1, state_dim, figsize=(18, 5))
        for i in range(state_dim):
            axes[i].set_xlabel(axis_name[i])
            axes[i].plot(X[:, i].numpy())
            highlight_value_ranges(labels, ax=axes[i], color_map='rr')
        fig.suptitle(title)

    # plot output/prediction (differences)
    output_offset = 2
    output_dim = 3
    output_name = axis_name[output_offset:output_offset+output_dim+1]
    f2, a2 = plt.subplots(1, output_dim, figsize=(18, 5))

    pi, normal = learned_model(X)
    if sample:
        Yhat = MixtureDensityNetwork.sample(pi, normal)
    else:
        Yhat = MixtureDensityNetwork.mean(pi, normal)
        # Yhat = learned_model(X).detach()

    posterior = pi.probs
    modes = np.argmax(posterior, axis=1)

    frames = np.arange(Yhat.shape[0])

    for i in range(output_dim):
        j = i
        a2[i].set_xlabel(output_name[i])
        a2[i].plot(Y[:, j].numpy())
        if sample:
            a2[i].scatter(frames, Yhat[:, j].numpy(), alpha=0.4, color='orange')
        else:
            a2[i].plot(Yhat[:, j].numpy())

        highlight_value_ranges(modes, ax=a2[i], ymin=0.5)
        highlight_value_ranges(labels, ax=a2[i], color_map='rr', ymax=0.5)
    f2.suptitle(title)

    plt.figure()
    components = posterior.shape[1]
    for i in range(components):
        plt.plot(posterior[:, i])
    highlight_value_ranges(modes, ymin=0.5)
    highlight_value_ranges(labels, color_map='rr', ymax=0.5)
    plt.title('{} component posterior'.format(title))


if __name__ == "__main__":
    preprocessor = preprocess.SklearnPreprocessing(skpre.MinMaxScaler())
    # preprocessor = None

    ds = exp.PushDataset(data_dir='pushing', preprocessor=preprocessor, validation_ratio=0.2)

    model = make_mdn_model(num_components=3)
    name = 'combined'
    prior = prior.Prior(model, name, ds, 1e-3, 1e-5)
    # learn prior model on data

    checkpoint = None
    # checkpoint = '/home/zhsh/catkin_ws/src/meta_contact/checkpoints/mdn_compare_standardized_not_affine.3315.tar'
    # checkpoint = '/home/zhsh/catkin_ws/src/meta_contact/checkpoints/mdn_compare_standardized.4845.tar'
    # checkpoint = '/home/zhsh/catkin_ws/src/meta_contact/checkpoints/mdn.5100.tar'
    checkpoint = '/home/zhsh/catkin_ws/src/meta_contact/checkpoints/mdn_quasistatic.2000.tar'
    # load data if we already have some, otherwise train from scratch
    if checkpoint and prior.load(checkpoint):
        logger.info("loaded checkpoint %s", checkpoint)
    else:
        prior.learn_model(100)

    # starting state
    actions = torch.tensor([0.03, 0] * 20)
    N = actions.shape[0] // 2

    ctrl = controller.PreDeterminedController(actions, 2)
    init_block_pos = [0+0.2, 0]
    init_block_yaw = 0
    init_pusher = [-0.1+0.2, 0]
    sim = exp.InteractivePush(ctrl, num_frames=N, mode=p.GUI, plot=False, save=False)
    sim.set_task_config(init_block=init_block_pos, init_yaw=init_block_yaw, init_pusher=init_pusher)
    sim.run(1)

    # compare simulated results to what the model predicts
    start_index = 0
    sample = True
    d = sim._export_data_dict()
    X = d['X']
    X = np.column_stack((X, d['U']))
    Y = d['Y']
    labels = d['contact']

    axis_name = ['x robot (m)', 'y robot (m)', 'x block (m)', 'y block (m)', 'block rotation (rads)', 'dx', 'dy']
    plot_comparison(prior.model, torch.from_numpy(X), torch.from_numpy(Y),
                    torch.from_numpy(labels.astype(int)).flatten(), axis_name,
                    'compared to sim', sample=sample, plot_states=True)

    plt.show()
    input()
