from hybrid_sysid.experiment import kuka_push, preprocess, example
import sklearn.preprocessing as skpre
import numpy as np
from hybrid_system_with_mixtures.mdn.model import MixtureDensityNetwork
import torch
import matplotlib.pyplot as plt
from hybrid_sysid.draw import highlight_value_ranges
from meta_contact.experiment import interactive_block_pushing as exp
from meta_contact import prior

import logging

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
    output_name = axis_name[output_offset:]
    output_dim = len(output_name)
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


def make_hand_tsf(xu_fit):
    def hand_tsf(xu):
        dd = kuka_push.dist_transform(xu)
        r = xu[:, -2]
        feat = torch.stack((dd, r), dim=1)
        return feat

    # preprocess it to scale different dimensions
    pre = skpre.StandardScaler()
    feat = hand_tsf(xu_fit)
    pre.fit(feat)

    def processed_tsf(xu):
        feat = hand_tsf(xu)
        feat = torch.from_numpy(pre.transform(feat))
        return feat

    return processed_tsf


def make_mdn_model(input_dim=7, output_dim=3, num_components=4, H_units=32):
    layers = []
    for i in range(3):
        in_dim = input_dim if i == 0 else H_units
        out_dim = H_units
        layers.append(torch.nn.Linear(in_dim, out_dim, bias=True))
        layers.append(torch.nn.LeakyReLU())

    layers.append(MixtureDensityNetwork(H_units, output_dim, num_components))

    mdn = torch.nn.Sequential(
        *layers
    ).double()
    return mdn


if __name__ == "__main__":
    preprocessor = preprocess.SklearnPreprocessing(skpre.MinMaxScaler())
    # preprocessor = None

    # ds = exp.PushDataset(data_dir='pushing', preprocessor=preprocessor)
    # compare on trajectory
    ds = exp.PushDataset(data_dir='pushing/4.mat', preprocessor=preprocessor, validation_ratio=0.02)

    model = make_mdn_model()
    name = 'mdn_compare_traj'
    prior = prior.Prior(model, name, ds, 1e-3, 1e-5)
    # learn prior model on data

    checkpoint = None
    # checkpoint = '/home/zhsh/catkin_ws/src/meta_contact/checkpoints/mdn_compare_standardized_not_affine.3315.tar'
    # checkpoint = '/home/zhsh/catkin_ws/src/meta_contact/checkpoints/mdn_compare_standardized.4845.tar'
    checkpoint = '/home/zhsh/catkin_ws/src/meta_contact/checkpoints/mdn.5100.tar'
    # load data if we already have some, otherwise train from scratch
    if checkpoint and prior.load(checkpoint):
        logger.info("loaded checkpoint %s", checkpoint)
    else:
        prior.learn_model(100)

    # TODO use the model for roll outs instead of just 1 step prediction
    start_index = 0
    N = 2000
    sample = True
    X = prior.XUv[start_index:N + start_index]
    Y = prior.Yv[start_index:N + start_index]
    labels = prior.labelsv[start_index:N + start_index]

    axis_name = ['x robot (m)', 'y robot (m)', 'x block (m)', 'y block (m)', 'block rotation (rads)']
    plot_comparison(prior.model, X, Y, labels, axis_name, 'validation', sample=sample)

    X = prior.XU[start_index:N + start_index]
    Y = prior.Y[start_index:N + start_index]
    labels = prior.labels[start_index:N + start_index]

    plot_comparison(prior.model, X, Y, labels, axis_name, 'training', sample=sample)

    plt.show()
    input()
