from hybrid_system_with_mixtures.mdn.model import MixtureDensityNetwork
import torch


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