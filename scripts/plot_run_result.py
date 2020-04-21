import matplotlib.pyplot as plt
from meta_contact import cfg
import pickle
import logging
import os

import torch
import pandas as pd

total_epoch = 3000

basedir = '/home/zhsh/Documents/results/quals/learning results/'
series = ['rex', 'extract', 'no extractor', 'feedforward']

series_names = {series[0]: 'full', series[1]: 'w/o REx', series[2]: 'w/o $h_\omega$', series[3]: 'feedforward'}
loss_positions = {'match': 0, 'reconstruction': 1}
dataset_positions = {'validation': 0, 'validation (50,50)': 1, 'test0': 2}
to_plot = []

for s in series:
    series_dir = os.path.join(basedir, s)
    files = os.listdir(series_dir)
    for f in files:
        tag = f[f.index('percent_'):]
        tag = tag.split('.')[0]
        tokens = tag.split('_')

        loss = tokens[1]
        dataset = tokens[2]
        if len(tokens) > 3:
            dataset = "{} ({})".format(dataset, ','.join(tokens[3:]))

        print("{} {} {}".format(s, loss, dataset))
        if loss not in loss_positions or dataset not in dataset_positions:
            raise RuntimeError("Unexpected {} loss {} or dataset {}".format(s, loss, dataset))

        data = torch.tensor(pd.read_csv(os.path.join(series_dir, f)).values)
        steps = data[:, 1]
        values = data[:, 2]
        steps_per_epoch = steps[-1] / total_epoch
        epochs = steps / steps_per_epoch

        to_plot.append([s, loss, dataset, epochs, values])

f, axes = plt.subplots(len(loss_positions), len(dataset_positions), figsize=(10, 4), constrained_layout=True)
plt.pause(0.1)

for s, loss, dataset, epochs, values in to_plot:
    i = loss_positions[loss]
    j = dataset_positions[dataset]
    ax = axes[i, j]
    ax.semilogx(epochs, values, label=series_names[s])
    ax.set_xlim(0, total_epoch)
    ax.set_ylim(0, 1)

for dataset, pos in dataset_positions.items():
    axes[0, pos].set_title(dataset)

axes[0, 0].set_ylabel('match loss')
axes[1, 0].set_ylabel('reconstruction loss')
axes[1, 1].set_xlabel('epochs')
axes[1, 0].legend()

plt.show()
