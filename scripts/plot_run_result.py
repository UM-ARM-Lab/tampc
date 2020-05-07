from tensorboard.backend.event_processing import event_accumulator
import re
import os
from meta_contact import cfg
import numpy as np
import matplotlib.pyplot as plt


def batch_size(run):
    b = run[run.index('batch') + len('batch'):]
    return int(re.search(r'\d+', b).group())


runs_basedir = os.path.join(cfg.ROOT_DIR, 'scripts/runs')
POINTS_PER_EPOCH = 50 * 200 * 0.9
MAX_POINTS = 3000
MAX_EPOCH = 6000
name_prefix = 'armconjunction'
largest_epoch_encountered = 0

# scalar name combinations
series = {'sep_dec': {'name': 'w/o $h_\omega$'}, 'extract': {'name': 'w/o REx', },
          'rex_extract': {'name': 'full'}, }
losses = {'percent_match': {'name': 'match', 'pos': 0}, 'percent_reconstruction': {'name': 'reconstruction', 'pos': 1}}
datasets = {'validation': {'name': 'validation', 'pos': 0},
            'validation_50_50': {'name': 'validation (50,50)', 'pos': 1}, 'test0': {'name': 'test0', 'pos': 2}}

# loss_names = ['percent_match', 'percent_reconstruction']
# dataset_names = ['validation', 'validation_50_50', 'test0']
#
# loss_positions = {'match': 0, 'reconstruction': 1}
# dataset_positions = {'validation': 0, 'validation (50,50)': 1, 'test0': 2}

runs = os.listdir(runs_basedir)

for r in runs:
    run_dir = os.path.join(runs_basedir, r)
    run = os.path.join(run_dir, os.listdir(run_dir)[0])
    name = r[r.index(name_prefix) + len(name_prefix):]
    print(name)

    run_series = None
    for s in series.keys():
        if name.startswith(s):
            run_series = s

    if run_series is None:
        print("Ignoring {} since it's not a recognized series".format(name))
        continue

    ea = event_accumulator.EventAccumulator(run, size_guidance={
        event_accumulator.COMPRESSED_HISTOGRAMS: 1,
        event_accumulator.IMAGES: 1,
        event_accumulator.AUDIO: 1,
        event_accumulator.SCALARS: MAX_POINTS,
        event_accumulator.HISTOGRAMS: 1,
    })
    ea.Reload()
    tags = sorted(ea.Tags()['scalars'])

    for loss in losses:
        for dataset in datasets:
            t = (loss, dataset)
            if t not in series[run_series]:
                series[run_series][t] = []

            data = ea.Scalars('{}/{}'.format(*t))

            steps = np.array([d.step for d in data])
            max_epoch = steps[-1] * batch_size(run) // POINTS_PER_EPOCH
            if largest_epoch_encountered < max_epoch:
                largest_epoch_encountered = max_epoch

            values = np.array([d.value for d in data])
            steps_per_epoch = steps[-1] / max_epoch
            epochs = steps / steps_per_epoch
            # TODO filter out points beyond max epoch

            series[run_series][t].append((epochs, values))

f, axes = plt.subplots(len(losses), len(datasets), figsize=(10, 4), constrained_layout=True)
plt.pause(0.1)

for s in series:
    for t in series[s]:
        if type(t) is str:
            continue

        loss, dataset = t
        i = losses[loss]['pos']
        j = datasets[dataset]['pos']
        ax = axes[i, j]

        all = series[s][t]
        # filter out series with insufficient points (need equal number of points for all of them)
        lens = [len(pair[0]) for pair in all]
        all = [pair for pair in all if len(pair[0]) == max(lens)]

        # just use the first series as epoch
        epochs = all[0][0]
        values = [pair[1] for pair in all]

        # average across seeds
        m = np.mean(values, axis=0)
        std = np.std(values, axis=0)
        ax.semilogx(epochs, m, label=series[s]['name'])
        ax.fill_between(epochs, m - std, m + std, alpha=0.3)
        ax.set_xlim(0, largest_epoch_encountered)
        ax.set_ylim(0, 1)

for dataset_pairs in datasets.values():
    axes[0, dataset_pairs['pos']].set_title(dataset_pairs['name'])

axes[0, 0].set_ylabel('match loss')
axes[1, 0].set_ylabel('reconstruction loss')
axes[1, 1].set_xlabel('epochs')
axes[1, 0].legend()

plt.show()
