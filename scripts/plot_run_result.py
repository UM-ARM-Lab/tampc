from tensorboard.backend.event_processing import event_accumulator
import re
import os
from tampc import cfg
import numpy as np
import matplotlib.pyplot as plt
import pickle
import socket


def batch_size(run):
    b = run[run.index('batch') + len('batch'):]
    return int(re.search(r'\d+', b).group())


runs_basedir = os.path.join(cfg.ROOT_DIR, 'scripts/runs')
POINTS_PER_EPOCH = 50 * 200 * 0.9
MAX_POINTS = 3000
MAX_EPOCH = 3000
name_prefix = socket.gethostname()  # 'armconjunction'
largest_epoch_encountered = 0
name_contains = None
ignore_cache = False

ymag = 6.68  # for push validation

# scalar name combinations
series = {
    'rex_extract_2_eval': {'name': 'ours', 'color': 'green'},
    'notransform_eval': {'name': 'feedforward baseline', 'color': 'red'},
}
# losses = {'percent_match': {'name': 'match', 'pos': 0}, 'percent_reconstruction': {'name': 'reconstruction', 'pos': 1}}
losses = {'mse_loss': {'name': 'MSE', 'pos': 0}}
# losses = {'percent_match': {'name': 'MSE', 'pos': 0}}
# datasets = {'validation': {'name': '(a) validation', 'pos': 0},
#             'validation_10_10': {'name': '(b) validation (10,10)', 'pos': 1}, 'test0': {'name': '(c) test', 'pos': 2}}
datasets = {'validation': {'name': '(a) validation', 'pos': 0},
            'validation_10_10': {'name': '(b) validation (10,10)', 'pos': 1}}

runs = os.listdir(runs_basedir)
runs_assignment = {s: [] for s in series.keys()}

for r in runs:
    run_dir = os.path.join(runs_basedir, r)
    run = os.path.join(run_dir, os.listdir(run_dir)[0])
    try:
        name = r[r.index(name_prefix) + len(name_prefix):]
    except ValueError:
        continue

    if name_contains is not None and name_contains not in name:
        print("Ignoring {} since it does not contain {}".format(name, name_contains))
        continue

    run_series = None
    for s in series.keys():
        if name.startswith(s):
            run_series = s
            runs_assignment[s].append(name)

    if run_series is None:
        print("Ignoring {} since it's not a recognized series".format(name))
        continue

    print(name)
    cache = os.path.join(cfg.DATA_DIR, 'run_cache', name + '.pkl')
    # load from cache if possible since it could take a while
    if not ignore_cache and os.path.isfile(cache):
        with open(cache, 'rb') as f:
            to_add = pickle.load(f)
            print('loaded cache for {}'.format(name))
    else:
        ea = event_accumulator.EventAccumulator(run, size_guidance={
            event_accumulator.COMPRESSED_HISTOGRAMS: 1,
            event_accumulator.IMAGES: 1,
            event_accumulator.AUDIO: 1,
            event_accumulator.SCALARS: MAX_POINTS,
            event_accumulator.HISTOGRAMS: 1,
        })
        ea.Reload()
        tags = sorted(ea.Tags()['scalars'])

        to_add = []
        for loss in losses:
            for dataset in datasets:
                t = (loss, dataset)
                try:
                    data = ea.Scalars('{}/{}'.format(*t))

                    steps = np.array([d.step for d in data])
                    max_epoch = steps[-1] * batch_size(run) // POINTS_PER_EPOCH

                    values = np.array([d.value for d in data])
                    if loss == 'mse_loss':
                        values /= ymag
                    steps_per_epoch = steps[-1] / max_epoch
                    epochs = steps / steps_per_epoch
                except KeyError as e:
                    print("-1 padding for missing key: {}".format(e))
                    epochs = np.linspace(0, MAX_EPOCH, MAX_EPOCH)
                    values = np.ones_like(epochs) * -1
                to_add.append((t, epochs, values))

        # save to cache
        if not os.path.exists(os.path.dirname(cache)):
            try:
                os.makedirs(os.path.dirname(cache))
            except OSError as exc:  # Guard against race condition
                import errno

                if exc.errno != errno.EEXIST:
                    raise
        with open(cache, 'wb') as f:
            pickle.dump(to_add, f)
            print('cached {}'.format(name))

    for t, epochs, values in to_add:
        if largest_epoch_encountered < epochs[-1]:
            largest_epoch_encountered = epochs[-1]
        if t not in series[run_series]:
            series[run_series][t] = []
        series[run_series][t].append((epochs, values))

f, axes = plt.subplots(len(losses), len(datasets), figsize=(10, 3), constrained_layout=True)
if len(losses) is 1:
    axes = axes.reshape(1, -1)
plt.pause(0.1)

for s, run_names in runs_assignment.items():
    print('---')
    print(s)
    for n in run_names:
        print(n)

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
        c = color = series[s]['color']
        ax.semilogx(epochs, m, label=series[s]['name'], color=c)
        ax.fill_between(epochs, m - std, m + std, alpha=0.3, color=c)
        ax.set_xlim(left=0, right=3200)
        ax.set_ylim(bottom=0, top=1.5)

for dataset_pairs in datasets.values():
    axes[0, dataset_pairs['pos']].set_title(dataset_pairs['name'])

axes[0, 0].set_ylabel('Relative MSE')
# axes[1, 0].set_ylabel('reconstruction loss')
axes[-1, 0].set_xlabel('epochs')
axes[-1, 1].set_xlabel('epochs')
axes[-1, 0].legend()

plt.show()
