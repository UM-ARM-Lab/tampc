from arm_pytorch_utilities import draw
from tampc.dynamics import model
from tampc import cfg

import contextlib
import os
import logging
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import re
import time
import argparse

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def modified_environ(*remove, **update):
    """
    Temporarily updates the ``os.environ`` dictionary in-place.

    The ``os.environ`` dictionary is updated in-place so that the modification
    is sure to work in all situations.

    :param remove: Environment variables to remove.
    :param update: Dictionary of environment variables and values to add/update.
    """
    env = os.environ
    update = update or {}
    remove = remove or []

    # List of environment variables being updated or removed.
    stomped = (set(update.keys()) | set(remove)) & set(env.keys())
    # Environment variables and values to restore on exit.
    update_after = {k: env[k] for k in stomped}
    # Environment variables and values to remove on exit.
    remove_after = frozenset(k for k in update if k not in env)

    try:
        env.update(update)
        [env.pop(k, None) for k in remove]
        yield
    finally:
        env.update(update_after)
        [env.pop(k) for k in remove_after]


def name_to_tokens(name):
    tk = {'name': name}
    tokens = name.split('__')
    # legacy fallback
    if len(tokens) < 3:
        pass
    elif len(tokens) < 5:
        tokens = name.split('_')
        # skip prefix
        tokens = tokens[2:]
        if tokens[0] == "NONE":
            tk['adaptation'] = tokens.pop(0)
        else:
            tk['adaptation'] = "{}_{}".format(tokens[0], tokens[1])
            tokens = tokens[2:]
        if tokens[0] in ("RANDOM", "NONE"):
            tk['recovery'] = tokens.pop(0)
        else:
            tk['recovery'] = "{}_{}".format(tokens[0], tokens[1])
            tokens = tokens[2:]
        tk['level'] = int(tokens.pop(0))
        tk['tsf'] = tokens.pop(0)
        tk['reuse'] = tokens.pop(0)
        tk['optimism'] = "ALLTRAP"
        tk['trap_use'] = "NOTRAPCOST"
    else:
        tokens.pop(0)
        tk['adaptation'] = tokens[0]
        tk['recovery'] = tokens[1]
        i = 2
        while True:
            try:
                tk['level'] = int(tokens[i])
                break
            except ValueError:
                i += 1
        tk['tsf'] = tokens[i + 1]
        tk['optimism'] = tokens[i + 2]
        tk['reuse'] = tokens[i + 3]
        if len(tokens) > 7:
            tk['trap_use'] = tokens[i + 4]
        else:
            tk['trap_use'] = "NOTRAPCOST"

    return tk


def plot_task_res_dist(series_to_plot, res_file,
                       task_type='block',
                       task_names=None,
                       max_t=500,
                       expected_data_len=498,
                       figsize=(8, 9),
                       set_y_label=True,
                       plot_cumulative_distribution=True,
                       success_min_dist=None,
                       plot_min_distribution=False):
    fullname = os.path.join(cfg.DATA_DIR, res_file)
    if os.path.exists(fullname):
        with open(fullname, 'rb') as f:
            runs = pickle.load(f)
            logger.info("loaded runs from %s", fullname)
    else:
        raise RuntimeError("missing cached task results file {}".format(fullname))

    tasks = {}
    for prefix, dists in runs.items():
        m = re.search(r"__\d+", prefix)
        if m is not None:
            level = int(m.group()[2:])
        else:
            m = re.search(r"_\d+", prefix)
            if m is not None:
                level = int(m.group()[1:])
            else:
                raise RuntimeError("Prefix has no level information in it")
        if level not in tasks:
            tasks[level] = {}
        if prefix not in tasks[level]:
            tasks[level][prefix] = dists

    all_series = {}
    mmdist = {}
    for level, res in tasks.items():
        mmdist[level] = [100, 0]

        res_list = {k: list(v.values()) for k, v in res.items()}
        series = []

        for series_name in series_to_plot:
            if series_name in res_list:
                tokens = name_to_tokens(series_name)
                dists = res_list[series_name]
                success = 0
                # remove any non-list elements (historical)
                dists = [dlist for dlist in dists if type(dlist) is list]
                # process the dists so they are all valid (replace nones)
                for dhistory in dists:
                    min_dist_up_to_now = 100
                    for i, d in enumerate(dhistory):
                        if d is None:
                            dhistory[i] = min_dist_up_to_now
                        else:
                            min_dist_up_to_now = min(min_dist_up_to_now, d)
                            dhistory[i] = min(min_dist_up_to_now, d)

                    # if list is shorter than expected that means it finished so should have 0 dist
                    if expected_data_len > len(dhistory):
                        dhistory.extend([0] * (expected_data_len - len(dhistory)))
                        success += 1
                    elif success_min_dist is not None:
                        success += min(dhistory) < success_min_dist
                    mmdist[level][0] = min(min(dhistory), mmdist[level][0])
                    mmdist[level][1] = max(max(dhistory), mmdist[level][1])

                series.append((series_name, tokens, np.stack(dists), success / len(dists)))
                all_series[level] = series

    if plot_min_distribution:
        for level, series in all_series.items():
            f, ax = plt.subplots(len(series), 1, figsize=figsize)
            f.suptitle("{} task {}".format(task_type, level))

            for i, data in enumerate(series):
                series_name, tk, dists, successes = data
                dists = np.min(dists, axis=1)
                logger.info("%s with %d runs mean {:.2f} ({:.2f})".format(np.mean(dists) * 10, np.std(dists) * 10),
                            series_name, len(dists))
                sns.distplot(dists, ax=ax[i], hist=True, kde=False,
                             bins=np.linspace(mmdist[level][0], mmdist[level][1], 20))
                ax[i].set_title((tk['adaptation'], tk['recovery'], tk['optimism']))
                ax[i].set_xlim(*mmdist[level])
                ax[i].set_ylim(0, int(0.6 * len(dists)))
            ax[-1].set_xlabel('closest dist to goal [m]')
            f.tight_layout(rect=[0, 0.03, 1, 0.95])
    if plot_cumulative_distribution:
        f, ax = plt.subplots(len(all_series), figsize=figsize)
        if isinstance(ax, plt.Axes):
            ax = [ax]
        for j, (level, series) in enumerate(all_series.items()):
            task_name = "{} task {}".format(task_type, level)
            if task_names is not None:
                task_name = task_names[level]
            ax[j].set_title(task_name)
            for i, data in enumerate(series):
                series_name, tk, dists, successes = data
                plot_info = series_to_plot[series_name]
                logger.info("%s\nsuccess percent %f%%", series_name, successes * 100)

                t = np.arange(dists.shape[1])
                m = np.median(dists, axis=0)
                lower = np.percentile(dists, 20, axis=0)
                upper = np.percentile(dists, 80, axis=0)

                c = plot_info['color']
                ax[j].plot(t, m, color=c, label=plot_info['name'] if 'label' in plot_info else '_nolegend_')
                ax[j].fill_between(t, lower, upper, facecolor=c, alpha=0.25)

            ax[j].legend()
            ax[j].set_xlim(0, max_t)
            ax[j].set_ylim(0, mmdist[level][1] * 1.05)
            if set_y_label:
                ax[j].set_ylabel('closest dist to goal')
        ax[-1].set_xlabel('control step')
        f.tight_layout(rect=[0, 0.03, 1, 0.95])

    plt.show()


class Graph:
    def __init__(self):
        from collections import defaultdict
        self.nodes = set()
        self.edges = defaultdict(list)
        self.distances = {}

    def add_node(self, value):
        self.nodes.add(value)

    def add_edge(self, from_node, to_node, distance):
        self.edges[from_node].append(to_node)
        self.distances[(from_node, to_node)] = distance


def dijsktra(graph, initial):
    visited = {initial: 0}
    path = {}

    nodes = set(graph.nodes)

    while nodes:
        min_node = None
        for node in nodes:
            if node in visited:
                if min_node is None:
                    min_node = node
                elif visited[node] < visited[min_node]:
                    min_node = node

        if min_node is None:
            break

        nodes.remove(min_node)
        current_weight = visited[min_node]

        for edge in graph.edges[min_node]:
            weight = current_weight + graph.distances[(min_node, edge)]
            if edge not in visited or weight < visited[edge]:
                visited[edge] = weight
                path[edge] = min_node

    return visited, path


def closest_distance_to_goal_whole_set(distance_runner, prefix, suffix=".mat", task_type='pushing', **kwargs):
    m = re.search(r"__\d+", prefix)
    if m is not None:
        level = int(m.group()[2:])
    else:
        raise RuntimeError("Prefix has no level information in it")

    fullname = os.path.join(cfg.DATA_DIR, '{}_task_res.pkl'.format(task_type))
    if os.path.exists(fullname):
        with open(fullname, 'rb') as f:
            runs = pickle.load(f)
            logger.info("loaded runs from %s", fullname)
    else:
        runs = {}

    if prefix not in runs:
        runs[prefix] = {}

    trials = [filename for filename in os.listdir(os.path.join(cfg.DATA_DIR, task_type)) if
              filename.startswith(prefix) and filename.endswith(suffix)]
    dists = []
    for i, trial in enumerate(trials):
        d = distance_runner("{}/{}".format(task_type, trial), visualize=i == 0, level=level, **kwargs)
        dists.append(min([dd for dd in d if dd is not None]))
        runs[prefix][trial] = d

    logger.info(dists)
    logger.info("mean {:.2f} std {:.2f} cm".format(np.mean(dists) * 10, np.std(dists) * 10))
    with open(fullname, 'wb') as f:
        pickle.dump(runs, f)
        logger.info("saved runs to %s", fullname)
    time.sleep(0.5)


plotter_map = {model.MDNUser: draw.plot_mdn_prediction, model.DeterministicUser: draw.plot_prediction}


def param_type(s):
    try:
        name, value = s.split('=')
        if '.' in value:
            value = float(value)
        else:
            value = int(value)
        return {name: value}
    except:
        raise argparse.ArgumentTypeError("Parameters must be given as name=scalar space-separated pairs")