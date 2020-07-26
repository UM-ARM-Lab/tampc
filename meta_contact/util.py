from arm_pytorch_utilities import draw
from meta_contact.dynamics import model
from meta_contact import cfg

import contextlib
import os
import logging
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import re

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
    if len(tokens) < 5:
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
        tk['level'] = int(tokens[2])
        tk['tsf'] = tokens[3]
        tk['optimism'] = tokens[4]
        tk['reuse'] = tokens[5]
        if len(tokens) > 7:
            tk['trap_use'] = tokens[7]
        else:
            tk['trap_use'] = "NOTRAPCOST"

    return tk


def plot_task_res_dist(series_to_plot, res_file,
                       task_type='block',
                       max_t=500,
                       expected_data_len=498,
                       figsize=(8, 9),
                       plot_cumulative_distribution=True,
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
        m = re.search(r"\d+", prefix)
        if m is not None:
            level = int(m.group())
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
                    dhistory.extend([0] * (expected_data_len - len(dhistory)))
                    mmdist[level][0] = min(min(dhistory), mmdist[level][0])
                    mmdist[level][1] = max(max(dhistory), mmdist[level][1])

                series.append((series_name, tokens, np.stack(dists)))
                all_series[level] = series

    if plot_min_distribution:
        for level, series in all_series.items():
            f, ax = plt.subplots(len(series), 1, figsize=figsize)
            f.suptitle("{} task {}".format(task_type, level))

            for i, data in enumerate(series):
                series_name, tk, dists = data
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
        if type(ax) is plt.Axes:
            ax = [ax]
        for j, (level, series) in enumerate(all_series.items()):
            ax[j].set_title("{} task {}".format(task_type, level))
            for i, data in enumerate(series):
                series_name, tk, dists = data
                plot_info = series_to_plot[series_name]

                t = np.arange(dists.shape[1])
                m = np.median(dists, axis=0)
                lower = np.percentile(dists, 20, axis=0)
                upper = np.percentile(dists, 80, axis=0)

                c = plot_info['color']
                ax[j].plot(t, m, color=c, label=plot_info['name'])
                ax[j].fill_between(t, lower, upper, facecolor=c, alpha=0.3)

            ax[j].legend()
            ax[j].set_xlim(0, max_t)
            ax[j].set_ylim(0, mmdist[level][1] * 1.05)
            ax[j].set_ylabel('closest dist to goal')
            ax[j].set_xlabel('control step')
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


plotter_map = {model.MDNUser: draw.plot_mdn_prediction, model.DeterministicUser: draw.plot_prediction}
