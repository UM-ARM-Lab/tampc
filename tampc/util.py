import enum
import abc
import typing

import torch
from arm_pytorch_utilities import draw, preprocess, rand
from arm_pytorch_utilities.model import make
from arm_pytorch_utilities.optim import get_device
from arm_pytorch_utilities.make_data import datasource
from tampc.dynamics import model, prior
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

from tampc.transform import invariant
from tampc.transform.block_push import CoordTransform
from tampc.transform.invariant import translation_generator, LearnedTransform

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
                       success_threshold_c='brown',
                       plot_cumulative_distribution=True,
                       plot_min_distribution=False,
                       plot_min_scatter=True,
                       store_min_up_to_now=False,
                       success_min_dist=None):
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

    legend_props = {'prop': {'size': 8}, 'framealpha': 0.5}
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
                    # strip anything beyond what we expect the data length to be
                    del dhistory[expected_data_len:]
                    min_dist_up_to_now = 100
                    for i, d in enumerate(dhistory):
                        if store_min_up_to_now:
                            if d is None:
                                dhistory[i] = min_dist_up_to_now
                            else:
                                min_dist_up_to_now = min(min_dist_up_to_now, d)
                                dhistory[i] = min(min_dist_up_to_now, d)
                        else:
                            if d is None:
                                dhistory[i] = dhistory[i - 1]

                    # if list is shorter than expected that means it finished so should have lower than success dist
                    if expected_data_len > len(dhistory):
                        dhistory.extend(
                            [min(success_min_dist * 0.8, dhistory[-1])] * (expected_data_len - len(dhistory)))
                        success += 1
                    elif success_min_dist is not None:
                        success += min(dhistory) < success_min_dist
                    mmdist[level][0] = min(min(dhistory), mmdist[level][0])
                    mmdist[level][1] = max(max(dhistory), mmdist[level][1])

                series.append((series_name, tokens, np.stack(dists), success / len(dists)))
                all_series[level] = series

    if plot_min_distribution:
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
                dists = np.min(dists, axis=1)
                plot_info = series_to_plot[series_name]
                logger.info("%s\nsuccess percent %f%% %d trials", series_name, successes * 100, dists.shape[0])
                logger.info("%s with %d runs mean {:.2f} ({:.2f})".format(np.mean(dists) * 10, np.std(dists) * 10),
                            series_name, len(dists))
                c = plot_info['color']
                sns.distplot(dists, ax=ax[j], hist=False, kde=True, color=c,
                             label=plot_info['name'] if 'label' in plot_info else '_nolegend_',
                             bins=np.linspace(mmdist[level][0], mmdist[level][1], 20))
            ax[j].axvline(x=success_min_dist, color=success_threshold_c)
            ax[j].set_xlim(left=0)
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
                logger.info("%s\nsuccess percent %f%% %d trials", series_name, successes * 100, dists.shape[0])

                t = np.arange(dists.shape[1])
                m = np.median(dists, axis=0)
                lower = np.percentile(dists, 20, axis=0)
                upper = np.percentile(dists, 80, axis=0)

                c = plot_info['color']
                ax[j].plot(t, m, color=c, label=plot_info['name'] if 'label' in plot_info else '_nolegend_')
                ax[j].fill_between(t, lower, upper, facecolor=c, alpha=plot_info.get('alpha', 0.2))

            ax[j].legend(**legend_props)
            ax[j].set_xlim(0, max_t)
            ax[j].set_ylim(bottom=0)
            ax[j].hlines(y=success_min_dist, xmin=0, xmax=max_t, colors=success_threshold_c, linestyles='--', lw=2)
            if set_y_label:
                ax[j].set_ylabel('closest ' if store_min_up_to_now else '' + 'dist to goal')
        ax[-1].set_xlabel('control step')
        f.tight_layout(rect=[0, 0.03, 1, 0.95])
    if plot_min_scatter:
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
                logger.info("%s\nsuccess percent %f%% %d trials", series_name, successes * 100, dists.shape[0])

                # only register if we decrease sufficiently from previous min
                n = dists.shape[0]
                t = torch.zeros(n)
                m = torch.ones(n) * 100
                for trial in range(n):
                    for tt in range(dists.shape[1]):
                        d = dists[trial, tt]
                        if d < 0.95 * m[trial]:
                            m[trial] = d
                            t[trial] = tt

                # returns first occurrence if repeated
                c = plot_info['color']
                # ax[j].scatter(t, m, color=c, label=plot_info['name'] if 'label' in plot_info else '_nolegend_')
                tm = t.median()
                mm = m.median()
                ax[j].errorbar(tm, mm, yerr=[[mm-np.percentile(m, 20)], [np.percentile(m, 80)-mm]],
                               xerr=[[tm-np.percentile(t, 20)], [np.percentile(t, 80)-tm]], color=c,
                               label=plot_info['name'] if 'label' in plot_info else '_nolegend_', fmt='o')

            ax[j].legend(**legend_props)
            ax[j].set_xlim(0, max_t)
            ax[j].set_ylim(bottom=0)
            ax[j].hlines(y=success_min_dist, xmin=0, xmax=max_t, colors=success_threshold_c, linestyles='--', lw=2)
            if set_y_label:
                ax[j].set_ylabel('closest dist to goal')
        ax[-1].set_xlabel('control step for closest dist')
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
        m = re.search(r"_\d+", prefix)
        if m is not None:
            level = int(m.group()[1:])
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


def update_ds_with_transform(env, ds, use_tsf, get_pre_invariant_preprocessor, evaluate_transform=True,
                             rep_name=None):
    invariant_tsf = get_transform(env, ds, use_tsf, override_name=rep_name)

    if invariant_tsf:
        # load transform (only 1 function for learning transform reduces potential for different learning params)
        if use_tsf is not UseTsf.COORD and not invariant_tsf.load(invariant_tsf.get_last_checkpoint()):
            raise RuntimeError("Transform {} should be learned before using".format(invariant_tsf.name))

        if evaluate_transform:
            losses = invariant_tsf.evaluate_validation(None)
            logger.info("tsf on validation %s",
                        "  ".join(
                            ["{} {:.5f}".format(name, loss.mean().cpu().item()) if loss is not None else "" for
                             name, loss
                             in zip(invariant_tsf.loss_names(), losses)]))

        components = [get_pre_invariant_preprocessor(use_tsf), invariant.InvariantTransformer(invariant_tsf)]
        if use_tsf not in [UseTsf.SKIP, UseTsf.REX_SKIP]:
            components.append(preprocess.PytorchTransformer(preprocess.RobustMinMaxScaler()))
        preprocessor = preprocess.Compose(components)
    else:
        preprocessor = no_tsf_preprocessor()
    # update the datasource to use transformed data
    untransformed_config = ds.update_preprocessor(preprocessor)
    tsf_name = use_tsf.name
    if rep_name is not None:
        tsf_name = "{}_{}".format(tsf_name, rep_name)
    return untransformed_config, tsf_name, preprocessor


def no_tsf_preprocessor():
    return preprocess.PytorchTransformer(preprocess.RobustMinMaxScaler())


class UseTsf(enum.Enum):
    NO_TRANSFORM = 0
    COORD = 1
    YAW_SELECT = 2
    LINEAR_ENCODER = 3
    DECODER = 4
    DECODER_SINCOS = 5
    # ones that actually work below
    FEEDFORWARD_PART = 10
    DX_TO_V = 11
    SEP_DEC = 12
    EXTRACT = 13
    REX_EXTRACT = 14
    SKIP = 15
    REX_SKIP = 16
    FEEDFORWARD_BASELINE = 17


def get_transform(env, ds, use_tsf, override_name=None):
    # add in invariant transform here
    d = get_device()
    if use_tsf is UseTsf.NO_TRANSFORM:
        return None
    elif use_tsf is UseTsf.COORD:
        return CoordTransform.factory(env, ds)
    elif use_tsf is UseTsf.YAW_SELECT:
        return LearnedTransform.ParameterizeYawSelect(ds, d, name=override_name or "_s2")
    elif use_tsf is UseTsf.LINEAR_ENCODER:
        return LearnedTransform.LinearComboLatentInput(ds, d, name=override_name or "rand_start_s9")
    elif use_tsf is UseTsf.DECODER:
        return LearnedTransform.ParameterizeDecoder(ds, d, name=override_name or "_s9")
    elif use_tsf is UseTsf.DECODER_SINCOS:
        return LearnedTransform.ParameterizeDecoder(ds, d, name=override_name or "sincos_s2", use_sincos_angle=True)
    elif use_tsf is UseTsf.FEEDFORWARD_PART:
        return LearnedTransform.LearnedPartialPassthrough(ds, d, name=override_name or "_s0")
    elif use_tsf is UseTsf.DX_TO_V:
        return LearnedTransform.DxToV(ds, d, name=override_name or "_s0")
    elif use_tsf is UseTsf.SEP_DEC:
        return LearnedTransform.SeparateDecoder(ds, d, name=override_name or "s1")
    elif use_tsf is UseTsf.EXTRACT:
        return LearnedTransform.ExtractState(ds, d, name=override_name or "s1")
    elif use_tsf is UseTsf.REX_EXTRACT:
        return LearnedTransform.RexExtract(ds, d, name=override_name or "s1")
    elif use_tsf is UseTsf.SKIP:
        return LearnedTransform.SkipLatentInput(ds, d, name=override_name or "ral_s1")
    elif use_tsf is UseTsf.REX_SKIP:
        return LearnedTransform.RexSkip(ds, d, name=override_name or "ral_s1")
    elif use_tsf is UseTsf.FEEDFORWARD_BASELINE:
        return LearnedTransform.NoTransform(ds, d, name=override_name)
    else:
        raise RuntimeError("Unrecgonized transform {}".format(use_tsf))


class TranslationNetworkWrapper(model.NetworkModelWrapper):
    """Network wrapper with some special validation evaluation"""

    def evaluate_validation(self):
        with torch.no_grad():
            XUv, _, _ = self.ds.original_validation_set()
            # try validation loss outside of our training region (by translating input)
            for dd in translation_generator():
                XU = torch.cat(
                    (XUv[:, :2] + torch.tensor(dd, device=XUv.device, dtype=XUv.dtype),
                     XUv[:, 2:]),
                    dim=1)
                if self.ds.preprocessor is not None:
                    XU = self.ds.preprocessor.transform_x(XU)
                vloss = self.user.compute_validation_loss(XU, self.Yv, self.ds)
                self.writer.add_scalar("loss/validation_{}_{}".format(dd[0], dd[1]), vloss.mean(),
                                       self.step)


class EnvGetter(abc.ABC):
    env_dir = None

    @classmethod
    def data_dir(cls, level=0) -> str:
        """Return data directory corresponding to an environment level"""
        return '{}{}.mat'.format(cls.env_dir, level)

    @staticmethod
    @abc.abstractmethod
    def ds(env, data_dir, **kwargs) -> datasource.FileDataSource:
        """Return a datasource corresponding to this environment and data directory"""

    @staticmethod
    @abc.abstractmethod
    def pre_invariant_preprocessor(use_tsf: UseTsf) -> preprocess.Transformer:
        """Return preprocessor applied before the invariant transform"""

    @staticmethod
    @abc.abstractmethod
    def controller_options(env) -> typing.Tuple[dict, dict]:
        """Return controller option default values suitable for this environment"""

    @classmethod
    @abc.abstractmethod
    def env(cls, mode, level=0, log_video=False):
        """Create and return an environment; internally should set cls.env_dir"""

    @classmethod
    def free_space_env_init(cls, seed=1, **kwargs):
        d = get_device()
        env = cls.env(kwargs.pop('mode', 0), **kwargs)
        ds = cls.ds(env, cls.data_dir(0), validation_ratio=0.1)

        logger.info("initial random seed %d", rand.seed(seed))
        return d, env, ds.current_config(), ds

    @classmethod
    def prior(cls, env, use_tsf=UseTsf.COORD, prior_class=prior.NNPrior, rep_name=None):
        """Get dynamics prior in transformed space, along with the datasource used for fitting the transform"""
        if use_tsf in [UseTsf.SKIP, UseTsf.REX_SKIP]:
            prior_class = prior.PassthroughLatentDynamicsPrior
        ds = cls.ds(env, cls.data_dir(0), validation_ratio=0.1)
        untransformed_config, tsf_name, preprocessor = update_ds_with_transform(env, ds, use_tsf,
                                                                                cls.pre_invariant_preprocessor,
                                                                                evaluate_transform=False,
                                                                                rep_name=rep_name)
        pm = cls.loaded_prior(prior_class, ds, tsf_name, False)

        return ds, pm

    @staticmethod
    @abc.abstractmethod
    def dynamics_prefix() -> str:
        """Return the prefix of dynamics functions corresponding to this environment"""

    @classmethod
    def loaded_prior(cls, prior_class, ds, tsf_name, relearn_dynamics, seed=0):
        """Directly get loaded dynamics prior, training it if necessary on some datasource"""
        d = get_device()
        if prior_class is prior.NNPrior:
            mw = TranslationNetworkWrapper(
                model.DeterministicUser(make.make_sequential_network(ds.config).to(device=d)),
                ds, name="{}_{}_{}".format(cls.dynamics_prefix(), tsf_name, seed))

            train_epochs = 500
            pm = prior.NNPrior.from_data(mw, checkpoint=None if relearn_dynamics else mw.get_last_checkpoint(
                sort_by_time=False), train_epochs=train_epochs)
        elif prior_class is prior.PassthroughLatentDynamicsPrior:
            pm = prior.PassthroughLatentDynamicsPrior(ds)
        elif prior_class is prior.NoPrior:
            pm = prior.NoPrior()
        else:
            pm = prior_class.from_data(ds)
        return pm

    @classmethod
    def learn_invariant(cls, use_tsf=UseTsf.REX_EXTRACT, seed=1, name="", MAX_EPOCH=1000, BATCH_SIZE=500, resume=False,
                        **kwargs):
        d, env, config, ds = cls.free_space_env_init(seed)

        ds.update_preprocessor(cls.pre_invariant_preprocessor(use_tsf))
        invariant_cls = get_transform(env, ds, use_tsf).__class__
        common_opts = {'name': "{}_s{}".format(name, seed)}
        invariant_tsf = invariant_cls(ds, d, **common_opts, **kwargs)
        if resume:
            invariant_tsf.load(invariant_tsf.get_last_checkpoint())
        invariant_tsf.learn_model(MAX_EPOCH, BATCH_SIZE)

    @classmethod
    def learn_model(cls, use_tsf, seed=1, name="", train_epochs=500, batch_N=500, rep_name=None):
        d, env, config, ds = cls.free_space_env_init(seed)

        _, tsf_name, _ = update_ds_with_transform(env, ds, use_tsf, cls.pre_invariant_preprocessor, rep_name=rep_name)
        mw = TranslationNetworkWrapper(model.DeterministicUser(make.make_sequential_network(config).to(device=d)), ds,
                                       name="{}_{}{}_{}".format(cls.dynamics_prefix(), tsf_name, name, seed))
        mw.learn_model(train_epochs, batch_N=batch_N)
