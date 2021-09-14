# plot results after evaluating contact tracking results and saving to runs file
import matplotlib.pyplot as plt
import logging
import typing
import numpy as np

from stucco.evaluation import load_runs_results
from stucco.defines import RunKey, RUN_INFO_KEYWORDS, RUN_AMBIGUITY, CONTACT_ID, NO_CONTACT_ID

logging.basicConfig(level=logging.INFO,
                    format='[%(levelname)s %(asctime)s %(pathname)s:%(lineno)d] %(message)s',
                    datefmt='%m-%d %H:%M:%S')

logging.getLogger('matplotlib.font_manager').disabled = True

logger = logging.getLogger(__name__)

MAX_PENETRATION = 0.133


class ContactTrackingResultsPlot:
    def __init__(self, runs, methods_to_run, aggregate_method=np.median, filter_on_ambiguity=None, plot_aggregate=False,
                 aggregate_perturbation=0., represent_cme_as_ratio=True, plot_only_best_params_on_metric=None):
        """
        :param runs:
        :param methods_to_run:
        :param aggregate_method:
        :param filter_on_ambiguity: optional callable taking in ambiguity that only selects runs where the callable
        returns true
        :param plot_aggregate:
        :param aggregate_perturbation: may want to be non-0 when duplicate values are often encountered to visually
        distinguish different series, only relevant when plot_aggregate is true
        :param represent_cme_as_ratio: whether the contact manifold error is in absolute units (m) or as relative to the
        max reported penetration
        :param plot_only_best_params_on_metric: if None, plot all param runs, else we expect a string in the form of
        [<>]metric_name tuple for use to sort runs from different param values based on metric mean; < or > prefix
        decides sorting order
        """
        self.runs = runs
        self.methods_to_run = methods_to_run
        self.aggregate_method = aggregate_method
        self.filter_on_ambiguity = filter_on_ambiguity
        self.plot_aggregate = plot_aggregate
        self.represent_cme_as_ratio = represent_cme_as_ratio
        self.plot_only_best_params_on_metric = plot_only_best_params_on_metric

        self.f = plt.figure(figsize=(5,4))
        self.ax = plt.gca()

        self._set_title_and_lims(self.f, self.ax)

        keys = set((k.level, k.seed) for k in self.runs.keys())
        run_mean_ambiguity = {}
        for k in keys:
            contact_id = runs[RunKey(level=k[0], seed=k[1], method=CONTACT_ID, params=None)]
            ambiguity = runs[RunKey(level=k[0], seed=k[1], method=RUN_AMBIGUITY, params=None)]
            ambiguity_when_in_contact = ambiguity[contact_id != NO_CONTACT_ID]
            m = np.mean(ambiguity_when_in_contact).item()
            run_mean_ambiguity[k] = m
            logger.info(f"level {k[0]} seed {k[1]} mean ambiguity in contact {round(m, 2)}")

        for method in self.methods_to_run:
            # check if there are multiple parameters values for this method
            this_method_runs = {k: v for k, v in self.runs.items() if method == k.method}

            # print runs by how problematic they are - allows us to examine specific runs
            sorted_runs = {k: v for k, v in sorted(this_method_runs.items(), key=lambda item: item[1][-1])}
            for k, v in sorted_runs.items():
                am = run_mean_ambiguity[(k.level, k.seed)]
                logger.info(f"{k} : {[round(metric, 2) for metric in v]} ambiguity in contact {am}")

            runs_per_param_value = {}
            for k, v in this_method_runs.items():
                if k.params not in runs_per_param_value:
                    runs_per_param_value[k.params] = []
                am = run_mean_ambiguity[(k.level, k.seed)]
                # filter to only show runs with certain ambiguity properties
                if self.filter_on_ambiguity is not None and not self.filter_on_ambiguity(am):
                    continue
                runs_per_param_value[k.params].append((am, v))

            # unpack values
            temp = {}
            for params, values in runs_per_param_value.items():
                a, metrics = zip(*values)
                # Fowlkes-Mallows index and contact manifold error
                fmi, cme, wcme = zip(*metrics)
                if self.represent_cme_as_ratio:
                    cme = [v / MAX_PENETRATION for v in cme]
                    wcme = [v / MAX_PENETRATION for v in wcme]
                data = {'ambiguity': a, 'fmi': fmi, 'cme': cme, 'wcme': wcme}
                temp[params] = data
            runs_per_param_value = temp

            if self.plot_only_best_params_on_metric is not None:
                items = sorted(runs_per_param_value.items(), reverse=self.plot_only_best_params_on_metric[0] == '>',
                               key=lambda item: np.mean(item[1][self.plot_only_best_params_on_metric[1:]]))
                # keep only the first one
                runs_per_param_value = {items[0][0]: items[0][1]}
                logger.info(
                    f"plot only best param runs on {self.plot_only_best_params_on_metric} which was {method} {items[0][0]}")

            for params, data in runs_per_param_value.items():
                method_label = f"{method} {params}" if len(runs_per_param_value) > 1 else method
                x, y = self._select_x_y(data)

                if self.plot_aggregate:
                    xa = aggregate_method(x)
                    ya = aggregate_method(y)
                    self.ax.errorbar(xa + np.random.randn() * aggregate_perturbation,
                                     ya + np.random.randn() * aggregate_perturbation,
                                     yerr=[[ya - np.percentile(y, 20)], [np.percentile(y, 80) - ya]],
                                     xerr=[[xa - np.percentile(x, 20)], [np.percentile(x, 80) - xa]],
                                     label=method_label, fmt='o')
                else:
                    self.ax.scatter(x, y, alpha=0.4, label=method_label)

                self._log_metric(method_label, "fmi", data['fmi'], 2)
                self._log_metric(method_label, "contact error", data['cme'], 3)
                self._log_metric(method_label, "weighted contact error", data['wcme'], 3)

        self.ax.legend()
        self.f.tight_layout(rect=[0, 0.03, 1, 0.95])

    def _log_metric(self, method_label, metric_name, metric, precision=2):
        logger.info(
            f"{method_label} {len(metric)} runs | {metric_name} {round(self.aggregate_method(metric).item(), precision)} "
            f"median {round(np.median(metric).item(), precision)} "
            f"20th {round(np.percentile(metric, 20), 3)} 80th {round(np.percentile(metric, 80), precision)}")

    def _set_title_and_lims(self, f, ax):
        raise NotImplementedError()

    def _select_x_y(self, data: dict) -> typing.Tuple[np.array, np.array]:
        raise NotImplementedError()


class PlotAmbiguityVsFMI(ContactTrackingResultsPlot):
    def _set_title_and_lims(self, f, ax):
        ax.set_xlabel('run mean ambiguity when in contact')
        ax.set_ylabel('FMI')
        ax.set_xlim(0, 1.1)
        ax.set_ylim(0, 1.1)

    def _select_x_y(self, data) -> typing.Tuple[np.array, np.array]:
        return data['ambiguity'], data['fmi']


class PlotContactErrorVsFMI(ContactTrackingResultsPlot):
    def __init__(self, *args, weight_cme=True, **kwargs):
        self.weight_cme = weight_cme
        super(PlotContactErrorVsFMI, self).__init__(*args, **kwargs)

    def _set_title_and_lims(self, f, ax):
        xlabel = []
        if self.weight_cme:
            xlabel.append('weighted')
        xlabel.append('contact error (cm)')
        if self.represent_cme_as_ratio:
            xlabel.append('(relative to max penetration dist)')
            ax.set_xlim(0, 1)
        else:
            ax.set_xlim(0, 5)
        ax.set_xlabel(' '.join(xlabel))
        ax.set_ylabel('FMI')
        ax.set_ylim(0.5, 1.1)

    def _select_x_y(self, data) -> typing.Tuple[np.array, np.array]:
        x = data['wcme'] if self.weight_cme else np.array(data['cme']) * 100
        return x, data['fmi']


if __name__ == "__main__":
    all_runs = load_runs_results()
    # plot all by default
    all_methods = set([k.method for k in all_runs.keys() if k.method not in RUN_INFO_KEYWORDS])
    logger.info(f"all methods: {all_methods}")
    methods = [
        # "ours UKF convexity merge constraint",
        # "ours UKF all cluster",
        # "ours UKF 0 dyn",
        # "ours soft",
        # "ours soft sq dist",
        # "ours soft sq dist sub tol",
        # "ours UKF",
        # "ours soft cached pts",
        # "ours soft full check",
        # "ours soft replace pts",
        # "ours soft replace",
        # "ours soft good",
        "ours",
        # "ours no replace",
        # "ours full",
        # "ours soft sq dist elim freespace"
        # "ours PF",
        # "kmeans",
        # "birch",
        # "dbscan",
        "online-kmeans",
        "online-dbscan",
        "online-birch"
    ]

    def only_high_ambiguity(a):
        return a > 0.3

    contact_error_vs_fmi = PlotContactErrorVsFMI(all_runs, methods, plot_aggregate=True, weight_cme=False,
                                                 represent_cme_as_ratio=False, aggregate_perturbation=0.00,
                                                 plot_only_best_params_on_metric='<cme',
                                                 filter_on_ambiguity=only_high_ambiguity
                                                 )

    plt.show()
