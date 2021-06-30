# plot results after evaluating contact tracking results and saving to runs file
import matplotlib.pyplot as plt
import logging
import numpy as np

from cottun.script_utils import load_runs_results
from cottun.defines import RunKey

logging.basicConfig(level=logging.INFO,
                    format='[%(levelname)s %(asctime)s %(pathname)s:%(lineno)d] %(message)s',
                    datefmt='%m-%d %H:%M:%S')

logging.getLogger('matplotlib.font_manager').disabled = True

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    runs = load_runs_results()

    # plot all by default
    all_methods = set([k.method for k in runs.keys()])
    logger.info(f"all methods: {all_methods}")
    methods_to_run = ["ours UKF"]

    # plot results for all methods and runs
    plot_median = True
    f = plt.figure()
    ax = plt.gca()
    ax.set_xlabel('homogenity')
    ax.set_ylabel('completeness')
    ax.set_xlim(0, 1.1)
    ax.set_ylim(0, 1.1)
    for method in methods_to_run:
        # check if there are multiple parameters values for this method
        this_method_runs = {k: v for k, v in runs.items() if method == k.method}
        runs_per_param_value = {}
        for k, v in this_method_runs.items():
            param_value = k.params
            if k.params not in runs_per_param_value:
                runs_per_param_value[k.params] = set()
            runs_per_param_value[k.params].add(v)

        for params, values in runs_per_param_value.items():
            h, c, v = zip(*values)
            method_label = f"{method} {params}" if len(runs_per_param_value) > 1 else method

            logger.info(
                f"{method_label} median {round(np.median(h), 2)} {round(np.median(c), 2)} {round(np.median(v), 2)}")
            if plot_median:
                # scatter for their median
                hm = np.median(h)
                cm = np.median(c)
                ax.errorbar(hm, cm, yerr=[[cm - np.percentile(c, 20)], [np.percentile(c, 80) - cm]],
                            xerr=[[hm - np.percentile(h, 20)], [np.percentile(h, 80) - hm]],
                            label=method_label, fmt='o')
            else:
                ax.scatter(h, c, alpha=0.4, label=method_label)

    ax.legend()
    plt.show()
