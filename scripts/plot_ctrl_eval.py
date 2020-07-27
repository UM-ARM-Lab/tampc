import torch
import matplotlib.pyplot as plt
from arm_pytorch_utilities.draw import cumulative_dist
from tampc import cfg
import pickle
import logging
import os

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG,
                    format='[%(levelname)s %(asctime)s %(pathname)s:%(lineno)d] %(message)s',
                    datefmt='%m-%d %H:%M:%S')
logging.getLogger('matplotlib.font_manager').disabled = True

fullname = os.path.join(cfg.DATA_DIR, 'ctrl_eval.pkl')
if os.path.exists(fullname):
    with open(fullname, 'rb') as f:
        runs = pickle.load(f)
        logger.info("loaded runs from %s", fullname)
        name_to_res_summary = {}

        tasks = []
        for task, res in runs.items():
            # optional filter on task

            tasks.append(task)
            names = list(res.keys())
            values = list(res.values())

            # name filter
            valid = [i for i in range(len(names)) if 'Global' in names[i]]
            names = [names[i] for i in valid]
            values = [values[i] for i in valid]

            # TODO abbreviate names

            total_costs = [v[0] for v in values]
            successes = [v[1] for v in values]

            for i in range(len(names)):
                if names[i] not in name_to_res_summary:
                    name_to_res_summary[names[i]] = {'success': [], 'costs': []}
                name_to_res_summary[names[i]]['costs'].append(torch.mean(total_costs[i]).item())
                name_to_res_summary[names[i]]['success'].append(int(torch.sum(successes[i]).item()))

            f = cumulative_dist(total_costs, names, 'total trajectory cost', successes)
            f.suptitle('task {}'.format(task))

        print(tasks)
        print(name_to_res_summary)
        for name in name_to_res_summary:
            print(name)
            values = name_to_res_summary[name]
            print('success {} {}'.format(sum(values['success']), values['success']))
            print('costs {:.2f} {}'.format(sum(values['costs']) / len(values['costs']),
                                           [round(v, 2) for v in values['costs']]))
        plt.show()
