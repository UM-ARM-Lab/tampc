import typing

import torch
from arm_pytorch_utilities import preprocess
from arm_pytorch_utilities.optim import get_device
from tampc.util import UseTsf, TAMPCEnvGetter
from stucco.env_getters import arm as arm_getter


class ArmGetter(arm_getter.ArmGetter, TAMPCEnvGetter):
    @staticmethod
    def pre_invariant_preprocessor(use_tsf: UseTsf) -> preprocess.Transformer:
        return preprocess.PytorchTransformer(preprocess.NullSingleTransformer(), preprocess.RobustMinMaxScaler())

    @staticmethod
    def controller_options(env) -> typing.Tuple[dict, dict]:
        d = get_device()
        u_min, u_max = env.get_control_bounds()
        Q = torch.tensor(env.state_cost(), dtype=torch.double)
        # Q = torch.tensor([1, 1, 1], dtype=torch.double)
        R = 0.001
        # sigma = [0.2, 0.2, 0.2]
        # noise_mu = [0, 0, 0]
        # u_init = [0, 0, 0]
        sigma = [0.2 for _ in range(env.nu)]
        noise_mu = [0 for _ in range(env.nu)]
        u_init = [0 for _ in range(env.nu)]
        sigma = torch.tensor(sigma, dtype=torch.double, device=d)

        common_wrapper_opts = {
            'Q': Q,
            'R': R,
            'u_min': u_min,
            'u_max': u_max,
            'compare_to_goal': env.compare_to_goal,
            'state_dist': env.state_distance,
            'u_similarity': env.control_similarity,
            'device': d,
            'terminal_cost_multiplier': 50,
            'trap_cost_annealing_rate': 0.8,
            'abs_unrecognized_threshold': 5,
            'dynamics_minimum_window': 3,
            'max_trap_weight': 1,
            'nonnominal_dynamics_penalty_tolerance': 0.01,
            'contact_detector': env.contact_detector,
        }
        mpc_opts = {
            'num_samples': 500,
            'noise_sigma': torch.diag(sigma),
            'noise_mu': torch.tensor(noise_mu, dtype=torch.double, device=d),
            'lambda_': 1e-2,
            'horizon': 25,
            'u_init': torch.tensor(u_init, dtype=torch.double, device=d),
            'sample_null_action': False,
            'step_dependent_dynamics': True,
            'rollout_samples': 10,
            'rollout_var_cost': 0,
        }
        return common_wrapper_opts, mpc_opts
