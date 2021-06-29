import typing

import torch
from arm_pytorch_utilities import load_data, preprocess
from arm_pytorch_utilities.optim import get_device
from tampc import contact
from tampc.env import arm
from tampc.env.arm import Levels
from tampc.util import EnvGetter, UseTsf


class ArmGetter(EnvGetter):
    @staticmethod
    def dynamics_prefix() -> str:
        return "arm"

    @staticmethod
    def ds(env, data_dir, **kwargs):
        d = get_device()
        config = load_data.DataConfig(predict_difference=True, predict_all_dims=True, expanded_input=False)
        ds = arm.ArmDataSource(env, data_dir=data_dir, config=config, device=d, **kwargs)
        return ds

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

    @staticmethod
    def contact_parameters(env: arm.ArmEnv) -> contact.ContactParameters:
        return contact.ContactParameters(state_to_pos=env.get_ee_pos_states,
                                         pos_to_state=env.get_state_ee_pos,
                                         control_similarity=env.control_similarity,
                                         state_to_reaction=env.get_ee_reaction,
                                         max_pos_move_per_action=env.MAX_PUSH_DIST,
                                         length=0.1,
                                         weight_multiplier=0.1,
                                         ignore_below_weight=0.2,
                                         force_threshold=0.5)

    @classmethod
    def env(cls, level=0, log_video=True, **kwargs):
        level = Levels(level)
        # env = arm.ArmEnv(environment_level=level, log_video=log_video, **kwargs)
        # cls.env_dir = '{}/raw'.format(arm.DIR)
        # env = arm.ArmJointEnv(environment_level=level, log_video=log_video, **kwargs)
        # cls.env_dir = '{}/joints'.format(arm.DIR)
        # env.set_task_config(goal=(0.8, 0.0, 0.3))
        # env = arm.PlanarArmEnv(environment_level=level, log_video=log_video, **kwargs)
        # cls.env_dir = '{}/planar'.format(arm.DIR)
        env = arm.FloatingGripperEnv(environment_level=level, log_video=log_video, **kwargs)
        cls.env_dir = '{}/gripper'.format(arm.DIR)
        if level is Levels.MOVEABLE_CANS:
            env.set_task_config(goal=(0.95, -0.4))
        if level in (Levels.STRAIGHT_LINE, Levels.WALL_BEHIND):
            env.set_task_config(goal=[0.0, 0.], init=[1, 0])
        if level in (Levels.NCB_C, Levels.NCB_S):
            env.set_task_config(goal=[0.0, 0.], init=[1, 0])
        if level is Levels.NCB_T:
            env.set_task_config(goal=[-0.02, 0.], init=[1, 0])
        return env