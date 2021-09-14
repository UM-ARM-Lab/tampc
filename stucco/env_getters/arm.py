import typing

from arm_pytorch_utilities import load_data
from arm_pytorch_utilities.optim import get_device
from stucco import tracking
from stucco.env import arm
from stucco.env.arm import Levels
from stucco.env_getters.getter import EnvGetter


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
    def controller_options(env) -> typing.Tuple[dict, dict]:
        return {}, {}

    @staticmethod
    def contact_parameters(env: arm.ArmEnv, **kwargs) -> tracking.ContactParameters:
        params = tracking.ContactParameters(state_to_pos=env.get_ee_pos_states,
                                            pos_to_state=env.get_state_ee_pos,
                                            control_similarity=env.control_similarity,
                                            state_to_reaction=env.get_ee_reaction,
                                            max_pos_move_per_action=env.MAX_PUSH_DIST,
                                            length=0.02,
                                            hard_assignment_threshold=0.4,
                                            intersection_tolerance=0.002,
                                            weight_multiplier=0.1,
                                            ignore_below_weight=0.2)
        if kwargs is not None:
            for k, v in kwargs.items():
                setattr(params, k, v)
        return params

    @classmethod
    def env(cls, level=0, log_video=True, **kwargs):
        level = Levels(level)
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


class RetrievalGetter(ArmGetter):
    @staticmethod
    def dynamics_prefix() -> str:
        return "arm"

    @classmethod
    def env(cls, level=Levels.NO_CLUTTER, log_video=True, **kwargs):
        level = Levels(level)
        init = None
        goal = None
        if level is Levels.SIMPLE_CLUTTER:
            init = [0, 0]
            goal = [0.5, -0.1, 0]
        elif level is Levels.FLAT_BOX:
            init = [0, 0.1]
            goal = [0.15, 0.05, 0]
        elif level is Levels.BEHIND_CAN:
            init = [0, 0.1]
            goal = [0.25, 0.05, 1.2]
        elif level is Levels.IN_BETWEEN:
            init = [0, 0.05]
            goal = [0.18, 0, 1.7]

        env = arm.ObjectRetrievalEnv(environment_level=level, log_video=log_video, init=init, goal=goal, **kwargs)
        cls.env_dir = '{}/gripper'.format(cls.dynamics_prefix())
        return env
