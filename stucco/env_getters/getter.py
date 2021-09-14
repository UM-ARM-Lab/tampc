import typing
import abc
import logging

from arm_pytorch_utilities.make_data import datasource
from arm_pytorch_utilities import rand
from arm_pytorch_utilities.optim import get_device

from stucco import tracking

logger = logging.getLogger(__name__)


class EnvGetter(abc.ABC):
    """Utility class that centralizes creation of environment related objects"""
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
    def contact_parameters(env, **kwargs) -> tracking.ContactParameters:
        """Return tracking parameters suitable for this environment, taking kwargs as overrides"""

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

    @staticmethod
    @abc.abstractmethod
    def dynamics_prefix() -> str:
        """Return the prefix of dynamics functions corresponding to this environment"""
