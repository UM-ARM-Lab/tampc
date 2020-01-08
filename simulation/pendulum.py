import gym
import numpy as np
import torch
import logging
import math
from gym import wrappers, logger as gym_log
from arm_pytorch_utilities import rand, load_data, math_utils

from meta_contact.controller import global_controller
from meta_contact.controller import online_controller
from meta_contact import prior
from meta_contact import model
from arm_pytorch_utilities.model import make
from arm_pytorch_utilities.make_data import datasource
import time

gym_log.set_level(gym_log.INFO)
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format='[%(levelname)s %(asctime)s %(pathname)s:%(lineno)d] %(message)s',
                    datefmt='%m-%d %H:%M:%S')


class PendulumDataset(datasource.DataSource):
    def __init__(self, data, preprocessor=None, config=load_data.DataConfig(), **kwargs):

        super().__init__(**kwargs)

        self.data = data
        self.preprocessor = preprocessor
        self.config = config
        self.make_data()

    def make_data(self):
        if self.data is None:
            return
        XU = self.data
        if self.config.predict_difference:
            dtheta = math_utils.angular_diff_batch(XU[1:, 0], XU[:-1, 0])
            dtheta_dt = XU[1:, 1] - XU[:-1, 1]
            Y = torch.cat((dtheta.view(-1, 1), dtheta_dt.view(-1, 1)), dim=1)  # x' - x residual
        else:
            Y = XU[1:, :2]
        XU = XU[:-1]  # make same size as Y

        self.N = XU.shape[0]
        self.config.load_data_info(XU[:, :2], XU[:, 2:], Y, XU)

        if self.preprocessor:
            self.preprocessor._fit_impl(XU, Y, None)
            # apply
            XU = self.preprocessor.transform_x(XU)
            Y = self.preprocessor.transform_y(Y)

        self._train = XU, Y, None
        self._val = self._train

    def data_id(self):
        """String identification for this data"""
        return "{}".format(self.N)


def angle_normalize(x):
    return (((x + math.pi) % (2 * math.pi)) - math.pi)


def compare_to_goal(state, goal):
    if len(goal.shape) == 1:
        goal = goal.view(1, -1)
    dtheta = math_utils.angular_diff_batch(state[:, 0], goal[:, 0])
    dtheta_dt = state[:, 1] - goal[:, 1]
    diff = torch.cat((dtheta.view(-1, 1), dtheta_dt.view(-1, 1)), dim=1)
    return diff


def compare_to_goal_np(state, goal):
    if len(goal.shape) == 1:
        goal = goal.reshape(1, -1)
    dtheta = math_utils.angular_diff_batch(state[:, 0], goal[:, 0])
    dtheta_dt = state[:, 1] - goal[:, 1]
    diff = np.column_stack((dtheta.reshape(-1, 1), dtheta_dt.reshape(-1, 1)))
    return diff


# def running_cost(state, action):
#     theta = state[:, 0]
#     theta_dt = state[:, 1]
#     action = action[:, 0]
#     cost = angle_normalize(theta) ** 2 + 0.1 * theta_dt ** 2 + 0.001 * action ** 2
#     return cost


def run(ctrl, env, retrain_dynamics, config, retrain_after_iter=50, iter=1000, render=True):
    dataset = torch.zeros((retrain_after_iter, config.nx + config.nu), dtype=torch.double)
    total_reward = 0
    for i in range(iter):
        state = env.state.copy()
        command_start = time.perf_counter()
        action = ctrl.command(np.array(state))
        if torch.is_tensor(action):
            action = action.numpy()
        elapsed = time.perf_counter() - command_start
        s, r, _, _ = env.step(action)
        total_reward += r
        logger.debug("action taken: %.4f cost received: %.4f time taken: %.5fs", action, -r, elapsed)
        if render:
            env.render()

        di = i % retrain_after_iter
        if di == 0 and i > 0:
            retrain_dynamics(dataset)
            # don't have to clear dataset since it'll be overridden, but useful for debugging
            dataset.zero_()
        dataset[di, :config.nx] = torch.tensor(state)
        dataset[di, config.nx:] = torch.tensor(action)
    return total_reward, dataset


if __name__ == "__main__":
    ENV_NAME = "Pendulum-v0"
    TIMESTEPS = 15  # T
    N_SAMPLES = 100  # K
    ACTION_LOW = -2.0
    ACTION_HIGH = 2.0
    num_frames = 1000

    d = "cpu"
    dtype = torch.double

    logger.info("random seed %d", rand.seed(7))

    # new hyperparmaeters for approximate dynamics
    TRAIN_EPOCH = 150
    BOOT_STRAP_ITER = 100

    nx = 2
    nu = 1

    Q = torch.tensor([[1, 0], [0, 0.1]], dtype=dtype)
    R = 0.001

    config = load_data.DataConfig(predict_difference=True, predict_all_dims=True)

    preprocessor = None
    ds = PendulumDataset(None, preprocessor=preprocessor, config=config)


    def fill_dataset(new_data):
        global ds
        # not normalized inside the simulator
        new_data[:, 0] = angle_normalize(new_data[:, 0])
        if not torch.is_tensor(new_data):
            new_data = torch.from_numpy(new_data)
        # clamp actions
        new_data[:, -1] = torch.clamp(new_data[:, -1], ACTION_LOW, ACTION_HIGH)
        # append data to whole dataset
        if ds.data is None:
            ds.data = new_data
        else:
            ds.data = torch.cat((ds.data, new_data), dim=0)
        ds.make_data()


    downward_start = True
    env = gym.make(ENV_NAME).env  # bypass the default TimeLimit wrapper
    env.reset()
    if downward_start:
        env.state = [np.pi, 1]

    # need to place this above definition of train so the dataset has some data
    # bootstrap network with random actions
    logger.info("bootstrapping with random action for %d actions", BOOT_STRAP_ITER)
    new_data = np.zeros((BOOT_STRAP_ITER, nx + nu))
    for i in range(BOOT_STRAP_ITER):
        pre_action_state = env.state
        action = np.random.uniform(low=ACTION_LOW, high=ACTION_HIGH)
        env.step([action])
        # env.render()
        new_data[i, :nx] = pre_action_state
        new_data[i, nx:] = action

    fill_dataset(new_data)
    logger.info("bootstrapping finished")

    # pm = prior.GMMPrior.from_data(ds)
    # pm = prior.LSQPrior.from_data(ds)
    mw = model.NetworkModelWrapper(
        model.DeterministicUser(
            make.make_sequential_network(config, activation_factory=torch.nn.Tanh, h_units=(16, 16))), ds)
    pm = prior.NNPrior.from_data(mw, train_epochs=0)

    Nv = 1000
    statev = torch.cat(((torch.rand(Nv, 1, dtype=torch.double) - 0.5) * 2 * math.pi,
                        (torch.rand(Nv, 1, dtype=torch.double) - 0.5) * 16), dim=1)
    actionv = (torch.rand(Nv, 1, dtype=torch.double) - 0.5) * (ACTION_HIGH - ACTION_LOW)


    def dynamics(state, perturbed_action):
        u = torch.clamp(perturbed_action, ACTION_LOW, ACTION_HIGH)
        if state.dim() is 1 or u.dim() is 1:
            state = state.view(1, -1)
            u = u.view(1, -1)
        if u.shape[1] > 1:
            u = u[:, 0].view(-1, 1)
        xu = torch.cat((state, u), dim=1)

        next_state = mw.predict(xu)
        next_state[:, 0] = angle_normalize(next_state[:, 0])
        return next_state


    def true_dynamics(state, perturbed_action):
        # true dynamics from gym
        th = state[:, 0].view(-1, 1)
        thdot = state[:, 1].view(-1, 1)

        g = 10
        m = 1
        l = 1
        dt = 0.05

        u = perturbed_action
        u = torch.clamp(u, -2, 2)

        newthdot = thdot + (-3 * g / (2 * l) * np.sin(th + np.pi) + 3. / (m * l ** 2) * u) * dt
        newth = th + newthdot * dt
        newthdot = torch.clamp(newthdot, -8, 8)

        state = torch.cat((newth, newthdot), dim=1)
        return state


    # ctrl = online_controller.OnlineController(pm, ds, max_timestep=num_frames, Q=Q.numpy(), R=R, horizon=20, lqr_iter=3,
    #                                           init_gamma=0.1, max_ctrl=ACTION_HIGH, compare_to_goal=compare_to_goal_np)
    # ctrl = global_controller.GlobalLQRController(ds, u_max=ACTION_HIGH, Q=Q, R=R)
    # NOTE setting u_max to be ACTION_HIGH doesn't work due to over-clamping trajectory (no longer Gaussian)
    # ctrl = global_controller.GlobalCEMController(dynamics, ds, R=R, Q=Q, compare_to_goal=compare_to_goal,
    #                                              u_max=torch.tensor(ACTION_HIGH, dtype=dtype), init_cov_diag=10)
    ctrl = global_controller.GlobalMPPIController(dynamics, ds, R=R, Q=Q, compare_to_goal=compare_to_goal,
                                                  u_max=torch.tensor(ACTION_HIGH, dtype=dtype),
                                                  noise_sigma=torch.eye(nu, dtype=dtype) * 5)

    ctrl.set_goal(np.array([0, 0]))


    def update_model():
        # TODO recreate prior if necessary (for non-network based priors; since they contain data directly)
        # pm = prior.LSQPrior.from_data(ds)
        # pm = prior.GMMPrior.from_data(ds)
        # ctrl.update_prior(pm)
        # # update model based on database change (for global linear controllers)
        # ctrl.update_model(ds)
        # retrain network (for nn dynamics based controllers)
        mw.unfreeze()
        mw.learn_model(TRAIN_EPOCH, batch_N=10000)
        mw.freeze()

        # evaluate network against true dynamics
        yt = true_dynamics(statev, actionv)
        yp = dynamics(statev, actionv)
        dtheta = math_utils.angular_diff_batch(yp[:, 0], yt[:, 0])
        dtheta_dt = yp[:, 1] - yt[:, 1]
        E = torch.cat((dtheta.view(-1, 1), dtheta_dt.view(-1, 1)), dim=1).norm(dim=1)
        logger.info("Error with true dynamics theta %f theta_dt %f norm %f", dtheta.abs().mean(),
                    dtheta_dt.abs().mean(), E.mean())
        logger.debug("Start next collection sequence")


    def train(new_data):
        fill_dataset(new_data)
        update_model()


    update_model()

    # reset state so it's ready to run
    env = wrappers.Monitor(env, '/tmp/meta_pend/', force=True)
    env.reset()
    if downward_start:
        env.env.state = [np.pi, 1]
    total_reward, data = run(ctrl, env, train, config, iter=num_frames)
    logger.info("Total reward %f", total_reward)
