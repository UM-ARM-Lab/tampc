import gym
import numpy as np
import torch
import logging
import math
import os
import scipy.io
from gym import wrappers, logger as gym_log
from arm_pytorch_utilities import rand, load_data, math_utils
from arm_pytorch_utilities import preprocess

from tampc.controller import online_controller
from tampc.controller import ilqr
from tampc.dynamics import online_model, model, prior
from tampc import cfg
from arm_pytorch_utilities.model import make
from arm_pytorch_utilities.make_data import datasource
from tampc.controller import gating_function
from tampc.dynamics import hybrid_model
from tampc.env.env import handle_data_format_for_state_diff
import time

gym_log.set_level(gym_log.INFO)
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format='[%(levelname)s %(asctime)s %(pathname)s:%(lineno)d] %(message)s',
                    datefmt='%m-%d %H:%M:%S')


class PendulumDataset(datasource.DataSource):
    def __init__(self, data, **kwargs):

        self.data = data
        self._val_unprocessed = None
        super().__init__(**kwargs)

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
        self.config.load_data_info(XU[:, :2], XU[:, 2:], Y)

        if self.preprocessor:
            self.preprocessor.tsf.fit(XU, Y)
            self.preprocessor.update_data_config(self.config)
            self._original_train = XU, Y, None
            self._original_val = self._original_train
            XU, Y, _ = self.preprocessor.tsf.transform(XU, Y)

        self._train = XU, Y, None
        self._val = self._train

    def data_id(self):
        """String identification for this data"""
        return "{}".format(self.N)


@handle_data_format_for_state_diff
def compare_to_goal(state, goal):
    dtheta = math_utils.angular_diff_batch(state[:, 0], goal[:, 0])
    dtheta_dt = state[:, 1] - goal[:, 1]
    return dtheta.reshape(-1, 1), dtheta_dt.reshape(-1, 1)


def compare_to_goal_np_in_transformed_space(state, goal):
    if len(goal.shape) == 1:
        goal = goal.reshape(1, -1)
    # dtheta = math_utils.angular_diff_batch(state[:, 0], goal[:, 0])
    # dtheta_dt = state[:, 1] - goal[:, 1]
    # diff = np.column_stack((dtheta.reshape(-1, 1), dtheta_dt.reshape(-1, 1)))
    return state - goal


def run(ctrl, env, retrain_dynamics, config, retrain_after_iter=50, iter=1000, render=True):
    dataset = torch.zeros((iter, config.nx + config.nu), dtype=torch.double)
    total_reward = 0
    for i in range(iter):
        state = env.state.copy()
        command_start = time.perf_counter()
        action = ctrl.command(np.array(state))
        if torch.is_tensor(action):
            action = action.cpu().numpy()
        elapsed = time.perf_counter() - command_start
        s, r, _, _ = env.step(action)
        total_reward += r
        logger.debug("action taken: %.4f cost received: %.4f time taken: %.5fs", action, -r, elapsed)
        if render:
            env.render()

        di = i % retrain_after_iter
        if di == 0 and i > 0:
            # just retrain with the recent points
            retrain_dynamics(dataset[i - retrain_after_iter:i])
        dataset[i, :config.nx] = torch.tensor(state)
        dataset[i, config.nx:] = torch.tensor(action)
    return total_reward, dataset


if __name__ == "__main__":
    USE_ILQR = False
    ENV_NAME = "Pendulum-v0"
    TIMESTEPS = 15  # T
    N_SAMPLES = 100  # K
    ACTION_LOW = -2.0
    ACTION_HIGH = 2.0
    USE_PREVIOUS_TRIAL_DATA = False
    SAVE_TRIAL_DATA = False
    num_frames = 500

    d = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    d = torch.device("cpu")
    dtype = torch.double

    seed = 6
    logger.info("random seed %d", rand.seed(seed))
    save_dir = os.path.join(cfg.DATA_DIR, ENV_NAME)
    save_to = os.path.join(save_dir, "{}.mat".format(seed))

    # new hyperparmaeters for approximate dynamics
    TRAIN_EPOCH = 150  # need more epochs if we're freezing prior (~800)
    BOOT_STRAP_ITER = 100

    nx = 2
    nu = 1

    Q = torch.tensor([[1, 0], [0, 0.1]], dtype=dtype, device=d)
    R = 0.001

    config = load_data.DataConfig(predict_difference=True, predict_all_dims=True)

    # preprocessor = None
    ds = PendulumDataset(None, config=config)


    def fill_dataset(new_data):
        global ds
        # not normalized inside the simulator
        new_data[:, 0] = math_utils.angle_normalize(new_data[:, 0])
        if not torch.is_tensor(new_data):
            new_data = torch.from_numpy(new_data)
        # clamp actions
        new_data[:, -1] = torch.clamp(new_data[:, -1], ACTION_LOW, ACTION_HIGH)
        new_data = new_data.to(device=d)
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

    # TODO directly making the change in state into angular representation is wrong
    preprocessor = preprocess.PytorchTransformer(preprocess.AngleToCosSinRepresentation(0),
                                                 preprocess.AngleToCosSinRepresentation(0))
    untransformed_config = ds.update_preprocessor(preprocessor)

    # pm = prior.GMMPrior.from_data(ds)
    # pm = prior.LSQPrior.from_data(ds)
    mw = model.NetworkModelWrapper(
        model.DeterministicUser(
            make.make_sequential_network(config, activation_factory=torch.nn.Tanh, h_units=(16, 16)).to(device=d)), ds)
    pm = prior.NNPrior.from_data(mw, train_epochs=0)
    # linearizable_dynamics = online_model.OnlineDynamicsModel(0.1, pm, ds, sigreg=1e-10)
    online_dynamics = online_model.OnlineLinearizeMixing(0.1, pm, ds,
                                                         compare_to_goal,
                                                         local_mix_weight_scale=1, xu_characteristic_length=10,
                                                         const_local_mix_weight=True, sigreg=1e-10,
                                                         device=d)
    hybrid_dynamics = hybrid_model.HybridDynamicsModel([ds], pm, compare_to_goal, ['none'], preprocessor=preprocessor,
                                                       nominal_model_kwargs={
                                                           'online_adapt': hybrid_model.OnlineAdapt.LINEARIZE_LIKELIHOOD})

    Nv = 1000
    statev = torch.cat(((torch.rand(Nv, 1, dtype=torch.double) - 0.5) * 2 * math.pi,
                        (torch.rand(Nv, 1, dtype=torch.double) - 0.5) * 16), dim=1).to(device=d)
    actionv = ((torch.rand(Nv, 1, dtype=torch.double) - 0.5) * (ACTION_HIGH - ACTION_LOW)).to(device=d)


    def dynamics(state, perturbed_action):
        u = torch.clamp(perturbed_action, ACTION_LOW, ACTION_HIGH)
        if state.dim() is 1 or u.dim() is 1:
            state = state.view(1, -1)
            u = u.view(1, -1)
        if u.shape[1] > 1:
            u = u[:, 0].view(-1, 1)
        xu = torch.cat((state, u), dim=1)

        next_state = mw.predict(xu)
        next_state[:, 0] = math_utils.angle_normalize(next_state[:, 0])
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

        newthdot = thdot + (-3 * g / (2 * l) * torch.sin(th + np.pi) + 3. / (m * l ** 2) * u) * dt
        newth = th + newthdot * dt
        newthdot = torch.clamp(newthdot, -8, 8)

        state = torch.cat((newth, newthdot), dim=1)
        return state


    # note that it works even if we don't constrain state
    def constrain_state(state):
        state[:, 0] = math_utils.angle_normalize(state[:, 0])
        return state


    if USE_ILQR:
        # iLQR requires more than just samples from the dynamics model but also its derivatives
        # so we have to do control directly in the transformed space
        ctrl = ilqr.OnlineLQR(online_dynamics, ds, max_timestep=num_frames, Q=np.diag([1, 1, 0.1]),
                              R=R, horizon=20,
                              lqr_iter=3,
                              u_noise=0.1, u_max=ACTION_HIGH, compare_to_goal=compare_to_goal_np_in_transformed_space,
                              device=d)
        goal = torch.tensor([0, 0, 0], dtype=torch.double, device=d)
        goal = preprocessor.transform_x(goal.view(1, -1))
        ctrl.set_goal(goal[0, :ds.config.nx].cpu().numpy())
    else:
        mppi_opts = {'num_samples': N_SAMPLES, 'horizon': TIMESTEPS, 'lambda_': 1,
                     'noise_sigma': torch.eye(nu, dtype=dtype, device=d) * 1}
        ctrl = online_controller.OnlineMPPI(ds, hybrid_dynamics, ds.original_config(),
                                            gating=gating_function.AlwaysSelectNominal(),
                                            Q=Q, R=R, u_max=ACTION_HIGH, u_min=ACTION_LOW,
                                            compare_to_goal=compare_to_goal,
                                            device=d,
                                            use_trap_cost=False,
                                            autonomous_recovery=online_controller.AutonomousRecovery.NONE,
                                            constrain_state=constrain_state, mpc_opts=mppi_opts)
        ctrl.set_goal(np.array([0, 0]))


    def update_model():
        # recreate prior if necessary (for non-network based priors; since they contain data directly)
        # pm = prior.LSQPrior.from_data(ds)
        # pm = prior.GMMPrior.from_data(ds)
        # ctrl.update_prior(pm)
        # # update model based on database change (for global linear controllers)
        # ctrl.update_model(ds)
        # retrain network (for nn dynamics based controllers)
        mw.train()
        mw.learn_model(TRAIN_EPOCH, batch_N=500)
        mw.eval()

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


    # load data from before if it exists
    if USE_PREVIOUS_TRIAL_DATA and os.path.isfile(save_to):
        loaded_data = scipy.io.loadmat(save_to)
        new_data = loaded_data['XU']
        logger.info("Load previous data from {} ({} rows)".format(save_to, new_data.shape[0]))
        fill_dataset(new_data)

    update_model()

    # reset state so it's ready to run
    env = wrappers.Monitor(env, '/tmp/meta_pend/', force=True)
    env.reset()
    if downward_start:
        env.env.state = [np.pi, 1]
    total_reward, data = run(ctrl, env, train, ds.original_config(), iter=num_frames)
    logger.info("Total reward %f", total_reward)
    # save data (on successful trials could be used as prior for the next)
    if SAVE_TRIAL_DATA:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # export in matlab/numpy compatible format
        scipy.io.savemat(save_to, mdict={'XU': data.numpy()})
        logger.info("Finished saving to {}".format(save_to))
