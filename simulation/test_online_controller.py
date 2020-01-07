import matplotlib.pyplot as plt
import torch
import numpy as np
import logging
from meta_contact.env import myenv
from meta_contact.env import toy
from meta_contact.controller import controller
from arm_pytorch_utilities import rand, load_data
from meta_contact import cfg
from meta_contact import prior
from meta_contact import online_dynamics
from meta_contact import model
from arm_pytorch_utilities.model import make
from arm_pytorch_utilities import preprocess
from meta_contact.controller import online_controller

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG,
                    format='[%(levelname)s %(asctime)s %(pathname)s:%(lineno)d] %(message)s',
                    datefmt='%m-%d %H:%M:%S')
logging.getLogger('matplotlib.font_manager').disabled = True

save_dir = 'linear/linear0'


def get_control_bounds():
    max_mag = 0.3
    u_min = np.array([-max_mag, -max_mag])
    u_max = np.array([max_mag, max_mag])
    return u_min, u_max


def get_env(mode=myenv.Mode.GUI):
    init_state = [-1.5, 1.5]
    goal = [2, -2]
    # noise = (0.04, 0.04)
    noise = (0.0, 0.0)
    env = toy.WaterWorld(init_state, goal, mode=mode, process_noise=noise, max_move_step=0.01)
    return env


def test_env_control():
    env = get_env(myenv.Mode.GUI)
    u_min, u_max = get_control_bounds()
    ctrl = controller.FullRandomController(env.nu, u_min, u_max)
    sim = toy.ToySim(env, ctrl, num_frames=100, plot=False, save=False)
    seed = rand.seed()
    sim.run(seed)


def collect_data(trials=20, trial_length=40, min_allowed_y=0):
    u_min, u_max = get_control_bounds()
    ctrl = controller.FullRandomController(2, u_min, u_max)

    env = get_env(myenv.Mode.DIRECT)
    sim = toy.ToySim(env, ctrl, num_frames=trial_length, plot=False, save=True, save_dir=save_dir)

    # randomly distribute data
    for _ in range(trials):
        seed = rand.seed()
        # randomize so that our prior is accurate in one mode but not the other
        init_state = np.random.uniform((-3, min_allowed_y), (3, 3))
        env.set_task_config(init_state=init_state)
        ctrl = controller.FullRandomController(env.nu, u_min, u_max)
        sim.ctrl = ctrl
        sim.run(seed)

    if sim.save:
        load_data.merge_data_in_dir(cfg, save_dir, save_dir)
    plt.ioff()
    plt.show()


def show_prior_accuracy():
    # create grid over state-input space
    delta = 0.2
    start = -3
    end = 3.01

    x = y = np.arange(start, end, delta)
    X, Y = np.meshgrid(x, y)
    XY = np.c_[X.ravel(), Y.ravel()]
    Z = np.zeros(XY.shape[0])

    # plot a contour map over the state space - input space of how accurate the prior is
    # can't use preprocessor except for the neural network prior because their returned matrices are wrt transformed
    preprocessor = preprocess.PytorchPreprocessing(preprocess.MinMaxScaler())
    preprocessor = None
    config = load_data.DataConfig(predict_difference=False, predict_all_dims=True, expanded_input=False)
    ds = toy.ToyDataSource(data_dir=save_dir + '.mat', preprocessor=preprocessor, validation_ratio=0.1,
                           config=config)
    env = get_env(myenv.Mode.DIRECT)

    # load prior
    pm = prior.LSQPrior.from_data(ds)
    # pm = prior.GMMPrior.from_data(ds)
    # mw = model.NetworkModelWrapper(model.DeterministicUser(make.make_sequential_network(config)), ds,
    #                                name='linear_full')
    # checkpoint = None
    # pm = prior.NNPrior.from_data(mw, checkpoint=checkpoint, train_epochs=30, batch_N=500)

    # we can evaluate just prior dynamics by mixing with N=0 (no weight for empirical data)
    nx, nu = toy.WaterWorld.nx, toy.WaterWorld.nu
    N = 0
    emp_mu = np.zeros(2 * nx + nu)
    emp_sigma = np.zeros((2 * nx + nu, 2 * nx + nu))

    dynamics = online_dynamics.OnlineDynamics(0.1, pm, emp_mu, emp_sigma, nx, nu)

    u = np.zeros(nu)
    # true dynamics
    F1 = np.concatenate((env.A1, env.B1), axis=1)
    F2 = np.concatenate((env.A2, env.B2), axis=1)
    if preprocessor is not None and not isinstance(pm, prior.NNPrior):
        raise RuntimeError("Can't use preprocessor with non NN-prior since it'll return matrices wrt transformed units")
    for i, xy in enumerate(XY):
        F, f, _ = dynamics.get_dynamics(i, xy, u, xy, u)
        # compare F against A and B
        if env.state_label(xy):
            diff = F - F2
        else:
            diff = F - F1
        Z[i] = (diff.T @ diff).trace()

    Z = np.sqrt(Z)
    Z = Z.reshape(X.shape)

    fig, ax = plt.subplots()

    CS = ax.contourf(X, Y, Z, cmap='plasma', vmin=0, vmax=0.8)
    CBI = fig.colorbar(CS)
    CBI.ax.set_ylabel('local model error')
    ax.set_ylabel('y')
    ax.set_xlabel('x')
    ax.set_title('linearized prior model error')
    plt.show()


def compare_empirical_and_prior_error(trials=20, trial_length=50):
    env = get_env(myenv.Mode.DIRECT)

    # data to collect
    N = trials * (trial_length - 1)
    xy = np.zeros((N, env.nx))
    u = np.zeros((N, env.nu))
    emp_error = np.zeros(N)
    prior_error = np.zeros_like(emp_error)

    # data source
    preprocessor = preprocess.PytorchPreprocessing(preprocess.MinMaxScaler())
    preprocessor = None
    config = load_data.DataConfig(predict_difference=False, predict_all_dims=True, expanded_input=True)
    ds = toy.ToyDataSource(data_dir=save_dir + '.mat', preprocessor=preprocessor, validation_ratio=0.1,
                           config=config)

    # load prior
    pm = prior.LSQPrior.from_data(ds)
    # pm = prior.GMMPrior.from_data(ds)
    # mw = model.NetworkModelWrapper(model.DeterministicUser(make.make_sequential_network(config)), ds,
    #                                name='linear')
    # # checkpoint = '/Users/johnsonzhong/Research/meta_contact/checkpoints/linear.1470.tar'
    # # checkpoint = '/Users/johnsonzhong/Research/meta_contact/checkpoints/linear_full.1470.tar'
    # checkpoint = None
    # pm = prior.NNPrior.from_data(mw, checkpoint=checkpoint, train_epochs=70, batch_N=500)
    u_min, u_max = get_control_bounds()
    ctrl = online_controller.OnlineController(pm, ds=ds, max_timestep=trial_length, R=5, horizon=10, lqr_iter=3,
                                              init_gamma=0.1, u_min=u_min, u_max=u_max)

    logger.info("random seed %d", rand.seed(1))
    # randomly distribute data
    min_allowed_y = -2
    i = 0
    total_cost = 0
    for t in range(trials):
        seed = rand.seed()
        ctrl.reset()
        # randomize so that our prior is accurate in one mode but not the other
        init_state = np.random.uniform((-3, min_allowed_y), (3, 3))
        goal = np.random.uniform((-3, -3), (3, 0))
        env.set_task_config(init_state=init_state, goal=goal)
        ctrl.set_goal(goal)

        for j in range(trial_length):
            xy[i] = env.state
            action = ctrl.command(env.state)
            # track error
            action = np.array(action).flatten()
            obs, rew, done, info = env.step(action)
            if env.mode == myenv.Mode.GUI:
                env.render()

            if ctrl.dynamics.emp_error is not None:
                emp_error[i], prior_error[i] = ctrl.dynamics.emp_error, ctrl.dynamics.prior_error
                u[i] = action
                i += 1

            if done:
                logger.info("Done %d in %d iterations with final cost %f", t, j, -rew)
                break

        # only add the final cost
        total_cost += -rew

    logger.info("Total cost %f", total_cost)
    # strip off unused
    xy = xy[:i]
    emp_error = emp_error[:i]
    prior_error = prior_error[:i]

    plt.ioff()

    fig, ax = plt.subplots()
    CS = ax.tricontourf(xy[:, 0], xy[:, 1], emp_error, cmap='plasma', vmin=0, vmax=0.3)
    CBI = fig.colorbar(CS)
    CBI.ax.set_ylabel('model error')
    ax.set_ylabel('y')
    ax.set_xlabel('x')
    ax.set_title('empirical local model error')

    fig2, ax2 = plt.subplots()
    CS2 = ax2.tricontourf(xy[:, 0], xy[:, 1], prior_error, cmap='plasma', vmin=0, vmax=0.3)
    CBI2 = fig2.colorbar(CS2)
    CBI2.ax.set_ylabel('model error')
    ax2.set_ylabel('y')
    ax2.set_xlabel('x')
    ax2.set_title('prior linearized model error')

    plt.show()


if __name__ == "__main__":
    # test_env_control()
    # collect_data(250, 50, min_allowed_y=-3)
    # show_prior_accuracy()
    compare_empirical_and_prior_error(200, 50)