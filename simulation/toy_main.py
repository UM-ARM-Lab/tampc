import os
from tensorboardX import SummaryWriter

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
from sklearn.preprocessing import PolynomialFeatures
from arm_pytorch_utilities import linalg

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
    global save_dir
    init_state = [-1.5, 1.5]
    goal = [2, -2]
    # noise = (0.04, 0.04)
    noise = (0.0, 0.0)
    # env = toy.WaterWorld(init_state, goal, mode=mode, process_noise=noise, max_move_step=0.01)
    # save_dir = 'linear/linear0'
    env = toy.PolynomialWorld(init_state, goal, mode=mode, process_noise=noise, max_move_step=0.01,
                              keep_within_bounds=False)
    save_dir = 'poly/poly0'
    return env


def test_env_control():
    env = get_env(myenv.Mode.GUI)
    u_min, u_max = get_control_bounds()
    ctrl = controller.FullRandomController(env.nu, u_min, u_max)
    sim = toy.ToySim(env, ctrl, num_frames=100, plot=False, save=False)
    seed = rand.seed()
    sim.run(seed)


def collect_data(trials=20, trial_length=40, x_min=(-3, 0), x_max=(3, 3)):
    logger.info("initial random seed %d", rand.seed(1))
    env = get_env(myenv.Mode.DIRECT)
    u_min, u_max = get_control_bounds()
    ctrl = controller.FullRandomController(env.nu, u_min, u_max)

    sim = toy.ToySim(env, ctrl, num_frames=trial_length, plot=False, save=True, save_dir=save_dir)

    # randomly distribute data
    for _ in range(trials):
        seed = rand.seed()
        # randomize so that our prior is accurate in one mode but not the other
        init_state = np.random.uniform(x_min, x_max)
        env.set_task_config(init_state=init_state)
        ctrl = controller.FullRandomController(env.nu, u_min, u_max)
        sim.ctrl = ctrl
        try:
            sim.run(seed)
        except ValueError as e:
            logger.warning("Ignoring trial with error: %s", e)

    if sim.save:
        load_data.merge_data_in_dir(cfg, save_dir, save_dir)
    plt.ioff()
    plt.show()


def show_prior_accuracy(expected_max_error=1., relative=True):
    """
    Plot a contour map of prior model accuracy linearized across a grid over the state and sampled actions at each state
    :return:
    """
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
    config = load_data.DataConfig(predict_difference=True, predict_all_dims=True, expanded_input=False)
    ds = toy.ToyDataSource(data_dir=save_dir + '.mat', preprocessor=preprocessor, validation_ratio=0.1,
                           config=config)
    env = get_env(myenv.Mode.DIRECT)

    # load prior
    # pm = prior.LSQPrior.from_data(ds)
    # pm = prior.GMMPrior.from_data(ds)
    mw = model.NetworkModelWrapper(model.DeterministicUser(make.make_sequential_network(config)), ds,
                                   name='linear')
    checkpoint = '/Users/johnsonzhong/Research/meta_contact/checkpoints/linear.1150.tar'
    pm = prior.NNPrior.from_data(mw, checkpoint=checkpoint, train_epochs=50, batch_N=500)

    # we can evaluate just prior dynamics by mixing with N=0 (no weight for empirical data)
    dynamics = online_dynamics.OnlineDynamics(0.1, pm, ds, N=0)

    bounds = get_control_bounds()
    num_actions = 20
    logger.info("random seed %d", rand.seed(1))
    u = torch.from_numpy(np.random.uniform(*bounds, (num_actions, env.nu)))
    if preprocessor is not None and not isinstance(pm, prior.NNPrior):
        raise RuntimeError("Can't use preprocessor with non NN-prior since it'll return matrices wrt transformed units")
    for i, xy in enumerate(XY):
        xy = torch.from_numpy(xy).repeat(num_actions, 1)
        params = dynamics.get_batch_dynamics(xy, u, xy, u)
        nxp = online_dynamics.batch_evaluate_dynamics(xy, u, *params)
        nxt = torch.from_numpy(env.true_dynamics(xy.numpy(), u.numpy()))

        diff = nxt - nxp

        if relative:
            actual_delta = torch.norm(nxt - xy, dim=1)
            valid = actual_delta > 0
            diff = diff[valid]
            actual_delta = actual_delta[valid]
            if torch.any(valid):
                Z[i] = (torch.norm(diff, dim=1) / actual_delta).mean()
            else:
                Z[i] = 0
        else:
            Z[i] = (torch.norm(diff, dim=1)).mean()

    # normalize to per action
    Z = Z.reshape(X.shape)
    logger.info("Error min %f max %f median %f std %f", Z.min(), Z.max(), np.median(Z), Z.std())

    fig, ax = plt.subplots()

    # CS = ax.contourf(X, Y, Z, cmap='plasma', vmin=0, vmax=expected_max_error)
    CS = ax.tripcolor(X.ravel(), Y.ravel(), Z.ravel(), cmap='plasma', vmin=0, vmax=expected_max_error)
    CBI = fig.colorbar(CS)
    CBI.ax.set_ylabel('local model relative error')
    ax.set_ylabel('y')
    ax.set_xlabel('x')
    ax.set_title('linearized prior model error')
    plt.show()


def compare_empirical_and_prior_error(trials=20, trial_length=50, expected_max_error=1.):
    """
    Compare the empirical linear model against the linearized prior model's accuracy through running
    an online controller on randomly sampled trials.
    :param trials:
    :param trial_length:
    :param expected_max_error:
    :return:
    """
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
    config = load_data.DataConfig(predict_difference=True, predict_all_dims=True, expanded_input=False)
    ds = toy.ToyDataSource(data_dir=save_dir + '.mat', preprocessor=preprocessor, validation_ratio=0.1,
                           config=config)

    # load prior
    # pm = prior.LSQPrior.from_data(ds)
    # pm = prior.GMMPrior.from_data(ds)
    mw = model.NetworkModelWrapper(model.DeterministicUser(make.make_sequential_network(config)), ds,
                                   name='linear')
    # checkpoint = '/Users/johnsonzhong/Research/meta_contact/checkpoints/linear.1470.tar'
    checkpoint = '/home/zhsh/catkin_ws/src/meta_contact/checkpoints/linear.630.tar'
    checkpoint = None
    pm = prior.NNPrior.from_data(mw, checkpoint=checkpoint, train_epochs=70, batch_N=500)
    u_min, u_max = get_control_bounds()
    ctrl = online_controller.OnlineLQR(pm, ds=ds, max_timestep=trial_length, R=3, horizon=10, lqr_iter=3,
                                       init_gamma=0.1, u_min=u_min, u_max=u_max)

    logger.info("initial random seed %d", rand.seed(1))
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
    # CS = ax.tricontourf(xy[:, 0], xy[:, 1], emp_error, 10, cmap='plasma', vmin=0, vmax=expected_max_error)
    CS = ax.tripcolor(xy[:, 0], xy[:, 1], emp_error, cmap='plasma', vmin=0, vmax=expected_max_error / 2)
    CBI = fig.colorbar(CS)
    CBI.ax.set_ylabel('model error')
    ax.set_ylabel('y')
    ax.set_xlabel('x')
    ax.set_title('empirical local model error')

    fig2, ax2 = plt.subplots()
    # CS2 = ax2.tricontourf(xy[:, 0], xy[:, 1], prior_error, 10, cmap='plasma', vmin=0, vmax=expected_max_error)
    CS2 = ax2.tripcolor(xy[:, 0], xy[:, 1], prior_error, cmap='plasma', vmin=0, vmax=expected_max_error)
    CBI2 = fig2.colorbar(CS2)
    CBI2.ax.set_ylabel('model error')
    ax2.set_ylabel('y')
    ax2.set_xlabel('x')
    ax2.set_title('prior linearized model error')

    plt.show()


def learn_invariance():
    dtype = torch.double
    MAX_EPOCH = 50
    BATCH_SIZE = 1000
    TOO_FAR_FOR_NEIGHBOUR = 1.0  # how far from current point to consider for neighbourhood
    env = get_env(myenv.Mode.DIRECT)

    # encoding of the invariance
    # for the easiest case, parameterize our encoder just broadly enough to include the actual encoding
    # we know there is linear dynamics in the invariant/latent space
    order = 2
    poly = PolynomialFeatures(order, include_bias=False)
    # create input sample to fit (tells sklearn what input sizes to expect)
    x = np.random.rand(env.nx).reshape(1, -1)
    poly.fit(x)

    seed = 1
    logger.info("initial random seed %d", rand.seed(seed))
    params = torch.rand(poly.n_output_features_, dtype=dtype, requires_grad=True)

    # TODO try a more generalized encoder later (NN with state as input; should be able to learn polynomial)
    def encode(state):
        polyout = poly.transform(state)
        z = torch.from_numpy(polyout) @ params
        return z

    preprocessor = None
    config = load_data.DataConfig(predict_difference=True, predict_all_dims=True, expanded_input=False)
    ds = toy.ToyDataSource(data_dir=save_dir + '.mat', preprocessor=preprocessor, validation_ratio=0.1,
                           config=config)
    # train from samples of ds that are close in euclidean space
    XU, Y, _ = ds.training_set()
    X, U = torch.split(XU, config.nx, dim=1)
    N = XU.shape[0]
    neighbourhood = []
    # predetermine which datapoints are neighbours since it's dependent on only XUY (which don't change during training)
    # since training set is not shuffled, we can just look at adjacent datapoints sequentially
    # TODO remove this assumption by doing (expensive) pairwise-distance, or sampling?
    # for each datapoint we look at other datapoints close to it
    for i in range(N):
        # TODO for now just consider distance in X, later consider U and also Y? (for example would indicate contact)
        # bounds on neighbourhood (assume continuous)
        li = i
        ri = i
        cur = X[i]
        while li > 0:
            next = X[li]
            # TODO encapsulate comparison for being too far
            if torch.norm(next - cur) > TOO_FAR_FOR_NEIGHBOUR:
                break
            li -= 1
            # TODO decide if we also should update our cur - will result in chaining neighbours
            # cur = next
        while ri < N:
            next = X[ri]
            if torch.norm(next - cur) > TOO_FAR_FOR_NEIGHBOUR:
                break
            ri += 1
        neighbourhood.append(slice(li, ri))

    # analysis for neighbourhood size; useful for tuning hyperparameters
    neighbourhood_size = [s.stop - s.start for s in neighbourhood]
    logger.info("min neighbourhood size %d max %d median %d median %f", np.min(neighbourhood_size),
                np.max(neighbourhood_size),
                np.median(neighbourhood_size), np.mean(neighbourhood_size))

    # TODO attach summarywriter
    writer = SummaryWriter(flush_secs=20, comment="{}_batch_{}".format(seed, BATCH_SIZE))

    # actually do training
    optimizer = torch.optim.Adam([params])
    losses = []
    cos_sim = []
    step = 0
    optimizer.zero_grad()
    batch_cov_loss = None
    batch_mse_loss = None
    for epoch in range(MAX_EPOCH):
        # randomize the order we're looking at the neighbourhoods
        neighbour_order = np.random.permutation(N)
        # TODO batch process neighbours
        for i in neighbour_order:
            bi = step % BATCH_SIZE
            if bi == 0:
                # treat previous batch
                if batch_cov_loss is not None:
                    # hold stats
                    batch_cov_loss.mean().backward()
                    optimizer.step()

                    losses.append(batch_cov_loss.mean().item())
                    c = torch.nn.functional.cosine_similarity(params, torch.from_numpy(env.true_params), dim=0)
                    cos_sim.append(c.item())
                    logger.info("step %d loss %f cos dist %f", step, losses[-1], cos_sim[-1])

                    writer.add_scalar('mse_loss', batch_mse_loss.mean().item(), step)
                    writer.add_scalar('cov_loss', batch_cov_loss.mean().item(), step)
                    writer.add_scalar('cosine_similiarty', cos_sim[-1], step)

                    optimizer.zero_grad()

                batch_cov_loss = torch.zeros(BATCH_SIZE)
                batch_mse_loss = torch.zeros_like(batch_cov_loss)

            neighbours = neighbourhood[i]
            # TODO reject neighbourhoods that are too small
            xu = XU[neighbours]
            y = Y[neighbours]
            z = encode(xu[:, :env.nx])

            if len(z.shape) < 2:
                z = z.view(-1, 1)
            # fit linear model to latent state
            p, cov = linalg.ls_cov(z, y)
            # covariance loss
            cov_loss = cov.trace()
            # mse loss
            yhat = z @ p.t()
            mse_loss = torch.norm(yhat - y, dim=1)

            batch_cov_loss[bi] = cov_loss.mean()
            batch_mse_loss[bi] = mse_loss.mean()

            step += 1

    plt.plot(losses)
    plt.xlabel('step')
    plt.ylabel('local loss')

    plt.figure()
    plt.plot(cos_sim)
    plt.xlabel('step')
    plt.ylabel('cosine similarity')
    plt.show()


if __name__ == "__main__":
    # test_env_control()
    # collect_data(500, 20, x_min=(-3, -3), x_max=(3, 3))
    # show_prior_accuracy(relative=False)
    # compare_empirical_and_prior_error(200, 50)
    learn_invariance()
