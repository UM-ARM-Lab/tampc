import copy
import logging

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn
from arm_pytorch_utilities import preprocess
from arm_pytorch_utilities import rand, load_data
from arm_pytorch_utilities.model import make
from meta_contact import cfg
from meta_contact.transform import invariant
from meta_contact.dynamics import online_model, model, prior
from meta_contact.controller import controller
from meta_contact.controller import online_controller
from meta_contact.env import myenv
from meta_contact.env import toy
from meta_contact.transform.invariant import InvariantTransform
from sklearn.preprocessing import PolynomialFeatures
from torch.nn.functional import cosine_similarity

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

    env = get_env(myenv.Mode.DIRECT)
    # plot a contour map over the state space - input space of how accurate the prior is
    # can't use preprocessor except for the neural network prior because their returned matrices are wrt transformed
    preprocessor = preprocess.PytorchTransformer(preprocess.MinMaxScaler())
    preprocessor = None
    config = load_data.DataConfig(predict_difference=True, predict_all_dims=True, expanded_input=False)
    ds = toy.ToyDataSource(data_dir=save_dir + '.mat', preprocessor=preprocessor, validation_ratio=0.1,
                           config=config)

    # load prior
    # pm = prior.LSQPrior.from_data(ds)
    # pm = prior.GMMPrior.from_data(ds)
    mw = model.NetworkModelWrapper(model.DeterministicUser(make.make_sequential_network(config)), ds,
                                   name='linear')
    pm = prior.NNPrior.from_data(mw, checkpoint=mw.get_last_checkpoint(), train_epochs=50, batch_N=500)

    XY, Z = evaluate_prior(env, pm, ds, relative)

    plot_prior_error(XY, Z, relative, expected_max_error)


def compare_empirical_and_prior_error(trials=20, trial_length=50, expected_max_error=0.5):
    """
    Compare the empirical linear model against the linearized prior model's accuracy through running
    an online controller on randomly sampled trials.
    :param trials:
    :param trial_length:
    :param expected_max_error:
    :return:
    """
    env = get_env(myenv.Mode.DIRECT)

    # data source
    preprocessor = preprocess.PytorchTransformer(preprocess.MinMaxScaler())
    preprocessor = None
    config = load_data.DataConfig(predict_difference=True, predict_all_dims=True, expanded_input=False)
    ds = toy.ToyDataSource(data_dir=save_dir + '.mat', preprocessor=preprocessor, validation_ratio=0.1,
                           config=config)

    # load prior
    # pm = prior.LSQPrior.from_data(ds)
    # pm = prior.GMMPrior.from_data(ds)
    mw = model.NetworkModelWrapper(model.DeterministicUser(make.make_sequential_network(config)), ds,
                                   name='linear')
    pm = prior.NNPrior.from_data(mw, checkpoint=mw.get_last_checkpoint(), train_epochs=70, batch_N=500)
    u_min, u_max = get_control_bounds()
    dynamics = online_model.OnlineDynamicsModel(0.1, pm, ds)
    ctrl = online_controller.OnlineCEM(dynamics, config, u_min=u_min, u_max=u_max,
                                       mpc_opts={'init_cov_diag': 1.})

    xy, emp_error, prior_error, costs = evaluate_ctrl(env, ctrl, trials, trial_length)

    plot_empirical_and_prior_error(xy, emp_error, prior_error, expected_max_error)


def evaluate_prior(env, pm, ds, relative=True):
    # create grid over state-input space
    delta = 0.2
    start = 1
    end = 6.01

    x = y = np.arange(start, end, delta)
    X, Y = np.meshgrid(x, y)
    XY = np.c_[X.ravel(), Y.ravel()]
    Z = np.zeros(XY.shape[0])

    # we can evaluate just prior dynamics by mixing with N=0 (no weight for empirical data)
    dynamics = online_model.OnlineDynamicsModel(0.1, pm, ds, local_mix_weight=0, sigreg=1e-10)

    if isinstance(pm, prior.NNPrior):
        # evaluate linearization by comparing error from applying model directly vs applying linearized model
        xuv, yv, _ = ds.validation_set()
        if ds.config.predict_difference:
            yv = yv + xuv[:, :ds.config.nx]
        # full model prediction
        yhat1 = pm.dyn_net.predict(xuv, already_transformed=True)

        # linearized prediction
        # these xuv are after transformation
        xv = xuv[:, :ds.config.nx]
        uv = xuv[:, ds.config.nx:]
        yhat2 = dynamics.predict(xv, uv, xv, uv, already_transformed=True)

        e1 = torch.norm((yhat1 - yv), dim=1)
        e2 = torch.norm((yhat2 - yv), dim=1)
        logger.info("Full model MSE %f linearized model MSE %f", e1.mean(), e2.mean())

    bounds = get_control_bounds()
    num_actions = 20
    logger.info("random seed %d", rand.seed(1))
    u = torch.from_numpy(np.random.uniform(*bounds, (num_actions, env.nu)))
    for i, xy in enumerate(XY):
        xy = torch.from_numpy(xy).repeat(num_actions, 1)
        nxt = torch.from_numpy(env.true_dynamics(xy.numpy(), u.numpy()))

        nxp = dynamics.predict(xy, u, xy, u)

        diff = nxt - nxp

        if relative:
            actual_delta = torch.norm(nxt - xy, dim=1)
            valid = actual_delta > 1e-10
            diff = diff[valid]
            actual_delta = actual_delta[valid]
            if torch.any(valid):
                Z[i] = (torch.norm(diff, dim=1) / actual_delta).mean()
            else:
                Z[i] = 0
        else:
            Z[i] = (torch.norm(diff, dim=1)).mean()

    # normalize to per action
    logger.info("Error min %f max %f median %f std %f", Z.min(), Z.max(), np.median(Z), Z.std())
    return XY, Z


def evaluate_ctrl(env, ctrl, trials, trial_length):
    # data to collect
    N = trials * (trial_length - 1)
    xy = np.zeros((N, env.nx))
    u = np.zeros((N, env.nu))
    emp_error = np.zeros(N)
    prior_error = np.zeros_like(emp_error)
    costs = np.zeros((trials, trial_length))

    logger.info("initial random seed %d", rand.seed(1))
    # randomly distribute data
    min_allowed_y = -2
    i = 0
    for t in range(trials):
        task_seed = rand.seed()
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
            cost = -rew

            # stepped too far out, just abort
            if cost > 1e5:
                costs[t, j] = float('inf')
                break

            costs[t, j] = cost
            if env.mode == myenv.Mode.GUI:
                env.render()

            if ctrl.dynamics.emp_error is not None:
                emp_error[i], prior_error[i] = ctrl.dynamics.emp_error, ctrl.dynamics.prior_error
                u[i] = action
                i += 1

            if done:
                break

        logger.info("task %d total iteration %d cost %f", task_seed, j, np.sum(costs[t]))

    # filter out all the ones that didn't terminate
    task_costs = np.sum(costs, axis=1)
    terminated = np.isfinite(task_costs)
    logger.info("total cost %f %d/%d terminated", np.sum(task_costs[terminated]), np.sum(terminated), trials)
    # strip off unused
    xy = xy[:i]
    emp_error = emp_error[:i]
    prior_error = prior_error[:i]
    return xy, emp_error, prior_error, costs


def plot_prior_error(XY, prior_error, relative, expected_max_error=0.5):
    fig, ax = plt.subplots()

    # CS = ax.contourf(XY[:, 0], XY[:, 1], Z, cmap='plasma', vmin=0, vmax=expected_max_error)
    CS = ax.tripcolor(XY[:, 0], XY[:, 1], prior_error, cmap='plasma', vmin=0, vmax=expected_max_error)
    CBI = fig.colorbar(CS)
    CBI.ax.set_ylabel('local model {}error'.format('relative ' if relative else ''))
    ax.set_ylabel('y')
    ax.set_xlabel('x')
    ax.set_title('linearized prior model error')
    plt.show()


def plot_empirical_and_prior_error(xy, emp_error, prior_error, expected_max_error=0.5):
    plt.ioff()

    fig, ax = plt.subplots()
    # CS = ax.tricontourf(xy[:, 0], xy[:, 1], emp_error, 10, cmap='plasma', vmin=0, vmax=expected_max_error)
    CS = ax.tripcolor(xy[:, 0], xy[:, 1], emp_error, cmap='plasma', vmin=0, vmax=expected_max_error)
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


class NoDecoderTransform(InvariantTransform):
    """
    Assume dynamics is dynamics is directly linear from z to dx, that is we don't need transforms between
    dx and dz; for simpler dynamics this assumption should be good enough
    """

    def __init__(self, ds, *args, nv=None, **kwargs):
        if nv is None:
            nv = ds.config.ny
        super().__init__(ds, *args, nv, **kwargs)

    def get_dx(self, x, z_o):
        return z_o

    def dx_to_zo(self, x, dx):
        return dx


class PolynomialInvariantTransform(NoDecoderTransform):
    def __init__(self, ds, true_params, order=2, dtype=torch.double, **kwargs):
        self.poly = PolynomialFeatures(order, include_bias=False)
        x = np.random.rand(ds.config.nx).reshape(1, -1)
        self.poly.fit(x)
        self.params = torch.rand(self.poly.n_output_features_, dtype=dtype, requires_grad=True)
        self.true_params = true_params
        # z_i is action * scalar, so it's nu sized
        super().__init__(ds, ds.config.nu, **kwargs)
        self.name = 'poly_{}'.format(self.name)

    def xu_to_z(self, state, action):
        poly_out = self.poly.transform(state)
        z = torch.from_numpy(poly_out) @ self.params

        if len(z.shape) < 2:
            z = z.view(-1, 1)
        z = action * z
        return z

    def parameters(self):
        return [self.params]

    def _model_state_dict(self):
        return self.params

    def _load_model_state_dict(self, saved_state_dict):
        self.params = saved_state_dict

    def _record_metrics(self, writer, losses, **kwargs):
        super()._record_metrics(writer, losses, **kwargs)

        cs = cosine_similarity(self.params, self.true_params, dim=0).item()
        dist = torch.norm(self.params - self.true_params).item()

        logger.debug("step %d cos dist %f dist %f", self.step, cs, dist)

        writer.add_scalar('cosine_similarity', cs, self.step)
        writer.add_scalar('param_diff', dist, self.step)
        writer.add_scalar('param_norm', self.params.norm().item(), self.step)


def learn_invariant(seed=1, name="", MAX_EPOCH=10, BATCH_SIZE=10):
    env = get_env(myenv.Mode.DIRECT)

    preprocessor = None
    config = load_data.DataConfig(predict_difference=True, predict_all_dims=True, expanded_input=False)
    ds = toy.ToyDataSource(data_dir=save_dir + '.mat', preprocessor=preprocessor, validation_ratio=0.1,
                           config=config)

    true_params = torch.from_numpy(env.true_params)
    logger.info("initial random seed %d", rand.seed(seed))
    # encoding of the invariance
    # for the easiest case, parameterize our encoder just broadly enough to include the actual encoding
    # we know there is linear dynamics in the invariant/latent space
    # invariant_tsf = PolynomialInvariantTransform(ds, env.nx, true_params,
    #                                              too_far_for_neighbour=1., train_on_continuous_data=True,
    #                                              name='{}_s{}'.format(name, seed))
    invariant_tsf = NetworkNoDecoder(ds, 2, too_far_for_neighbour=0.3,
                                     name='{}_s{}'.format(name, seed))
    # more generalized encoder

    invariant_tsf.learn_model(MAX_EPOCH, BATCH_SIZE)


class UseTransform:
    NO_TRANSFORM = 0
    POLYNOMIAL_TRANSFORM = 1
    NETWORK_TRANSFORM = 2


def evaluate_invariant(name='', trials=5, trial_length=50):
    env = get_env(myenv.Mode.DIRECT)
    seed = 1
    use_tsf = UseTransform.POLYNOMIAL_TRANSFORM

    preprocessor = None
    config = load_data.DataConfig(predict_difference=True, predict_all_dims=True, expanded_input=False)
    ds = toy.ToyDataSource(data_dir=save_dir + '.mat', preprocessor=preprocessor, validation_ratio=0.1,
                           config=config)

    logger.info("initial random seed %d", rand.seed(seed))

    # create the invariant transform
    base_name = '{}_s{}'.format(name, seed)
    transforms = {UseTransform.NO_TRANSFORM: None,
                  UseTransform.POLYNOMIAL_TRANSFORM: PolynomialInvariantTransform(ds,
                                                                                  torch.from_numpy(env.true_params),
                                                                                  too_far_for_neighbour=1.,
                                                                                  train_on_continuous_data=True,
                                                                                  name=base_name),
                  UseTransform.NETWORK_TRANSFORM: NetworkNoDecoder(ds, 2, too_far_for_neighbour=0.3,
                                                                   name=base_name)}
    transform_names = {UseTransform.NO_TRANSFORM: 'none', UseTransform.POLYNOMIAL_TRANSFORM: 'poly',
                       UseTransform.NETWORK_TRANSFORM: 'net'}
    invariant_tsf = transforms[use_tsf]

    if invariant_tsf:
        training_epochs = 10 if use_tsf is UseTransform.POLYNOMIAL_TRANSFORM else 40
        # either load or learn the transform
        if not invariant_tsf.load(invariant_tsf.get_last_checkpoint()):
            invariant_tsf.learn_model(training_epochs, 5)

        # wrap the transform as a data preprocessor
        preprocessor = invariant.InvariantTransformer(invariant_tsf)
        # update the datasource to use transformed data
        ds.update_preprocessor(preprocessor)

    prior_name = '{}_prior'.format(transform_names[use_tsf])

    # train global prior dynamics model on the same datasource
    # pm = prior.LSQPrior.from_data(ds)
    mw = model.NetworkModelWrapper(model.DeterministicUser(make.make_sequential_network(ds.config)), ds,
                                   name=prior_name)
    pm = prior.NNPrior.from_data(mw, checkpoint=mw.get_last_checkpoint(), train_epochs=70, batch_N=500)

    # evaluate prior accuracy
    # relative = True
    # XY, prior_error_offline = evaluate_prior(env, pm, ds, relative=relative)
    # plot_prior_error(XY, prior_error_offline, relative)

    # create online controller with this prior (and transformed data)
    u_min, u_max = get_control_bounds()
    dynamics = online_model.OnlineDynamicsModel(0.1, pm, ds)
    ctrl = online_controller.OnlineCEM(dynamics, ds.original_config(), u_min=u_min, u_max=u_max,
                                       mpc_opts={'init_cov_diag': 1.})

    # evaluate controller performance
    xy, emp_error, prior_error, costs = evaluate_ctrl(env, ctrl, trials, trial_length)

    plot_empirical_and_prior_error(xy, emp_error, prior_error)


if __name__ == "__main__":
    # test_env_control()
    # collect_data(500, 20, x_min=(-3, -3), x_max=(3, 3))
    # show_prior_accuracy(relative=True)
    # compare_empirical_and_prior_error(20, 50)
    # learn_invariant(0, "default", MAX_EPOCH=40, BATCH_SIZE=5)
    evaluate_invariant('default', 20, 50)


class NetworkNoDecoder(NoDecoderTransform):
    def __init__(self, ds, nz, model_opts=None, **kwargs):
        if model_opts is None:
            model_opts = {}
        config = copy.deepcopy(ds.config)
        # output the latent space instead of y
        config.ny = nz
        self.user = model.DeterministicUser(make.make_sequential_network(config, **model_opts))
        super().__init__(ds, nz, **kwargs)

    def modules(self):
        return {'dynamics': self.user.model}

    def xu_to_z(self, state, action):
        xu = torch.cat((state, action), dim=1)
        z = self.user.sample(xu)

        if self.nz is 1:
            z = z.view(-1, 1)
        # TODO see if we need to formulate it as action * z for toy problem (less generalized, but easier, and nz=1)
        # z = action * z
        return z
