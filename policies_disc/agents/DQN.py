import copy
import jax
import optax
import numpy as np
import haiku as hk
from jax import numpy as jnp
from collections import namedtuple

from policies_disc                  import utils
from policies_disc.networks.network import Network


NetworkState = namedtuple('Network', ['params', 'sn_state', 'opt_state'])
TargetState  = namedtuple('Target', ['network'])
State        = namedtuple('State',  ['network', 'target'])


class DQN:
    def __init__(self, obs, num_actions, args):
        obs_dim         = obs.shape[0]
        self.network_fn = lambda obs: Network(num_actions, args.network, args.network_groups)(obs)
        self.initialize_all(obs, args)

    def initialize_all(self, obs, args):
        self.rngs     = utils.PRNGKeys(args.seed)
        self.args     = args
        self.obs      = obs[None, :]
        network_state = self.initialize_network()
        target_state  = self.initialize_target(network_state)
        self.state    = State(network_state, target_state)
        self.index    = jnp.arange(self.args.batch_size)
        self.jit()

    def initialize_network(self):
        self.network_model = hk.transform(self.network_fn)
        params             = self.network_model.init(self.rngs.get_key(), self.obs)
        if self.args.network_sn:
            self.network_sn = hk.without_apply_rng(hk.transform_with_state(utils.sn))
            _, sn_state     = self.network_sn.init(self.rngs.get_key(), params)
        else:
            sn_state = {}
        self.network_opt = optax.adam(self.args.network_lr)
        opt_state        = self.network_opt.init(params)
        return NetworkState(params, sn_state, opt_state)

    def initialize_target(self, network_state):
        return TargetState(copy.deepcopy(network_state.params))


    def jit(self):
        if self.args.jit:
            self.network = jax.jit(self._network)
            self.step    = jax.jit(self._step)
        else:
            self.network = self._network
            self.step    = self._step


    def network_model_apply(self, params, sn_state, seed, state, update_stats=False):
        if self.args.network_sn:
            params, sn_state = self.network_sn.apply(None, sn_state, params, update_stats=update_stats)
        info = self.network_model.apply(params, seed, state)
        return info, sn_state


    def _network(self, params, sn_state, seed, state):
        return self.network_model_apply(params, sn_state, seed, state)


    def select_action(self, state):
        state  = state[None]
        seed   = self.rngs.get_key()
        action = self.network(self.state.network.params, self.state.network.sn_state, seed, state)[0]['q']
        action = action.argmax(-1)
        action = np.asarray(action)[0]
        return action

    def sample_action(self, state):
        return self.select_action(state)

    def online_network(self, state):
        q    = self.network(self.state.network.params, self.state.network.sn_state, self.rngs.get_key(), state)[0].pop('q')
        q    = q.max(-1)
        info = {'online_expected': q}
        return info

    def target_network(self, state):
        q = self.network(self.state.target.network, self.state.network.sn_state, self.rngs.get_key(), state)[0].pop('q')
        q = q.max(-1)
        return q


    def network_loss(self, params, state, batch, seed):
        state        = jax.lax.stop_gradient(state)
        seed1, seed2 = jax.random.split(seed)

        target_info = self.network_model_apply(state.target.network, state.network.sn_state, seed1, batch['next_obs'])[0]
        target_Q    = target_info.pop('q')
        target_Q    = target_Q.max(-1)
        discount    = self.args.discount ** self.args.nstep
        done        = 1. - batch['done'].squeeze()
        target_Q    = done * discount * target_Q
        target_Q    = batch['rew'].squeeze() + target_Q

        online_info, sn_state = self.network_model_apply(params, state.network.sn_state, seed2, batch['obs'], update_stats=True)
        online_Q              = online_info.pop('q')
        action                = batch['act'].squeeze().astype(int)
        online_Q              = online_Q[self.index, action]

        td_loss   = online_Q - target_Q
        loss      = td_loss ** 2.
        loss_mean = loss.mean()
        loss_std  = loss.std()

        aux = {
            'td_loss'    : td_loss.mean(),
            'td_loss_std': td_loss.std(),
            'sn_state'   : sn_state,
            'loss'       : loss_mean,
            'loss_std'   : loss_std,

            'target_Q_mean': target_Q.mean(),
            'online_Q'     : online_Q.mean(),
            'target_Q_std' : target_Q.std(),
            'online_Q_std' : online_Q.std(),
        }
        return loss_mean, aux


    def network_step(self, state, batch, seed):
        vgrad_fn        = jax.value_and_grad(self.network_loss, has_aux=True)
        (_, aux), grad  = vgrad_fn(state.network.params, state, batch, seed)
        grad, opt_state = self.network_opt.update(grad, state.network.opt_state, state.network.params)
        params          = optax.apply_updates(state.network.params, grad)
        network_state   = NetworkState(params, aux.pop('sn_state'), opt_state)
        return aux, network_state


    def soft_update(self, params, target_params, tau):
        return jax.tree_map(lambda p, tp: p * tau + tp * (1-tau), params, target_params)


    def _step(self, batch, seed, state):
        def slice(x, i):
            '''
            from https://github.com/ikostrikov/walk_in_the_park/blob/40321ecb3561f7be98d73a2c12337a878b0c18e1/rl/ag\
            ents/sac/sac_learner.py#L242-L245
            '''
            batch_size = x.shape[0] // self.args.replay_ratio
            return x[batch_size * i:batch_size * (i + 1)]

        for i in range(self.args.replay_ratio):
            mini_batch          = jax.tree_util.tree_map(lambda t: slice(t, i), batch)
            seed1, seed         = jax.random.split(seed)
            info, network_state = self.network_step(state, mini_batch, seed)
            state               = state._replace(network=network_state)
            target_params       = self.soft_update(state.network.params, state.target.network, self.args.tau)
            state               = state._replace(target=state.target._replace(network=target_params))

        return info, state


    def train(self, replay_buffer, samples=100):
        replay_samples           = replay_buffer.sample(samples)
        network_info, self.state = self.step(replay_samples, self.rngs.get_key(), self.state)
        return network_info
