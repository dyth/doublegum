import jax
from jax               import numpy as jnp
from jax.scipy.special import logsumexp
from policies_disc.agents.DQN        import DQN
from policies_disc.networks.gaussian import Gaussian as Network


class DoubleGumv2(DQN):
    def __init__(self, obs, num_actions, args):
        self.network_fn = lambda obs: Network(num_actions, args.network, args.network_groups)(obs)
        self.initialize_all(obs, args)


    def beta_log_sum_exp(self, info):
        locs        = info.pop('q')
        std         = info.pop('std')
        spread      = std * jnp.sqrt(6) / jnp.pi
        exponents   = locs / spread
        log_sum_exp = logsumexp(exponents, axis=-1)
        return log_sum_exp.squeeze(), spread.squeeze()


    def soft_v(self, state, seed, obs):
        info                = self.network_model_apply(state.target.network, state.network.sn_state, seed, obs)[0]
        log_sum_exp, spread = self.beta_log_sum_exp(info)
        return spread * (log_sum_exp + jnp.euler_gamma)


    def target_network(self, state):
        info                = self.network(self.state.target.network, self.state.network.sn_state, self.rngs.get_key(), state)[0]
        log_sum_exp, spread = self.beta_log_sum_exp(info)
        return spread * (log_sum_exp + jnp.euler_gamma)


    def network_loss(self, params, state, batch, seed):
        state        = jax.lax.stop_gradient(state)
        seed1, seed2 = jax.random.split(seed)

        target_Q = self.soft_v(state, seed1, batch['next_obs'])
        discount = self.args.discount ** self.args.nstep
        done     = 1. - batch['done'].squeeze()
        target_Q = done * discount * target_Q
        target_Q = batch['rew'].squeeze() + target_Q

        online_info, sn_state = self.network_model_apply(params, state.network.sn_state, seed2, batch['obs'], update_stats=True)
        online_Q              = online_info.pop('q')
        online_std            = online_info.pop('std').squeeze()
        online_spread         = online_std * jnp.sqrt(6) / jnp.pi
        action                = batch['act'].squeeze().astype(int)
        online_Q              = online_Q[self.index, action]
        online_Q             += jnp.euler_gamma * online_spread

        std_sg    = jax.lax.stop_gradient(online_std)
        td_loss   = online_Q - target_Q
        loss      = jnp.log(online_std) + .5 * (td_loss / online_std) ** 2.
        loss      = std_sg * loss
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


    def _td_loss(self, state, batch, seed):
        state        = jax.lax.stop_gradient(state)
        seed1, seed2 = jax.random.split(seed)

        target_Q = self.soft_v(state, seed1, batch['next_obs'])
        discount = self.args.discount ** self.args.nstep
        done     = 1. - batch['done'].squeeze()
        target_Q = done * discount * target_Q
        target_Q = batch['rew'].squeeze() + target_Q

        online_info, sn_state = self.network_model_apply(state.network.params, state.network.sn_state, seed2, batch['obs'], update_stats=True)
        online_Q              = online_info.pop('q')
        online_std            = online_info.pop('std').squeeze()
        online_spread         = online_std * jnp.sqrt(6) / jnp.pi
        action                = batch['act'].squeeze().astype(int)
        online_Q              = online_Q[self.td_index, action]
        online_Q             += jnp.euler_gamma * online_spread

        aux = {
            'target_Q': target_Q,
            'online_Q': online_Q,
            'online_std': online_std,
        }
        return aux
