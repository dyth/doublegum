import jax
from jax import numpy as jnp

from policies_cont.networks.gaussian_critic import Critic
from policies_cont.networks.actor           import Actor
from policies_cont.agents.DDPG              import DDPG


class DoubleGum(DDPG):
    def __init__(self, obs, action, args):
        obs_dim        = obs.shape[0]
        action_dim     = action.shape[0]
        self.actor_fn  = lambda obs, temp, clip: Actor(action_dim, args.expl_noise, args.actor, args.actor_groups)(obs, temp, clip)
        self.critic_fn = lambda obs, action: Critic(obs_dim+action_dim, args.critic, args.critic_groups)(obs, action)
        self.initialize_all(obs, action, args)


    def jit(self):
        if self.args.jit:
            self.actor      = jax.jit(self._actor)
            self.critic     = jax.jit(self._critic)
            self.step       = jax.jit(self._step)
            self.td_loss    = jax.jit(self._td_loss)
        else:
            self.actor      = self._actor
            self.critic     = self._critic
            self.step       = self._step
            self.td_loss    = self._td_loss


    def critic_loss(self, critic_params, state, batch, seed):
        seed1, seed2, seed3 = jax.random.split(seed, num=3)

        actor_info     = self.actor_model_apply(state.target.actor, state.actor.sn_state, seed1, batch['next_obs'], self.args.critic_noise, .3)[0]
        next_action    = actor_info['action']
        # next_log_prob  = actor_info['log_prob']
        target_info    = self.critic_model_apply(state.target.critic, state.critic.sn_state, seed2, batch['next_obs'], next_action)[0]
        target_Q       = target_info.pop('value').mean(0)
        target_std     = target_info.pop('std').mean(0)

        # target_Q      -= self.args.alpha * next_log_prob
        # target_Q      -= target_spread * next_log_prob
        target_Q += self.args.c * target_std

        discount = self.args.discount ** self.args.nstep
        done     = 1. - batch['done']
        target_Q = done * discount * target_Q
        target_Q = batch['rew'] + target_Q

        online_info, sn_critic_state = self.critic_model_apply(critic_params, state.critic.sn_state, seed3, batch['obs'], batch['act'], update_stats=True)
        online_Q                     = online_info.pop('value')
        online_std                   = online_info.pop('std')

        # beta-nll, std network
        std_sg      = jax.lax.stop_gradient(online_std)
        td_loss     = online_Q - target_Q
        critic_loss = jnp.log(online_std) + .5 * (td_loss / online_std) ** 2.
        critic_loss = std_sg * critic_loss

        critic_loss_mean = critic_loss.mean()

        aux = {
            'td_loss'        : td_loss.mean(),
            'sn_critic_state': sn_critic_state,
            'critic_loss'    : critic_loss_mean,

            'target_Q_mean'  : target_Q.mean(),
            'online_Q'       : online_Q.mean(1),

            'online_std'     : online_std.mean(),
            'target_std'     : target_std.mean(),
        }
        return critic_loss_mean, aux


    def _td_loss(self, state, batch, seed):
        seed1, seed2, seed3, seed4 = jax.random.split(seed, num=4)

        next_action = self.actor_model_apply(state.actor.params, state.actor.sn_state, seed1, batch['next_obs'], self.args.critic_noise, .3)[0]['action']
        target_info = self.critic_model_apply(state.target.critic, state.critic.sn_state, seed2, batch['next_obs'], next_action)[0]
        target_Q    = target_info.pop('value').mean(0)
        target_std  = target_info.pop('std').mean(0)
        target_Q   += self.args.c * target_std

        discount = self.args.discount ** self.args.nstep
        done     = 1. - batch['done']
        target_Q = done * discount * target_Q
        target_Q = batch['rew'] + target_Q

        online_info, sn_critic_state = self.critic_model_apply(state.critic.params, state.critic.sn_state, seed3, batch['obs'], batch['act'], update_stats=True)
        online_Q                     = online_info.pop('value')
        online_std                   = online_info.pop('std')

        aux = {
            'target_Q'  : target_Q,
            'online_Q'  : online_Q,
            'online_std': online_std,
        }
        return aux
