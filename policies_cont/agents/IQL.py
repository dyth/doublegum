import haiku as hk
import optax
import jax
from jax import numpy as jnp

from policies_cont             import utils
from policies_cont.agents.DDPG import DDPG, CriticState


class IQL(DDPG):
    '''
    Implicit Q-Learning, from (https://arxiv.org/abs/2110.06169)
    '''
    def initialize_critic(self):
        self.critic_model = hk.transform(self.critic_fn)
        rngs              = jax.random.split(self.rngs.get_key(), self.args.ensemble)
        critic_params     = jax.vmap(lambda r: self.critic_model.init(r, self.obs, self.action))(rngs)
        if self.args.critic_sn:
            self.critic_sn     = hk.without_apply_rng(hk.transform_with_state(utils.sn))
            _, sn_critic_state = self.critic_sn.init(self.rngs.get_key(), critic_params)
        else:
            sn_critic_state = {}
        self.critic_opt   = optax.chain(
            optax.clip_by_global_norm(40.),
            optax.adam(self.args.critic_lr),
        )
        critic_opt_state  = self.critic_opt.init(critic_params)
        return CriticState(critic_params, sn_critic_state, critic_opt_state)


    def critic_loss(self, critic_params, state, batch, seed):
        seed1, seed2, seed3 = jax.random.split(seed, num=3)
        actor_info     = self.actor_model_apply(state.target.actor, state.actor.sn_state, seed1, batch['next_obs'], self.args.critic_noise, .3)[0]
        next_action    = actor_info['action']
        target_info    = self.critic_model_apply(state.target.critic, state.critic.sn_state, seed2, batch['next_obs'], next_action)[0]
        # target_Q       = target_info.pop('value').mean(0)
        target_Q       = target_info.pop('value').sort(0)[self.args.pessimism]

        online_info, sn_critic_state = self.critic_model_apply(critic_params, state.critic.sn_state, seed3, batch['obs'], batch['act'], update_stats=True)
        online_Q                     = online_info.pop('value')

        discount  = self.args.discount ** self.args.nstep
        done      = 1. - batch['done']
        target_Q  = done * discount * target_Q
        target_Q  = batch['rew'] + target_Q

        td_loss  = online_Q - target_Q
        positive = (self.args.beta + 1e-5) * jnp.maximum(td_loss, 0.) ** 2.
        negative = (1. - self.args.beta + 1e-5) * jnp.minimum(td_loss, 0.) ** 2.

        critic_loss      = negative + positive
        critic_loss_mean = critic_loss.mean()
        critic_loss_std  = critic_loss.std()

        aux = {
            'td_loss'        : td_loss.mean(),
            'td_loss_std'    : td_loss.std(),
            'sn_critic_state': sn_critic_state,
            'critic_loss'    : critic_loss_mean,
            'critic_loss_std': critic_loss_std,

            'target_Q_mean'  : target_Q.mean(),
            'online_Q'       : online_Q.mean(1),
            'target_Q_std'   : target_Q.std(),
            'online_Q_std'   : online_Q.std(1),
        }
        return critic_loss_mean, aux
