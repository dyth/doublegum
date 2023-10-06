import jax
from jax import numpy as jnp
from policies_cont.networks.actor           import Actor
from policies_cont.networks.quantile_critic import QuantileCritic as Critic
from policies_cont.agents.TD3               import TD3
from policies_cont.quantile_regression      import batch_quantile_regression_loss


class QR_DDPG(TD3):
    def __init__(self, obs, action, args):
        obs_dim        = obs.shape[0]
        action_dim     = action.shape[0]
        num_quantiles  = 201
        self.actor_fn  = lambda obs, temp, clip: Actor(action_dim, args.expl_noise, args.actor, args.actor_groups)(obs, temp, clip)
        self.critic_fn = lambda obs, action: Critic(obs_dim+action_dim, args.critic, args.critic_groups, num_quantiles=num_quantiles)(obs, action)
        self.initialize_all(obs, action, args)
        self.quantiles = (jnp.arange(0, num_quantiles) + .5) / float(num_quantiles)


    def critic_loss(self, critic_params, state, batch, seed):
        seed1, seed2, seed3 = jax.random.split(seed, num=3)
        next_action   = self.actor_model_apply(state.target.actor, state.actor.sn_state, seed1, batch['next_obs'], self.args.critic_noise, .3)[0]['action']
        target_info   = self.critic_model_apply(state.target.critic, state.critic.sn_state, seed2, batch['next_obs'], next_action)[0]
        target_Q      = target_info.pop('value')
        target_Q      = target_Q.sort(0)[self.args.pessimism]

        discount = self.args.discount ** self.args.nstep
        done     = 1. - batch['done']
        target_Q = done * discount * target_Q
        target_Q = batch['rew'] + target_Q

        online_info, sn_critic_state = self.critic_model_apply(critic_params, state.critic.sn_state, seed3, batch['obs'], batch['act'], update_stats=True)
        online_Q                     = online_info.pop('value')

        critic_loss = jax.vmap(lambda oQ: batch_quantile_regression_loss(oQ.squeeze(), self.quantiles, target_Q.squeeze()))(online_Q)

        critic_loss_mean = critic_loss.mean()
        critic_loss_std  = critic_loss.std()

        aux = {
            'sn_critic_state': sn_critic_state,
            'critic_loss'    : critic_loss_mean,
            'critic_loss_std': critic_loss_std,

            'target_Q_mean'  : target_Q.mean(),
            # 'online_Q'       : online_Q.mean(1),
            'target_Q_std'   : target_Q.std(),
            # 'online_Q_std'   : online_Q.std(1),
        }
        return critic_loss_mean, aux


    def get_target(self, mc_batch):
        target_info = self.critic_model_apply(self.state.target.critic, self.state.critic.sn_state, self.rngs.get_key(), mc_batch['obs'], mc_batch['act'])[0]
        target_Q    = target_info.pop('value')
        target_Q    = target_Q.sort(0)[self.args.pessimism]
        return target_Q.mean(-1)
