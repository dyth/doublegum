'''
Last line of Appendix D.6: we found TD3 performed marginally better overall
Section 4.2: In all environments expect cheetah-run, XQL-DQ performs better than standard TD3
We swept over values 3, 4, 10, 20 (D.6) (single)
For we swept over 1, 2, 5 (Twin)
In online experiemnts we used an exponential clip value of 8.
They restarted runs that became unstable during training
'''

import copy
import jax
import haiku as hk
import optax
from jax import numpy as jnp
from collections import namedtuple

from policies_cont                 import utils
from policies_cont.agents.DDPG     import DDPG, CriticState
from policies_cont.networks.critic import Critic
from policies_cont.networks.actor  import Actor
# from policies.networks.squashed_gaussian_actor import SquashedGaussianActor as Actor


State       = namedtuple('State',  ['actor', 'critic', 'target'])
TargetState = namedtuple('Target', ['actor', 'critic'])


class XQL(DDPG):
    def __init__(self, obs, action, args):
        obs_dim        = obs.shape[0]
        action_dim     = action.shape[0]
        self.actor_fn  = lambda obs, temp, clip: Actor(action_dim, args.expl_noise, args.actor, args.actor_groups)(obs, temp, clip)
        self.critic_fn = lambda obs, action: Critic(obs_dim+action_dim, args.critic, args.critic_groups)(obs, action)
        self.initialize_all(obs, action, args)

    def initialize_all(self, obs, action, args):
        self.rngs    = utils.PRNGKeys(args.seed)
        self.args    = args
        self.obs     = obs[None, :]
        self.action  = action[None, :]
        actor_state  = self.initialize_actor()
        critic_state = self.initialize_critic()
        target_state = self.initialize_target(actor_state, critic_state)
        self.state   = State(actor_state, critic_state, target_state)
        self.jit()


    def initialize_critic(self):
        self.critic_model = hk.transform(self.critic_fn)
        rngs              = jax.random.split(self.rngs.get_key(), self.args.ensemble) # default 2 because of Twin Critics
        critic_params     = jax.vmap(lambda r: self.critic_model.init(r, self.obs, self.action))(rngs)
        if self.args.critic_sn:
            self.critic_sn     = hk.without_apply_rng(hk.transform_with_state(utils.sn))
            _, sn_critic_state = self.critic_sn.init(self.rngs.get_key(), critic_params)
        else:
            sn_critic_state = {}
        self.critic_opt   = optax.adam(self.args.critic_lr)
        critic_opt_state  = self.critic_opt.init(critic_params)
        return CriticState(critic_params, sn_critic_state, critic_opt_state)


    def initialize_target(self, actor_state, critic_state):
        return TargetState(copy.deepcopy(actor_state.params), copy.deepcopy(critic_state.params))


    def critic_loss(self, critic_params, state, batch, seed):
        state = jax.lax.stop_gradient(state)
        seed1, seed2, seed3 = jax.random.split(seed, num=3)

        next_action = self.actor_model_apply(state.target.actor, state.actor.sn_state, seed1, batch['next_obs'], self.args.critic_noise, .3)[0]['action']
        target_info = self.critic_model_apply(state.target.critic, state.critic.sn_state, seed2, batch['next_obs'], next_action)[0]
        # target_info = self.critic_model_apply(critic_params, state.critic.sn_state, seed2, batch['next_obs'], next_action)[0]
        target_Q    = target_info.pop('value')
        target_Q    = target_Q.sort(0)[0]
        discount    = self.args.discount ** self.args.nstep
        done        = 1. - batch['done']
        target_Q    = done * discount * target_Q
        target_Q    = batch['rew'] + target_Q

        online_info, sn_critic_state = self.critic_model_apply(critic_params, state.critic.sn_state, seed3, batch['obs'], batch['act'], update_stats=True)
        online_Q                     = online_info.pop('value')

        # td_loss     = -(target_Q - online_Q)

        # LINEX
        z           = (target_Q - online_Q) / self.args.beta
        z           = jnp.clip(z, a_min=-8, a_max=8)
        max_z       = jnp.max(z)
        max_z       = jnp.where(max_z < -1.0, -1.0, max_z)
        max_z       = jax.lax.stop_gradient(max_z)
        critic_loss = jnp.exp(z - max_z) - z * jnp.exp(-max_z) - jnp.exp(-max_z)

        # # Generalized Moment Matching
        # # online_spread = online_info.pop('scalar_spread')
        # online_spread = online_info.pop('spread')
        # online_mean   = online_loc - jnp.euler_gamma * online_spread
        # td_loss       = online_mean - target_loc
        # online_std    = online_spread * jnp.pi / jnp.sqrt(6)
        # std_sg        = jax.lax.stop_gradient(online_std)
        # value_loss    = jnp.log(online_std) + .5 * (td_loss / online_std) ** 2.
        # value_loss    = std_sg * value_loss

        critic_loss_mean = critic_loss.mean()
        critic_loss_std  = critic_loss.std()

        aux = {
            # 'td_loss'        : td_loss.mean(),
            # 'td_loss_std'    : td_loss.std(),
            'sn_critic_state': sn_critic_state,
            'critic_loss'    : critic_loss_mean,
            'critic_loss_std': critic_loss_std,

            'target_Q_mean'  : target_Q.mean(),
            'online_Q'       : online_Q.mean(1),
            'target_Q_std'   : target_Q.std(),
            'online_Q_std'   : online_Q.std(1),
        }
        return critic_loss_mean, aux
