# Copyright 2023 David Yu-Tung Hui.
# Copyright 2018 DeepMind Technologies Limited. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Adapted from
https://github.com/deepmind/acme/blob/master/acme/agents/tf/mog_mpo/networks.py#L60
https://github.com/deepmind/acme/blob/f8a4edbcb81165b4cd93fd7926b879f0e7fbfc49/acme/tf/networks/distributional.py#L242
https://github.com/deepmind/acme/blob/f8a4edbcb81165b4cd93fd7926b879f0e7fbfc49/acme/tf/networks/distributional.py#L142
"""

import jax
from jax import numpy as jnp
from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions
tfb = tfp.bijectors

from policies_cont.networks.mog_critic import MoGCritic as Critic
from policies_cont.networks.actor      import Actor
from policies_cont.agents.DDPG         import DDPG


class MoG_DDPG(DDPG):
    def __init__(self, obs, action, args):
        obs_dim         = obs.shape[0]
        action_dim      = action.shape[0]
        self.action_dim = action_dim
        self.actor_fn   = lambda obs, temp, clip: Actor(action_dim, args.expl_noise, args.actor, args.actor_groups)(obs, temp, clip)
        if args.init_scale == 0:
            scale = 1
        else:
            scale = args.init_scale
        self.critic_fn = lambda obs, action: Critic(obs_dim+action_dim, args.critic, args.critic_groups, args.num_components, scale)(obs, action)
        self.initialize_all(obs, action, args)


    def to_distribution(self, mus, stdevs, logits):
        if self.args.num_components == 1:
            base = tfd.Normal(loc=mus[:, 0], scale=stdevs[:, 0])
        else:
            base = tfd.MixtureSameFamily(
                mixture_distribution    = tfd.Categorical(logits=logits),
                components_distribution = tfd.Normal(loc=mus, scale=stdevs)
            )
        return tfd.Independent(base)


    def critic_loss(self, critic_params, state, batch, seed):
        seed1, seed2, seed3, seed4, seed5 = jax.random.split(seed, num=5)
        next_action = self.actor_model_apply(state.target.actor, state.actor.sn_state, seed1, batch['next_obs'], self.args.critic_noise, .3)[0]['action']
        target_info = self.critic_model_apply(state.target.critic, state.critic.sn_state, seed2, batch['next_obs'], next_action)[0]

        mus      = target_info.pop('mus').mean(0)
        stdevs   = target_info.pop('stdevs').mean(0)
        logits   = target_info.pop('logits').mean(0)
        if self.args.init_scale == 0:
            target_Q = self.to_distribution(mus, stdevs, logits).mean()
        else:
            target_Q = self.to_distribution(mus, stdevs, logits).sample(20, seed=seed4)
        discount = self.args.discount ** self.args.nstep
        done     = 1. - batch['done']
        target_Q = batch['rew'] + done * discount * target_Q

        online_info, sn_critic_state = self.critic_model_apply(critic_params, state.critic.sn_state, seed3, batch['obs'], batch['act'], update_stats=True)

        mus      = online_info.pop('mus').mean(0)
        stdevs   = online_info.pop('stdevs').mean(0)
        logits   = online_info.pop('logits').mean(0)
        online_Q = self.to_distribution(mus, stdevs, logits)

        critic_loss      = - online_Q.log_prob(target_Q)
        critic_loss_mean = critic_loss.mean()
        critic_loss_std  = critic_loss.std()

        aux = {
            'sn_critic_state': sn_critic_state,
            'critic_loss'    : critic_loss_mean,
            'critic_loss_std': critic_loss_std,

            'target_Q_mean': target_Q.mean(),
            'online_Q'     : online_Q.mean().mean(),
            'target_Q_std' : target_Q.std(),

            # **aux
            # **self.critic_info(online_info, target_info, done.squeeze()[None, :], discount)
        }
        return critic_loss_mean, aux


    def actor_loss(self, actor_params, state, batch, seed):
        seed1, seed2         = jax.random.split(seed)
        info, sn_actor_state = self.actor_model_apply(actor_params, state.actor.sn_state, seed1, batch['obs'], self.args.actor_noise, .3, update_stats=True)
        action               = info.pop('action')

        critic_info = self.critic_model_apply(state.critic.params, state.critic.sn_state, seed2, batch['obs'], action)[0]
        mus         = critic_info.pop('mus').mean(0)
        stdevs      = critic_info.pop('stdevs').mean(0)
        logits      = critic_info.pop('logits').mean(0)
        actor_loss  = - self.to_distribution(mus, stdevs, logits).mean()

        action_diff     = (action - batch['act']) ** 2.
        action_diff_l1  = jnp.abs(action - batch['act'])
        actor_loss_mean = actor_loss.mean()
        actor_loss_std  = actor_loss.std()
        aux             = {
            'action_diff'    : action_diff.mean(),
            'action_diff_l1' : action_diff_l1.mean(),
            'sn_actor_state' : sn_actor_state,
            'actor_loss_mean': actor_loss_mean,
            'actor_loss_std' : actor_loss_std,
            # 'layers'         : info,
        }
        return actor_loss_mean, aux


    def online_critic(self, state):
        action      = self.select_action(state)
        state       = state[None]
        action      = action[None]
        critic_info = self.critic(self.state.critic.params, self.state.critic.sn_state, self.rngs.get_key(), state, action)[0]
        mus         = critic_info.pop('mus').mean(0)
        stdevs      = critic_info.pop('stdevs').mean(0)
        logits      = critic_info.pop('logits').mean(0)
        q           = self.to_distribution(mus, stdevs, logits).mean().mean()
        info        = {'online_expected': q}
        return info


    def target_critic(self, state):
        action      = self.select_action(state)
        state       = state[None]
        action      = action[None]
        critic_info = self.critic(self.state.target.critic, self.state.critic.sn_state, self.rngs.get_key(), state, action)[0]
        mus         = critic_info.pop('mus').mean(0)
        stdevs      = critic_info.pop('stdevs').mean(0)
        logits      = critic_info.pop('logits').mean(0)
        q           = self.to_distribution(mus, stdevs, logits).mean().mean()
        return q
