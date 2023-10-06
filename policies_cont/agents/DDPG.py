import copy
import jax
import optax
import numpy as np
import haiku as hk
from jax import numpy as jnp
from collections import namedtuple

from policies_cont                 import utils
from policies_cont.networks.critic import Critic
from policies_cont.networks.actor  import Actor


ActorState  = namedtuple('Actor' , ['params', 'sn_state', 'opt_state'])
CriticState = namedtuple('Critic', ['params', 'sn_state', 'opt_state'])
TargetState = namedtuple('Target', ['actor', 'critic'])
State       = namedtuple('State',  ['actor', 'critic', 'target'])


class DDPG:
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

    def initialize_actor(self):
        self.actor_model = hk.transform(self.actor_fn)
        actor_params     = self.actor_model.init(self.rngs.get_key(), self.obs, 1., 1.)
        if self.args.actor_sn:
            self.actor_sn     = hk.without_apply_rng(hk.transform_with_state(utils.sn))
            _, sn_actor_state = self.actor_sn.init(self.rngs.get_key(), actor_params)
        else:
            sn_actor_state = {}
        self.actor_opt  = optax.adam(self.args.actor_lr)
        actor_opt_state = self.actor_opt.init(actor_params)
        return ActorState(actor_params, sn_actor_state, actor_opt_state)

    def initialize_critic(self):
        self.critic_model = hk.transform(self.critic_fn)
        rngs              = jax.random.split(self.rngs.get_key(), 1) # 1 because there is only 1 network in DDPG
        critic_params     = jax.vmap(lambda r: self.critic_model.init(r, self.obs, self.action))(rngs)
        if self.args.critic_sn:
            self.critic_sn     = hk.without_apply_rng(hk.transform_with_state(utils.sn))
            _, sn_critic_state = self.critic_sn.init(self.rngs.get_key(), critic_params)
        else:
            sn_critic_state = {}
        self.critic_opt  = optax.adam(self.args.critic_lr)
        critic_opt_state = self.critic_opt.init(critic_params)
        return CriticState(critic_params, sn_critic_state, critic_opt_state)

    def initialize_target(self, actor_state, critic_state):
        return TargetState(copy.deepcopy(actor_state.params), copy.deepcopy(critic_state.params))


    def jit(self):
        if self.args.jit:
            self.actor   = jax.jit(self._actor)
            self.critic  = jax.jit(self._critic)
            self.step    = jax.jit(self._step)
            self.td_loss = jax.jit(self._td_loss)
        else:
            self.actor   = self._actor
            self.critic  = self._critic
            self.step    = self._step
            self.td_loss = self._td_loss


    def actor_model_apply(self, actor_params, sn_actor_state, seed, state, temp, clip, update_stats=False):
        if self.args.actor_sn:
            actor_params, sn_actor_state = self.actor_sn.apply(None, sn_actor_state, actor_params, update_stats=update_stats)
        action = self.actor_model.apply(actor_params, seed, state, temp, clip)
        return action, sn_actor_state

    def critic_model_apply(self, critic_params, sn_critic_state, seed, state, action, update_stats=False):
        if self.args.critic_sn:
            critic_params, sn_critic_state = self.critic_sn.apply(None, sn_critic_state, critic_params, update_stats=update_stats)
        q = jax.vmap(self.critic_model.apply, in_axes=(0, None, None, None), out_axes=0)(critic_params, seed, state, action)
        return q, sn_critic_state


    def _actor(self, actor_params, sn_actor_state, seed, state, temp, clip):
        return self.actor_model_apply(actor_params, sn_actor_state, seed, state, temp, clip)

    def _critic(self, critic_params, sn_critic_state, seed, state, action):
        return self.critic_model_apply(critic_params, sn_critic_state, seed, state, action)


    def select_action(self, state, noise=0.):
        state  = state[None]
        seed   = self.rngs.get_key()
        action = self.actor(self.state.actor.params, self.state.actor.sn_state, seed, state, noise, 2.)[0]['action']
        action = np.asarray(action)[0]
        action = action.clip(-1., 1.)
        return action

    def sample_action(self, state):
        return self.select_action(state, float(self.args.explore))

    def online_critic(self, state):
        action = self.select_action(state)
        state  = state[None]
        action = action[None]
        q      = self.critic(self.state.critic.params, self.state.critic.sn_state, self.rngs.get_key(), state, action)[0].pop('value').mean(1)
        info   = {'online_expected': q}
        return info

    def target_critic(self, state):
        action = self.select_action(state)
        state  = state[None]
        action = action[None]
        q      = self.critic(self.state.target.critic, self.state.critic.sn_state, self.rngs.get_key(), state, action)[0].pop('value').mean()
        return q


    def log_td_loss(self, replay_buffer, samples=1000):
        return self.td_loss(self.state, replay_buffer.sample(samples), self.rngs.get_key())


    def _td_loss(self, state, batch, seed):
        seed1, seed2, seed3 = jax.random.split(seed, num=3)

        actor_info  = self.actor_model_apply(state.target.actor, state.actor.sn_state, seed1, batch['next_obs'], self.args.critic_noise, .3)[0]
        next_action = actor_info['action']
        target_info = self.critic_model_apply(state.target.critic, state.critic.sn_state, seed2, batch['next_obs'], next_action)[0]
        target_Q    = target_info.pop('value').mean(0)

        online_info, sn_critic_state = self.critic_model_apply(state.critic.params, state.critic.sn_state, seed3, batch['obs'], batch['act'], update_stats=True)
        online_Q                     = online_info.pop('value')

        discount = self.args.discount ** self.args.nstep
        done     = 1. - batch['done']
        target_Q = done * discount * target_Q
        target_Q = batch['rew'] + target_Q

        aux = {
            'target_Q': target_Q,
            'online_Q': online_Q,
        }
        return aux


    def critic_loss(self, critic_params, state, batch, seed):
        state = jax.lax.stop_gradient(state)
        seed1, seed2, seed3 = jax.random.split(seed, num=3)

        next_action = self.actor_model_apply(state.target.actor, state.actor.sn_state, seed1, batch['next_obs'], self.args.critic_noise, .3)[0]['action']
        target_info = self.critic_model_apply(state.target.critic, state.critic.sn_state, seed2, batch['next_obs'], next_action)[0]
        # target_info = self.critic_model_apply(critic_params, state.critic.sn_state, seed2, batch['next_obs'], next_action)[0]
        target_Q    = target_info.pop('value')
        target_Q    = target_Q.mean(0)
        discount    = self.args.discount ** self.args.nstep
        done        = 1. - batch['done']
        target_Q    = done * discount * target_Q
        target_Q    = batch['rew'] + target_Q

        online_info, sn_critic_state = self.critic_model_apply(critic_params, state.critic.sn_state, seed3, batch['obs'], batch['act'], update_stats=True)
        online_Q                     = online_info.pop('value')

        td_loss     = -(target_Q - online_Q)
        critic_loss = td_loss ** 2.

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

    def critic_step(self, state, batch, rng):
        vgrad_fn               = jax.value_and_grad(self.critic_loss, has_aux=True)
        (_, aux), grad         = vgrad_fn(state.critic.params, state, batch, rng)
        # grad, critic_opt_state = self.critic_opt.update(grad, critic_opt_state)
        grad, critic_opt_state = self.critic_opt.update(grad, state.critic.opt_state, state.critic.params)
        critic_params          = optax.apply_updates(state.critic.params, grad)
        # aux['grads']           = l2_norm(grad)
        # aux['params']          = l2_norm(critic_params)
        critic_state           = CriticState(critic_params, aux.pop('sn_critic_state'), critic_opt_state)
        return aux, critic_state


    def actor_loss(self, actor_params, state, batch, seed):
        seed1, seed2         = jax.random.split(seed)
        info, sn_actor_state = self.actor_model_apply(actor_params, state.actor.sn_state, seed1, batch['obs'], self.args.actor_noise, .3, update_stats=True)
        action               = info.pop('action')
        actor_loss           = -self.critic_model_apply(state.critic.params, state.critic.sn_state, seed2, batch['obs'], action)[0].pop('value')

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

    def actor_step(self, state, batch, seed):
        vgrad_fn              = jax.value_and_grad(self.actor_loss, has_aux=True)
        (_, aux), grad        = vgrad_fn(state.actor.params, state, batch, seed)
        # grad, actor_opt_state = self.actor_opt.update(grad, actor_opt_state)
        grad, actor_opt_state = self.actor_opt.update(grad, state.actor.opt_state, state.actor.params)
        actor_params          = optax.apply_updates(state.actor.params, grad)
        # aux['grads']          = l2_norm(grad)
        # aux['params']         = l2_norm(actor_params)
        actor_state           = ActorState(actor_params, aux.pop('sn_actor_state'), actor_opt_state)
        return aux, actor_state


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
            mini_batch = jax.tree_util.tree_map(lambda t: slice(t, i), batch)

            # critic step
            seed1, seed               = jax.random.split(seed)
            critic_info, critic_state = self.critic_step(state, mini_batch, seed1)
            state                     = state._replace(critic=critic_state)

            critic_target_params      = self.soft_update(state.critic.params, state.target.critic, self.args.tau)
            state                     = state._replace(target=state.target._replace(critic=critic_target_params))

            # actor_step
            actor_info, actor_state = self.actor_step(state, mini_batch, seed)
            state                   = state._replace(actor=actor_state)

            actor_target_params     = self.soft_update(state.actor.params, state.target.actor, self.args.actor_tau)
            state                   = state._replace(target=state.target._replace(actor=actor_target_params))

        return actor_info, critic_info, state


    def train(self, replay_buffer, samples=100):
        replay_samples                      = replay_buffer.sample(samples)
        actor_info, critic_info, self.state = self.step(replay_samples, self.rngs.get_key(), self.state)
        return actor_info, critic_info
