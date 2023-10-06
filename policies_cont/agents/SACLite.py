import jax
import optax
from jax import numpy as jnp
from collections import namedtuple

from policies_cont                                  import utils
from policies_cont.networks.critic                  import Critic
from policies_cont.networks.squashed_gaussian_actor import SquashedGaussianActor as Actor
from policies_cont.agents.TD3                       import TD3


TempState   = namedtuple('temp'  , ['temp', 'opt_state'])
State       = namedtuple('State' , ['actor', 'critic', 'target', 'temp'])


class SACLite(TD3):
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
        temp_state   = self.initialize_temp(action)
        self.state   = State(actor_state, critic_state, target_state, temp_state)
        self.jit()


    def actor_loss(self, actor_params, state, batch, seed):
        seed1, seed2         = jax.random.split(seed)
        info, sn_actor_state = self.actor_model_apply(actor_params, state.actor.sn_state, seed1, batch['obs'], self.args.actor_noise, .3, update_stats=True)
        action               = info.pop('action')
        log_prob             = info.pop('log_prob')
        std                  = info.pop('std')
        value                = self.critic_model_apply(state.critic.params, state.critic.sn_state, seed2, batch['obs'], action)[0].pop('value')
        # actor_loss           = jax.lax.stop_gradient(std) * (self.args.alpha * log_prob - value)
        actor_loss           = state.temp.temp * log_prob - value

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
            'log_prob'       : log_prob.mean(),
            'entropy'        : - log_prob,
        }
        return actor_loss_mean, aux


    def initialize_temp(self, action):
        temp           = jnp.array(1., dtype=jnp.float32)
        self.temp_opt  = optax.adam(self.args.critic_lr)
        temp_opt_state = self.temp_opt.init(temp)
        self.target_entropy = - action.shape[0] / 2.
        return TempState(temp, temp_opt_state)


    def temp_loss(self, temp, entropy):
        temp_loss = temp * (entropy - self.target_entropy).mean()
        aux       = {
            'temp' : temp,
        }
        return temp_loss.mean(), aux


    def temp_step(self, state, entropy):
        vgrad_fn             = jax.value_and_grad(self.temp_loss, has_aux=True)
        (_, aux), grad       = vgrad_fn(state.temp.temp, entropy)
        grad, temp_opt_state = self.temp_opt.update(grad, state.temp.opt_state, state.temp.temp)
        temp                 = optax.apply_updates(state.temp.temp, grad)
        temp_state           = TempState(temp, temp_opt_state)
        return aux, temp_state


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

            critic_target_params = self.soft_update(state.critic.params, state.target.critic, self.args.tau)
            state                = state._replace(target=state.target._replace(critic=critic_target_params))

            # actor step
            actor_info, actor_state = self.actor_step(state, mini_batch, seed)
            state                   = state._replace(actor=actor_state)

            actor_target_params = self.soft_update(state.actor.params, state.target.actor, self.args.actor_tau)
            state               = state._replace(target=state.target._replace(actor=actor_target_params))

            # temp step
            temp_info, temp_state = self.temp_step(state, actor_info.pop('entropy'))
            state                 = state._replace(temp=temp_state)

            critic_info = {**critic_info, **temp_info}

        return actor_info, critic_info, state
