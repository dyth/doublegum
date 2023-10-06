import haiku as hk
from jax import nn
from jax import numpy as jnp

from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions
tfb = tfp.bijectors


class Actor(hk.Module):
    def __init__(self, action_dim, noise, structure, num_groups, **kwargs):
        super().__init__(**kwargs)
        self.action_dim = action_dim
        self.std        = noise * jnp.ones(action_dim)
        self.structure  = structure
        self.num_groups = num_groups
        self.init       = hk.initializers.Orthogonal(jnp.sqrt(2.))
        self.out_init   = hk.initializers.Orthogonal(1.)

    def __call__(self, x, temp, clip):
        info = {}
        info['input'] = x

        for i, s in enumerate(self.structure):
            name = f'linear{i}'
            x    = hk.Linear(s, w_init=self.init, name=name)(x)
            if self.num_groups:
                x = hk.GroupNorm(self.num_groups, -1, create_scale=False, create_offset=False, name=f'ln{i}_nosn')(x)
            info[name] = x
            x = nn.relu(x)

        x = hk.Linear(self.action_dim, w_init=self.out_init, name='preact')(x)
        info['preact'] = x
        x = nn.tanh(x)

        # noise = jrandom.normal(hk.next_rng_key(), x.shape) * self.std
        # noise = noise.clip(-clip, clip)
        #
        # # scale
        # info['log_prob'] = - jnp.log(self.std * jnp.sqrt(2 * jnp.pi)) - .5 * (noise / self.std) ** 2
        #
        # action         = x + temp * noise
        # action         = action.clip(-1., 1.)
        # info['action'] = action

        dist             = tfd.MultivariateNormalDiag(loc=jnp.zeros_like(x), scale_diag=self.std)
        noise            = dist.sample(seed=hk.next_rng_key())
        # noise            = noise.clip(-clip, clip)
        info['log_prob'] = dist.log_prob(temp*noise)
        info['beta_log_prob'] = dist.log_prob(temp*noise)
        info['std']      = self.std
        info['mu']       = x
        info['prob']     = dist.prob(temp*noise)
        noise            = noise.clip(-clip, clip)
        action           = x + temp * noise
        action           = action.clip(-1., 1.)
        info['action']   = action


        # SACLite
        # dist = tfd.TransformedDistribution(
        #     distribution = tfd.MultivariateNormalDiag(loc=mu, scale_diag=temp*std),
        #     bijector     = tfb.Tanh()
        # )
        # action           = dist.sample(seed=hk.next_rng_key())
        # info['action']   = action
        # info['log_prob'] = dist.log_prob(action)


        # SACLite, Manual
        # dist           = tfd.MultivariateNormalDiag(loc=mu, scale_diag=temp*std)
        # z              = dist.sample(seed=hk.next_rng_key())
        # info['action'] = nn.tanh(z)
        # def log_prob(z, mu, std):
        #     # mu  = jax.lax.stop_gradient(mu)
        #     # std = jax.lax.stop_gradient(std)
        #     lp  = -.5 * ((z - mu) / std) ** 2
        #     lp -= .5 * jnp.log(2 * jnp.pi) + jnp.log(std)
        #     lp -= 2 * (jnp.log(2) - z - nn.softplus(-2 * z))
        #     return lp.sum(-1)
        # info['log_prob'] = log_prob(z, mu, std)

        return info
