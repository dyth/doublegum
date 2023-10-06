import haiku as hk
from jax import nn
from jax import numpy as jnp

from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions
tfb = tfp.bijectors


class SquashedGaussianActorFixed(hk.Module):
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

        mu          = hk.Linear(self.action_dim, w_init=self.out_init, name='mu')(x)
        std         = self.std
        info['mu']  = mu
        info['std'] = std

        seed = hk.next_rng_key()
        dist = tfd.TransformedDistribution(
            distribution = tfd.MultivariateNormalDiag(loc=mu, scale_diag=temp*std),
            bijector     = tfb.Tanh()
        )
        action = dist.sample(seed=seed)
        # action = action.clip(jnp.tanh(mu-clip*std), jnp.tanh(mu+clip*std))
        lp     = dist.log_prob(action)
        info['action']        = action
        info['log_prob']      = lp
        info['beta_log_prob'] = lp
        return info
