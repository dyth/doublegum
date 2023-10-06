import haiku as hk
import jax
from jax import nn
from jax import numpy as jnp

from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions
tfb = tfp.bijectors

LOG_STD_MIN = -10.
LOG_STD_MAX = 2.


class SquashedGaussianActor(hk.Module):
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
        std         = hk.Linear(self.action_dim, w_init=self.out_init, name='std')(x)
        # std         = .1 * jnp.ones(log_std.shape)                                                   # fixed std
        # std         = jnp.exp(jnp.clip(std, LOG_STD_MIN, LOG_STD_MAX))                               # clipped
        # std         = jnp.exp(jnp.clip(std, jnp.exp(LOG_STD_MIN), jnp.exp(LOG_STD_MAX)))             # clipped
        # std         = jnp.exp(LOG_STD_MIN + .5 * (LOG_STD_MAX - LOG_STD_MIN) * (1 + nn.tanh(std)))   # smooth clipping
        # std         = nn.softplus(std) + 1e-6                                                        # softplus
        std         = self.std
        # info['mu']  = 2. * nn.tanh(mu)
        info['mu']  = mu
        info['std'] = std

        # from https://github.com/dyth/pg-scale/commit/3fb38d525f037ff87f8c87a2d4fa9c6c92e80558
        # dist = tfd.TransformedDistribution(
        #     distribution = tfd.MultivariateNormalDiag(loc=mu, scale_diag=temp*std),
        #     bijector     = tfb.Tanh()
        # )
        # action           = dist.sample(seed=hk.next_rng_key())
        # info['action']   = action
        # info['log_prob'] = dist.log_prob(action)

        # manual log prob inspired by faithful
        # dist           = tfd.MultivariateNormalDiag(loc=mu, scale_diag=temp*std)
        # z              = dist.sample(seed=hk.next_rng_key())
        # info['action'] = nn.tanh(z)
        #
        # def log_prob(z, mu, std):
        #     # mu  = jax.lax.stop_gradient(mu)
        #     # std = jax.lax.stop_gradient(std)
        #     lp  = -.5 * ((z - mu) / std) ** 2
        #     lp -= .5 * jnp.log(2 * jnp.pi) + jnp.log(std)
        #     lp -= 2 * (jnp.log(2) - z - nn.softplus(-2 * z))
        #     return lp.sum(-1)
        #
        # def mu_loss(z, mu, std):
        #     z  = jax.lax.stop_gradient(z)
        #     ml = (z - mu) ** 2
        #     return ml.sum(-1)
        #
        # def sigma_loss(z, mu, std):
        #     z  = jax.lax.stop_gradient(z)
        #     mu = jax.lax.stop_gradient(mu)
        #     sl = - jnp.log(std) -.5 * ((z - mu) / std) ** 2
        #     return sl.sum(-1)
        #
        # info['log_prob'] = log_prob(z, mu, std) #+ mu_loss(z, mu, std) + sigma_loss(z, mu, std)
        # return info


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

        # # manual log prob
        # dist           = tfd.MultivariateNormalDiag(loc=mu, scale_diag=temp*std)
        # z              = dist.sample(seed=hk.next_rng_key())
        # # z              = z.clip(mu-clip*std, mu+clip*std)
        # info['action'] = nn.tanh(z)
        #
        # def log_prob(z, mu, std):
        #     lp      = -.5 * ((z - mu) / std) ** 2.
        #     norm    = .5 * jnp.log(2. * jnp.pi) + jnp.log(std)
        #     logvf   = 2. * (jnp.log(2.) - z - nn.softplus(-2. * z))
        #     lp      = lp - norm + logvf
        #     lp_beta = lp * jax.lax.stop_gradient(std)
        #     return lp.sum(-1), lp_beta.sum(-1)
        #
        # lp, beta_lp           = log_prob(z, mu, std)
        # info['log_prob']      = lp
        # info['beta_log_prob'] = beta_lp
        return info
