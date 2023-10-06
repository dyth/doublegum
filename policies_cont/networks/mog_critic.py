'''
Implementation Notes


https://github.com/deepmind/acme/blob/master/acme/agents/tf/mog_mpo/networks.py#L60
Critic Architecture:
num_dimensions = 1
num_components
init_scale


https://github.com/deepmind/acme/blob/f8a4edbcb81165b4cd93fd7926b879f0e7fbfc49/acme/tf/networks/distributional.py#L242
UnivariateGaussianMixture:
init_scale
num_components
num_dimensions = 1
multivariate=False


https://github.com/deepmind/acme/blob/f8a4edbcb81165b4cd93fd7926b879f0e7fbfc49/acme/tf/networks/distributional.py#L142
self._scale_factor = init_scale / tf.nn.softplus(0.)
logits_size        = self._num_dimensions * self._num_components

w_init = tf.initializers.VarianceScaling(1e-5)
self._logit_layer = snt.Linear(logits_size, w_init=w_init)
self._loc_layer   = snt.Linear(self._num_dimensions * self._num_components, w_init=w_init)
self._scale_layer = snt.Linear(self._num_dimensions * self._num_components, w_init=w_init)

when called:
logits = self._logit_layer(inputs)
locs   = self._loc_layer(inputs)
scales = self._scale_layer(inputs)

scales = self._scale_factor * tf.nn.softplus(scales) + _MIN_SCALE

shape = [-1, self._num_dimensions, self._num_components]

locs   = tf.reshape(locs, shape)
scales = tf.reshape(scales, shape)
components_distribution = tfd.Normal(loc=locs, scale=scales)

logits = tf.reshape(logits, shape)

# Create the mixture distribution.
distribution = tfd.MixtureSameFamily(
    mixture_distribution=tfd.Categorical(logits=logits),
    components_distribution=components_distribution
)

distribution = tfd.Independent(distribution)
'''

import haiku as hk
from jax import nn
from jax import numpy as jnp


class MoGCritic(hk.Module):
    def __init__(self, input_dim, structure, num_groups, num_components, init_scale):
        super().__init__()
        self.structure      = structure
        self.num_components = num_components
        self.init_scale     = init_scale
        self.num_groups     = num_groups
        self.init           = hk.initializers.Orthogonal(jnp.sqrt(2.))
        self.out_init       = hk.initializers.Orthogonal(1.)


    def __call__(self, state, action):
        info = {}
        x = jnp.concatenate([state, action], axis=-1)
        info['input'] = x

        for i, width in enumerate(self.structure):
            name = f'layer{i}'
            x = hk.Linear(width, w_init=self.init, name=name)(x)
            if self.num_groups:
                x = hk.GroupNorm(self.num_groups, axis=-1, create_scale=False, create_offset=False, name=f'{name}_nosn_ln')(x)

            info[name] = x
            x = nn.relu(x)

        shape = [-1, 1, self.num_components]

        mus = hk.Linear(self.num_components, w_init=self.out_init, name='mus')(x)
        mus = mus.reshape(shape)
        info['mus'] = mus

        stdevs = hk.Linear(self.num_components, w_init=self.out_init, name='stdevs')(x)
        stdevs = self.init_scale * nn.softplus(stdevs) / nn.softplus(0.) + 1e-4
        stdevs = stdevs.reshape(shape)
        info['stdevs'] = stdevs

        logits = hk.Linear(self.num_components, w_init=self.out_init, name='logits')(x)
        logits = logits.reshape(shape)
        info['logits'] = logits

        return info
