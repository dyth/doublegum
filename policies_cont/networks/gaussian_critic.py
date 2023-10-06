import jax
import haiku as hk
from jax import nn
from jax import numpy as jnp


def mish(x):
    return x * nn.tanh(nn.softplus(x))


class Critic(hk.Module):
    def __init__(self, input_dim, structure, num_groups):
        super().__init__()
        self.input_dim  = input_dim
        self.structure  = structure
        self.num_groups = num_groups
        self.init       = hk.initializers.Orthogonal(jnp.sqrt(2.))
        self.out_init   = hk.initializers.Orthogonal(jnp.sqrt(1.))


    def __call__(self, state, action):
        info = {}

        x = jnp.concatenate([state, action], axis=-1)
        info['input'] = x

        for i, width in enumerate(self.structure):
            name = f'layer{i}'
            x = hk.Linear(width, w_init=self.init, name=name)(x)
            if self.num_groups:
                x = hk.GroupNorm(self.num_groups, -1, create_scale=False, create_offset=False, name=f'ln{i}_nosn')(x)
            info[name] = x
            x = nn.relu(x)
            # x = mish(x)

        info['value'] = hk.Linear(1, w_init=self.init, name='value')(x)

        # x           = jax.lax.stop_gradient(x)
        # std         = hk.Linear(1, w_init=self.out_init, name='std')(x)
        std         = hk.Linear(1, w_init=self.init, name='std')(x)
        info['std'] = nn.softplus(std) + 1e-5

        # neg_std         = hk.Linear(1, w_init=self.out_init, name='neg_std')(x)
        neg_std         = hk.Linear(1, w_init=self.init, name='neg_std')(x)
        info['neg_std'] = nn.softplus(neg_std) + 1e-5

        offset         = hk.get_parameter("offset", [1], x.dtype, init=jnp.zeros)
        info['offset'] = offset
        return info
