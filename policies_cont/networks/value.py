import haiku as hk
from jax import nn
from jax import numpy as jnp


class Value(hk.Module):
    def __init__(self, input_dim, structure, num_groups):
        super().__init__()
        self.input_dim   = input_dim
        self.structure   = structure
        self.num_groups  = num_groups
        self.init        = hk.initializers.Orthogonal(jnp.sqrt(2.))

    def __call__(self, state):
        info = {}

        x = state
        info['input'] = x

        for i, width in enumerate(self.structure):
            name = f'layer{i}'
            x = hk.Linear(width, w_init=self.init, name=name)(x)
            if self.num_groups:
                x = hk.GroupNorm(self.num_groups, axis=-1, create_scale=False, create_offset=False, name=f'{name}_nosn_ln')(x)

            info[name] = x
            x = nn.relu(x)

        info['value'] = hk.Linear(1, w_init=self.init, name='layerout')(x)

        spread         = hk.Linear(1, w_init=self.init, name='spread')(x)
        info['spread'] = nn.softplus(spread) + 1e-5

        scalar_spread         = hk.get_parameter("scalar_spread", [1], x.dtype, init=jnp.zeros)
        info['scalar_spread'] = nn.softplus(scalar_spread) + 1e-5
        return info
