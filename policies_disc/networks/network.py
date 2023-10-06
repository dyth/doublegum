import haiku as hk
from jax import nn
from jax import numpy as jnp


class Network(hk.Module):
    def __init__(self, action_dim, structure, num_groups):
        super().__init__()
        self.action_dim  = action_dim
        self.structure   = structure
        self.num_groups  = num_groups
        self.init        = hk.initializers.Orthogonal(jnp.sqrt(2.))
        self.out_init    = hk.initializers.Orthogonal(1.)

    def __call__(self, state):
        info = {}
        x             = state
        info['input'] = x
        for i, width in enumerate(self.structure):
            name = f'layer{i}'
            x = hk.Linear(width, w_init=self.init, name=name)(x)
            if self.num_groups:
                x = hk.GroupNorm(self.num_groups, axis=-1, create_scale=False, create_offset=False, name=f'{name}_nosn_ln')(x)

            info[name] = x
            x = nn.relu(x)

        info['q'] = hk.Linear(self.action_dim, w_init=self.out_init, name='q')(x)
        return info
