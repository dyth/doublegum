import haiku as hk
from jax import nn

from policies_disc.networks.network import Network


class Duelling(Network):
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

        pre_value = hk.Linear(256, w_init=self.init, name='pre_value')(x)
        if self.num_groups:
            pre_value = hk.GroupNorm(self.num_groups, axis=-1, create_scale=False, create_offset=False, name=f'prevalue_nosn_ln')(pre_value)
        pre_value = nn.relu(pre_value)
        value     = hk.Linear(1, w_init=self.init, name='value', with_bias=False)(pre_value)

        pre_advantage = hk.Linear(256, w_init=self.init, name='pre_advantage')(x)
        if self.num_groups:
            pre_advantage = hk.GroupNorm(self.num_groups, axis=-1, create_scale=False, create_offset=False, name=f'preadvantage_nosn_ln')(pre_advantage)
        pre_advantage = nn.relu(pre_advantage)
        advantage = hk.Linear(self.action_dim, w_init=self.init, name='advantage', with_bias=False)(pre_advantage)

        info['value']     = value
        info['advantage'] = advantage
        info['q']         = value + advantage - advantage.mean(-1, keepdims=True)
        return info
