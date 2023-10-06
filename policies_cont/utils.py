import jax
import haiku as hk
import jax.numpy as jnp
from jax import nn
from jax import random as jrandom


class PRNGKeys:
    def __init__(self, seed=0):
        self._key = jrandom.PRNGKey(seed)

    def get_key(self):
        self._key, subkey = jrandom.split(self._key)
        return subkey


def flatten(p, label=None):
    'from https://stackoverflow.com/a/72436805'
    if isinstance(p, dict):
        for k, v in p.items():
            yield from flatten(v, k if label is None else f"{label}.{k}")
    else:
        yield label, p


def l2_norm_all(grads):
    return jnp.linalg.norm(
        jax.tree_util.tree_leaves(
            jax.tree_util.tree_map(jnp.linalg.norm, grads)
        )
    )


def l2_norm(grads):
    return dict(
        flatten(
            jax.tree_util.tree_map(jnp.linalg.norm, grads)
        )
    )


def l2_decay(params):
    return 0.5 * sum(
        jnp.sum(jnp.square(p)) for p in jax.tree_util.tree_leaves(
            hk.data_structures.filter(
                lambda module_name, name, value: name == 'w', params
            )
        )
    )
    # return jnp.sum(
    #     # flatten(
    #         jax.tree_util.tree_map(
    #             jnp.linalg.norm,
    #             params
    #             # hk.data_structures.filter(
    #             #     # lambda *x: 'nosn' not in ''.join(x), params
    #             #     lambda module_name, name, value: 'nosn' not in name, params
    # ))




def linf_norm(grads):
    return jnp.max(
        jnp.stack(
            jax.tree_util.tree_leaves(
                jax.tree_util.tree_map(lambda x: jnp.max(jnp.abs(x)), grads)
            )
        )
    )


def sn(x, update_stats=False):
    return hk.SNParamsTree(ignore_regex='([^?!.]*.b$|[^?!.]*nosn[^?!.]*)', eps=1e-12)(x, update_stats=update_stats)


def mish(x):
    return x * nn.tanh(nn.softplus(x))