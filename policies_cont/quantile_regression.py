'''
Quantile Regression, adapted from RLax
https://github.com/deepmind/rlax/blob/master/rlax/_src/value_learning.py#L816#L871
'''
import chex
import jax
import jax.numpy as jnp

Array   = chex.Array
Numeric = chex.Numeric


def quantile_regression_loss(
    dist_src: Array,
    tau_src: Array,
    dist_target: Array,
    huber_param: float = 1.
) -> Numeric:
    """Compute (Huber) QR loss between two discrete quantile-valued distributions.
    See "Distributional Reinforcement Learning with Quantile Regression" by
    Dabney et al. (https://arxiv.org/abs/1710.10044).
    Args:
        dist_src: source probability distribution.
        tau_src: source distribution probability thresholds.
        dist_target: target probability distribution.
        huber_param: Huber loss parameter, defaults to 0 (no Huber loss).
    Returns:
        Quantile regression loss.
    """
    chex.assert_rank([dist_src, tau_src, dist_target], 1)
    chex.assert_type([dist_src, tau_src, dist_target], float)

    # Calculate quantile error.
    delta     = dist_target[None, :] - dist_src[:, None]
    delta_neg = (delta < 0.).astype(jnp.float32)
    delta_neg = jax.lax.stop_gradient(delta_neg)
    weight    = jnp.abs(tau_src[:, None] - delta_neg)

    # Use MSE instead of Huber loss
    loss = delta ** 2.
    loss *= weight

    # Average over target-samples dimension, sum over src-samples dimension.
    return jnp.sum(jnp.mean(loss, axis=-1))


batch_quantile_regression_loss = jax.vmap(quantile_regression_loss, in_axes=(0, None, 0))
