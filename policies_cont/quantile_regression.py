'''
Quantile Regression, adapted from RLax
https://github.com/deepmind/rlax/blob/master/rlax/_src/value_learning.py#L816#L871
'''
import chex
import jax
import jax.numpy as jnp

Array   = chex.Array
Numeric = chex.Numeric



def huber_loss(x: Array, delta: float = 1.) -> Array:
    """Huber loss, similar to L2 loss close to zero, L1 loss away from zero.
    See "Robust Estimation of a Location Parameter" by Huber.
    (https://projecteuclid.org/download/pdf_1/euclid.aoms/1177703732).
    Args:
        x: a vector of arbitrary shape.
        delta: the bounds for the huber loss transformation, defaults at 1.
        Note `grad(huber_loss(x))` is equivalent to `grad(0.5 * clip_gradient(x)**2)`.
    Returns:
        a vector of same shape of `x`.
    """
    chex.assert_type(x, float)

    # 0.5 * x^2                  if |x| <= d
    # 0.5 * d^2 + d * (|x| - d)  if |x| > d
    abs_x     = jnp.abs(x)
    quadratic = jnp.minimum(abs_x, delta)
    # Same as max(abs_x - delta, 0) but avoids potentially doubling gradient.
    linear = abs_x - quadratic
    return 0.5 * quadratic**2 + delta * linear


def quantile_regression_loss(
    dist_src: Array,
    tau_src: Array,
    dist_target: Array,
    huber_param: float = 1.
    # huber_param: float = 0.
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

    # Calculate Huber loss.
    # if huber_param > 0.:
    #     loss = huber_loss(delta, huber_param)
    # else:
    #     loss = jnp.abs(delta)
    loss = delta ** 2.
    loss *= weight

    # Average over target-samples dimension, sum over src-samples dimension.
    return jnp.sum(jnp.mean(loss, axis=-1))


batch_quantile_regression_loss = jax.vmap(quantile_regression_loss, in_axes=(0, None, 0))







# def iqn_atari_network(num_actions: int, latent_dim: int) -> NetworkFn:
# """IQN network, expects `uint8` input."""
#
#     def net_fn(iqn_inputs):
#         """Function representing IQN-DQN Q-network."""
#         state = iqn_inputs.state  # batch x state_shape
#         taus  = iqn_inputs.taus  # batch x samples
#
#         # Apply DQN convnet to embed state.
#         state_embedding = dqn_torso()(state)
#         state_dim       = state_embedding.shape[-1]
#
#         # Embed taus with cosine embedding + linear layer.
#         # cos(pi * i * tau) for i = 1,...,latents for each batch_element x sample.
#         # Broadcast everything to batch x samples x latent_dim.
#         pi_multiples  = jnp.arange(1, latent_dim + 1, dtype=jnp.float32) * jnp.pi
#         tau_embedding = jnp.cos(pi_multiples[None, None, :] * taus[:, :, None])
#
#         # Map tau embedding onto state_dim via linear layer.
#         embedding_layer = linear(state_dim)
#         tau_embedding   = hk.BatchApply(embedding_layer)(tau_embedding)
#         tau_embedding   = jax.nn.relu(tau_embedding)
#
#         # Reshape/broadcast both embeddings to batch x num_samples x state_dim
#         # and multiply together, before applying value head.
#         head_input = tau_embedding * state_embedding[:, None, :]
#         value_head = dqn_value_head(num_actions)
#         q_dist     = hk.BatchApply(value_head)(head_input)
#         q_values   = jnp.mean(q_dist, axis=1)
#         q_values   = jax.lax.stop_gradient(q_values)
#         return IqnOutputs(q_dist=q_dist, q_values=q_values)
#
#
#
# _batch_quantile_q_learning = jax.vmap(rlax.quantile_q_learning, in_axes=(0, 0, 0, 0, 0, 0, 0, None))
#
#
#     def loss_fn(online_params, target_params, transitions, rng_key):
#         """Calculates loss given network parameters and transitions."""
#         # Sample tau values for q_tm1, q_t_selector, q_t.
#         batch_size = self._batch_size
#         rng_key, *sample_keys = jax.random.split(rng_key, 4)
#         tau_tm1 = _sample_tau(sample_keys[0], (batch_size, tau_samples_s_tm1))
#         tau_t_selector = _sample_tau(sample_keys[1],
#         (batch_size, tau_samples_policy))
#         tau_t = _sample_tau(sample_keys[2], (batch_size, tau_samples_s_t))
#
#         # Compute Q value distributions.
#         _, *apply_keys = jax.random.split(rng_key, 4)
#         dist_q_tm1 = network.apply(online_params, apply_keys[0],
#             IqnInputs(transitions.s_tm1, tau_tm1)).q_dist
#         dist_q_t_selector = network.apply(
#             target_params, apply_keys[1],
#             IqnInputs(transitions.s_t, tau_t_selector)).q_dist
#         dist_q_target_t = network.apply(target_params, apply_keys[2],
#             IqnInputs(transitions.s_t, tau_t)).q_dist
#         losses = _batch_quantile_q_learning(
#             dist_q_tm1,
#             tau_tm1,
#             transitions.a_tm1,
#             transitions.r_t,
#             transitions.discount_t,
#             dist_q_t_selector,
#             dist_q_target_t,
#             huber_param,
#         )
#         chex.assert_shape(losses, (self._batch_size,))
#         loss = jnp.mean(losses)
#         return loss