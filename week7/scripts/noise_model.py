"""Time-conditioned noise predictor and DDPM sampler for the 2D mixture."""

from functools import partial

from scripts.diffusion import ALPHA_BAR, BETA, STEPS, sample_mixture

import jax
import jax.numpy as jnp


def initialize_model(key):
    params = {}
    for i, (layer_key, inputs, outputs) in enumerate(zip(jax.random.split(key, 4), (34, 128, 128, 128), (128, 128, 128, 2))):
        params[f"w{i}"] = jax.random.normal(layer_key, (inputs, outputs), dtype=jnp.float32) / jnp.sqrt(inputs)
        params[f"b{i}"] = jnp.zeros(outputs, dtype=jnp.float32)
    return params


def predict_noise(params, x, t):
    """Predict epsilon for a batch of points and scalar or per-point integer levels."""
    level = jnp.broadcast_to(jnp.asarray(t, dtype=x.dtype), x.shape[:-1])[..., None] - 1
    angles = level * 10000. ** (-jnp.arange(16, dtype=jnp.float32) / 16)
    h = jnp.concatenate((x, jnp.sin(angles), jnp.cos(angles)), axis=-1)
    for i in range(3):
        h = jax.nn.silu(h @ params[f"w{i}"] + params[f"b{i}"])
    return h @ params["w3"] + params["b3"]


def training_batch(key, count):
    """Fresh x_0, uniform t in 1..1000, and independent standard-normal epsilon."""
    data_key, time_key, noise_key = jax.random.split(key, 3)
    x0 = sample_mixture(data_key, count).astype(jnp.float32)
    t = jax.random.randint(time_key, (count,), 1, STEPS + 1)
    epsilon = jax.random.normal(noise_key, x0.shape, dtype=jnp.float32)
    a = ALPHA_BAR[t, None].astype(jnp.float32)
    return jnp.sqrt(a) * x0 + jnp.sqrt(1 - a) * epsilon, t, epsilon


def noise_loss(params, x, t, epsilon):
    """Mean squared error over both samples and coordinates."""
    return jnp.mean((predict_noise(params, x, t) - epsilon) ** 2)


def learning_rate(index, updates, initial):
    """Update indices 0..updates-1 span initial..0, including both endpoints."""
    index = jnp.asarray(index, dtype=jnp.float32)
    return initial * 0.5 * (1 + jnp.cos(jnp.pi * index / (updates - 1)))


@jax.jit
def ddpm_step(params, key, x, t):
    """Unclipped t -> t-1 update with variance beta_t; t=1 adds no noise."""
    beta = BETA[t].astype(x.dtype)
    a = ALPHA_BAR[t].astype(x.dtype)
    mean = (x - beta / jnp.sqrt(1 - a) * predict_noise(params, x, t)) / jnp.sqrt(1 - beta)
    return jax.lax.cond(t > 1,
                        lambda: mean + jnp.sqrt(beta) * jax.random.normal(key, x.shape, dtype=x.dtype),
                        lambda: mean)


@partial(jax.jit, static_argnames=("count",))
def ddpm_trajectory(params, key, count):
    """Return x_1000,...,x_0 with shape (1001,count,2), starting from N(0,I)."""
    key, initial_key = jax.random.split(key)
    initial = jax.random.normal(initial_key, (count, 2), dtype=jnp.float32)

    def step(state, t):
        x, key = state
        key, noise_key = jax.random.split(key)
        x = ddpm_step(params, noise_key, x, t)
        return (x, key), x

    _, path = jax.lax.scan(step, (initial, key), jnp.arange(STEPS, 0, -1))
    return jnp.concatenate((initial[None], path))
