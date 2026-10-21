"""Time-conditioned noise predictor and DDPM sampler for the 2D mixture."""

from functools import partial

from scripts.diffusion import ALPHA_BAR, BETA, CENTRES, COMPONENT_VARIANCE, STEPS, sample_mixture

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


@jax.jit
def ddpm_trajectory(params, initial, key):
    """Return x_1000,...,x_0 from explicit starting points and a reverse-noise key."""
    def step(state, t):
        x, key = state
        key, noise_key = jax.random.split(key)
        x = ddpm_step(params, noise_key, x, t)
        return (x, key), x

    _, path = jax.lax.scan(step, (initial, key), jnp.arange(STEPS, 0, -1))
    return jnp.concatenate((initial[None], path))


def exact_noise(x, t):
    """E[epsilon | x_t=x] for the known equally weighted Gaussian mixture."""
    a = ALPHA_BAR[t].astype(x.dtype)
    variance = COMPONENT_VARIANCE * a + 1 - a
    displacement = x[..., None, :] - jnp.sqrt(a) * CENTRES
    weights = jax.nn.softmax(-jnp.sum(displacement ** 2, axis=-1) / (2 * variance), axis=-1)
    return jnp.sqrt(1 - a) * jnp.sum(weights[..., None] * displacement, axis=-2) / variance


def ddim_step(x, epsilon, t, previous_t):
    """Deterministic DDIM (eta=0) from t to any earlier selected level."""
    a, b = ALPHA_BAR[t].astype(x.dtype), ALPHA_BAR[previous_t].astype(x.dtype)
    clean = (x - jnp.sqrt(1 - a) * epsilon) / jnp.sqrt(a)
    return jnp.sqrt(b) * clean + jnp.sqrt(1 - b) * epsilon


@partial(jax.jit, static_argnames=("levels", "exact"))
def ddim_trajectory(params, initial, levels=tuple(range(STEPS, -1, -1)), *, exact=False):
    """Return states at a strictly decreasing tuple of integer levels ending in 0.

    The first level is in 1..1000; initial contains the points at that level.
    exact=True selects the mixture's conditional-mean noise instead of the MLP.
    No random key, extra noise, or clipping is used.
    """
    if (len(levels) < 2 or not all(isinstance(t, int) for t in levels)
            or not 1 <= levels[0] <= STEPS or levels[-1] != 0
            or any(t <= s for t, s in zip(levels[:-1], levels[1:]))):
        raise ValueError("levels must be a strictly decreasing integer tuple from 1..1000 to 0")

    def step(x, pair):
        t, previous_t = pair
        epsilon = exact_noise(x, t) if exact else predict_noise(params, x, t)
        x = ddim_step(x, epsilon, t, previous_t)
        return x, x

    _, path = jax.lax.scan(step, initial, (jnp.array(levels[:-1]), jnp.array(levels[1:])))
    return jnp.concatenate((initial[None], path))
