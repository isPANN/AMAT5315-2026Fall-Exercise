"""JAX target sampling and forward diffusion for a four-component 2D mixture."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp


STEPS = 1000
CENTRES = jnp.array([[3., 0.], [0., 3.], [-3., 0.], [0., -3.]], dtype=jnp.float32)
COMPONENT_VARIANCE = 0.09
BETA = jnp.concatenate((jnp.zeros(1), jnp.linspace(0.0001, 0.02, STEPS)))
ALPHA_BAR = jnp.cumprod(1 - BETA)


def sample_mixture(key, count):
    """Draw independent samples with equal component probabilities."""
    component_key, noise_key = jax.random.split(key)
    components = jax.random.randint(component_key, (count,), 0, 4)
    return CENTRES[components] + jnp.sqrt(COMPONENT_VARIANCE) * jax.random.normal(noise_key, (count, 2))


def forward_sample(key, x0, t):
    """Draw x_t | x_0 directly, for a scalar diffusion level 0 <= t <= 1000."""
    a = ALPHA_BAR[t]
    return jnp.sqrt(a) * x0 + jnp.sqrt(1 - a) * jax.random.normal(key, x0.shape)


@jax.jit
def forward_trajectory(key, x0):
    """Return the Markov chain in increasing level order, including x_0."""
    def step(state, beta):
        x, key = state
        key, noise_key = jax.random.split(key)
        x = jnp.sqrt(1 - beta) * x + jnp.sqrt(beta) * jax.random.normal(noise_key, x.shape)
        return (x, key), x

    _, path = jax.lax.scan(step, (x0, key), BETA[1:])
    return jnp.concatenate((x0[None], path))
