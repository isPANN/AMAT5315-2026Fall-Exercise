import json
import math
import subprocess
import sys

from scripts.diffusion import ALPHA_BAR
from scripts.noise_model import ddpm_trajectory, ddim_step, ddim_trajectory, exact_noise, initialize_model

import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.mark.parametrize("t", [0, 1, 300, 1000])
def test_exact_conditional_noise_agrees_with_independent_density_gradient(t):
    a = float(ALPHA_BAR[t])
    means = math.sqrt(a) * jnp.array([[3., 0.], [0., 3.], [-3., 0.], [0., -3.]])
    variance = 0.09 * a + 1 - a
    def log_density(x):
        return jax.scipy.special.logsumexp(-jnp.sum((x - means) ** 2, axis=-1) / (2 * variance))
    x = jnp.array([[0., 0.], [3.2, -0.1], [-1., 1.5], [20., -30.]])
    expected = -math.sqrt(1 - a) * jax.vmap(jax.grad(log_density))(x)
    np.testing.assert_allclose(exact_noise(x, t), expected, rtol=2e-5, atol=3e-5)


@pytest.mark.parametrize("levels", [(1000, 999), (1000, 500), (500, 100), (1, 0), (1000, 0)])
def test_ddim_step_recovers_known_clean_point_at_arbitrary_level_pair(levels):
    t, previous = levels
    x0 = np.array([[3., -0.2], [-4., 0.5]])
    epsilon = np.array([[0.4, -0.7], [-1., 2.]])
    a, b = float(ALPHA_BAR[t]), float(ALPHA_BAR[previous])
    x = math.sqrt(a) * x0 + math.sqrt(1 - a) * epsilon
    expected = math.sqrt(b) * x0 + math.sqrt(1 - b) * epsilon
    # Inverting a nearly pure-noise state amplifies float32 rounding by 1/sqrt(a).
    roundoff = 4 * np.finfo(np.float32).eps * np.max(np.abs(x)) / math.sqrt(a)
    np.testing.assert_allclose(ddim_step(jnp.array(x), jnp.array(epsilon), t, previous), expected, atol=roundoff)


def test_ddim_subset_is_deterministic_and_uses_selected_times():
    params = jax.tree.map(jnp.zeros_like, initialize_model(jax.random.key(51)))
    params["b3"] = jnp.array([0.5, -0.25])
    initial = jnp.array([[1., -2.], [-0.3, 0.7]])
    levels = (1000, 700, 100, 1, 0)
    first = np.array(ddim_trajectory(params, initial, levels))
    second = np.array(ddim_trajectory(params, initial, levels))
    np.testing.assert_array_equal(first, second)
    expected = np.array(initial, dtype=np.float64)
    for i, (t, previous) in enumerate(zip(levels[:-1], levels[1:])):
        a, b = float(ALPHA_BAR[t]), float(ALPHA_BAR[previous])
        clean = (expected - math.sqrt(1 - a) * np.array([0.5, -0.25])) / math.sqrt(a)
        expected = math.sqrt(b) * clean + math.sqrt(1 - b) * np.array([0.5, -0.25])
        np.testing.assert_allclose(first[i + 1], expected, rtol=2e-6, atol=2e-5)


@pytest.mark.parametrize("levels", [(), (1000,), (1000, 1000, 0), (1001, 0), (1000, 1), (1, 3, 0), (1000, 0.5, 0)])
def test_ddim_rejects_invalid_noise_level_subsets(levels):
    with pytest.raises(ValueError):
        ddim_trajectory(None, jnp.ones((2, 2)), levels, exact=True)


def test_ddpm_keeps_explicit_start_but_changes_with_reverse_noise_key():
    params = jax.tree.map(jnp.zeros_like, initialize_model(jax.random.key(52)))
    initial = jax.random.normal(jax.random.key(53), (10, 2))
    first = np.array(ddpm_trajectory(params, initial, jax.random.key(54)))
    second = np.array(ddpm_trajectory(params, initial, jax.random.key(55)))
    np.testing.assert_array_equal(first[0], initial)
    np.testing.assert_array_equal(second[0], initial)
    assert np.max(np.abs(first[-1] - second[-1])) > 1


def test_exact_noise_ddim_chain_recovers_target_modes():
    initial = jax.random.normal(jax.random.key(56), (8192, 2))
    final = np.array(ddim_trajectory(None, initial, tuple(range(1000, -1, -1)), exact=True)[-1])
    centres = np.array([[3., 0.], [0., 3.], [-3., 0.], [0., -3.]])
    labels = ((final[:, None] - centres) ** 2).sum(-1).argmin(1)
    for k in range(4):
        group = final[labels == k]
        assert abs(len(group) / len(final) - 0.25) < 6 * math.sqrt(0.25 * 0.75 / len(final))
        np.testing.assert_allclose(group.mean(0), centres[k], atol=0.04)
        np.testing.assert_allclose(np.cov(group.T), 0.09 * np.eye(2), atol=0.015)


def test_comparison_command_saves_matched_runs_and_reports_zero_ddim_difference(tmp_path):
    params = initialize_model(jax.random.key(57))
    checkpoint = tmp_path / "checkpoint.npz"
    np.savez(checkpoint, **params)
    run, figures = tmp_path / "artifacts", tmp_path / "evidence"
    result = subprocess.run([sys.executable, "-m", "scripts.compare_samplers", "--checkpoint", str(checkpoint),
                             "--samples", "32", "--levels", "1000", "500", "100", "0",
                             "--output", str(run), "--figures", str(figures)],
                            check=True, capture_output=True, text=True)
    settings = json.loads((run / "settings.json").read_text())
    assert settings["ddim_max_absolute_difference"] == 0.
    assert "Largest DDIM difference (all trajectory coordinates): 0" in result.stdout
    with np.load(run / "runs.npz") as saved:
        np.testing.assert_array_equal(saved["ddim_first"], saved["ddim_second"])
        for name in ["ddim_first", "ddim_second", "ddpm_first", "ddpm_second", "ddim_exact"]:
            np.testing.assert_array_equal(saved[name][0], saved["initial"])
            assert np.isfinite(saved[name]).all()
        assert saved["ddim_first"].shape == (4, 32, 2)
        assert saved["ddpm_first"].shape == (1001, 32, 2)
        assert saved["exact_samples"].shape == (32, 2)
        assert not np.array_equal(saved["ddpm_first"][-1], saved["ddpm_second"][-1])
    from PIL import Image
    for name in ["ddim-ddpm-trajectories.png", "ddim-ddpm-samples.png"]:
        with Image.open(figures / name) as image:
            image.verify()
