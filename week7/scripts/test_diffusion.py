import json
import math
from pathlib import Path
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from diffusion import ALPHA_BAR, BETA, STEPS, forward_sample, forward_trajectory, sample_mixture


def reference_alpha_bar(t):
    return np.prod(1 - np.linspace(0.0001, 0.02, 1000, dtype=np.float64)[:t])


def test_clean_samples_have_equal_weights_and_component_covariance():
    samples = np.array(sample_mixture(jax.random.key(10), 32768))
    assert samples.shape == (32768, 2)
    centres = np.array([[3., 0.], [0., 3.], [-3., 0.], [0., -3.]])
    labels = np.argmin(np.sum((samples[:, None] - centres) ** 2, axis=-1), axis=1)
    counts = np.bincount(labels, minlength=4)
    assert np.max(abs(counts / len(samples) - 0.25)) < 6 * math.sqrt(0.25 * 0.75 / len(samples))
    for component in range(4):
        group = samples[labels == component]
        np.testing.assert_allclose(group.mean(axis=0), centres[component], atol=0.035)
        np.testing.assert_allclose(np.cov(group.T), 0.09 * np.eye(2), atol=0.006)


def test_schedule_includes_clean_state_and_exact_requested_endpoints():
    assert STEPS == 1000
    np.testing.assert_allclose(BETA, np.r_[0., np.linspace(0.0001, 0.02, 1000)], rtol=2e-7)
    np.testing.assert_allclose(ALPHA_BAR, np.r_[1., np.cumprod(1 - np.linspace(0.0001, 0.02, 1000))], rtol=2e-6)


@pytest.mark.parametrize("t", [0, 1, 500, 1000])
def test_forward_marginal_matches_closed_form(t):
    x0 = jnp.array([[3., 0.], [0., -3.], [1., -2.]])
    key = jax.random.key(20)
    a = reference_alpha_bar(t)
    expected = math.sqrt(a) * np.array(x0) + math.sqrt(1 - a) * np.array(jax.random.normal(key, x0.shape))
    np.testing.assert_allclose(forward_sample(key, x0, t), expected, atol=4e-6)


def test_forward_chain_has_requested_transition_variances_and_terminal_distribution():
    x0 = jnp.tile(jnp.array([3., 0.]), (8192, 1))
    path = np.array(forward_trajectory(jax.random.key(30), x0))
    assert path.shape == (1001, 8192, 2)
    np.testing.assert_array_equal(path[0], x0)
    for t in [1, 500, 1000]:
        beta = np.linspace(0.0001, 0.02, 1000)[t - 1]
        innovations = (path[t] - math.sqrt(1 - beta) * path[t - 1]) / math.sqrt(beta)
        np.testing.assert_allclose(innovations.mean(axis=0), 0., atol=0.05)
        np.testing.assert_allclose(np.cov(innovations.T), np.eye(2), atol=0.06)
        if t > 1:
            assert abs(np.corrcoef(innovations[:, 0], path[t - 1, :, 1])[0, 1]) < 0.05
    a = reference_alpha_bar(1000)
    np.testing.assert_allclose(path[-1].mean(axis=0), math.sqrt(a) * np.array([3., 0.]), atol=0.05)
    np.testing.assert_allclose(np.cov(path[-1].T), (1 - a) * np.eye(2), atol=0.06)


def test_forward_command_saves_raw_chain_settings_and_figure(tmp_path):
    run = tmp_path / "artifacts"
    figure = tmp_path / "evidence" / "forward.png"
    subprocess.run([sys.executable, str(Path(__file__).with_name("run_diffusion.py")),
                    "--samples", "128", "--seed", "71", "--output", str(run),
                    "--figure", str(figure)], check=True, capture_output=True, text=True)
    settings = json.loads((run / "settings.json").read_text())
    assert settings["samples"] == 128
    assert settings["seed"] == 71
    assert settings["backend"] == "cpu"
    with np.load(run / "forward.npz") as data:
        assert data["trajectory"].shape == (1001, 128, 2)
        np.testing.assert_array_equal(data["levels"], np.arange(1001))
        key = jax.random.wrap_key_data(jnp.array(settings["keys"]["mixture"], dtype=jnp.uint32))
        np.testing.assert_allclose(data["trajectory"][0], sample_mixture(key, 128))
    from PIL import Image
    with Image.open(figure) as image:
        image.verify()
