import csv
import json
import math
import subprocess
import sys

from scripts.diffusion import ALPHA_BAR
from scripts.noise_model import ddpm_trajectory, ddpm_step, initialize_model, learning_rate, noise_loss, predict_noise, training_batch

import jax
import jax.numpy as jnp
import numpy as np
import pytest


def test_time_conditioned_network_and_mean_squared_noise_loss():
    params = initialize_model(jax.random.key(11))
    x = jnp.array([[1., -2.], [1., -2.]])
    t = jnp.array([1, 1000])
    angles = (np.array(t)[:, None] - 1) * 10000. ** (-np.arange(16) / 16)
    h = np.concatenate((x, np.sin(angles), np.cos(angles)), axis=1)
    for i in range(3):
        h = h @ np.array(params[f"w{i}"]) + np.array(params[f"b{i}"])
        h = h / (1 + np.exp(-h))
    expected = h @ np.array(params["w3"]) + np.array(params["b3"])
    predictions = predict_noise(params, x, t)
    np.testing.assert_allclose(predictions, expected, atol=2e-6)
    assert not np.allclose(predictions[0], predictions[1])
    epsilon = jnp.array([[0.5, -0.25], [-1., 2.]])
    np.testing.assert_allclose(noise_loss(params, x, t, epsilon), np.mean((expected - epsilon) ** 2), rtol=2e-6)


def test_model_initialization_matches_recipe_widths_and_variance():
    params = initialize_model(jax.random.key(16))
    for i, (inputs, outputs) in enumerate(zip((34, 128, 128, 128), (128, 128, 128, 2))):
        weights = np.array(params[f"w{i}"])
        assert weights.shape == (inputs, outputs)
        assert weights.dtype == np.float32
        np.testing.assert_array_equal(params[f"b{i}"], np.zeros(outputs))
        assert abs(weights.mean()) < 6 / math.sqrt(inputs * weights.size)
        assert abs(weights.var() * inputs - 1) < 6 * math.sqrt(2 / weights.size)


def test_training_draws_uniform_levels_and_the_actual_added_noise():
    x, t, epsilon = map(np.array, training_batch(jax.random.key(12), 65536))
    assert t.min() == 1 and t.max() == 1000
    counts = np.histogram(t, bins=np.arange(1, 1002, 100))[0]
    assert np.max(abs(counts - len(t) / 10)) < 6 * math.sqrt(len(t) * 0.1 * 0.9)
    np.testing.assert_allclose(epsilon.mean(axis=0), 0., atol=0.025)
    np.testing.assert_allclose(np.cov(epsilon.T), np.eye(2), atol=0.025)
    a = np.array(ALPHA_BAR)[t, None]
    clean = (x - np.sqrt(1 - a) * epsilon) / np.sqrt(a)
    centres = np.array([[3., 0.], [0., 3.], [-3., 0.], [0., -3.]])
    labels = np.argmin(np.sum((clean[:, None] - centres) ** 2, axis=-1), axis=1)
    for k in range(4):
        group = clean[labels == k]
        assert abs(len(group) / len(clean) - 0.25) < 0.015
        np.testing.assert_allclose(group.mean(axis=0), centres[k], atol=0.025)
        np.testing.assert_allclose(np.cov(group.T), 0.09 * np.eye(2), atol=0.005)
    assert abs(np.corrcoef(t, epsilon[:, 0])[0, 1]) < 0.025


def test_learning_rate_is_cosine_with_zero_on_the_last_update():
    assert learning_rate(0, 101, 0.001).dtype == jnp.float32
    np.testing.assert_allclose([learning_rate(i, 101, 0.001) for i in [0, 25, 50, 100]],
                               [0.001, 0.001 * (1 + math.sqrt(0.5)) / 2, 0.0005, 0.], atol=1e-10)


@pytest.mark.parametrize("t", [1, 2, 500, 1000])
def test_ddpm_uses_noise_predictions_beta_variance_and_no_final_noise(t):
    params = jax.tree.map(jnp.zeros_like, initialize_model(jax.random.key(13)))
    params["b3"] = jnp.array([0.5, -0.25])
    x = jnp.tile(jnp.array([3., -4.]), (32768, 1))
    beta = np.linspace(0.0001, 0.02, 1000)[t - 1]
    a = np.prod(1 - np.linspace(0.0001, 0.02, 1000)[:t])
    mean = (np.array([3., -4.]) - beta / math.sqrt(1 - a) * np.array([0.5, -0.25])) / math.sqrt(1 - beta)
    actual = np.array(ddpm_step(params, jax.random.key(14), x, t))
    if t == 1:
        np.testing.assert_allclose(actual, np.broadcast_to(mean, actual.shape), atol=2e-6)
        np.testing.assert_array_equal(actual, ddpm_step(params, jax.random.key(15), x, t))
        assert np.max(abs(actual)) > 3
    else:
        np.testing.assert_allclose(actual.mean(axis=0), mean, atol=6 * math.sqrt(beta / len(x)) + 3e-6)
        np.testing.assert_allclose(np.cov(actual.T), beta * np.eye(2), atol=0.035 * beta)


def test_training_command_retains_ten_reloadable_models_and_generates_samples(tmp_path):
    run, figure = tmp_path / "artifacts", tmp_path / "evidence" / "training.png"
    subprocess.run([sys.executable, "-m", "scripts.train_diffusion",
                    "--updates", "900", "--batch-size", "128", "--samples", "128",
                    "--checkpoint-updates", "0", "5", "10", "20", "50", "100", "200", "500", "700", "900",
                    "--output", str(run), "--figure", str(figure)],
                   check=True, capture_output=True, text=True)
    settings = json.loads((run / "settings.json").read_text())
    assert settings["backend"] == "cpu"
    assert settings["updates"] == 900
    with (run / "stages.csv").open() as stream:
        stages = list(csv.DictReader(stream))
    assert [int(row["update"]) for row in stages] == [0, 5, 10, 20, 50, 100, 200, 500, 700, 900]
    assert float(stages[-1]["evaluation_loss"]) < 0.7 * float(stages[0]["evaluation_loss"])
    assert len(list(run.glob("checkpoint-*.npz"))) == 10
    with (run / "loss.csv").open() as stream:
        history = list(csv.DictReader(stream))
    assert len(history) == 900
    assert float(history[0]["learning_rate"]) == pytest.approx(0.001)
    assert float(history[-1]["learning_rate"]) == 0.
    assert np.isfinite([float(row["training_loss"]) for row in history]).all()
    initial_key = jax.random.wrap_key_data(jnp.array(settings["keys"]["initialization"], dtype=jnp.uint32))
    with np.load(run / stages[0]["checkpoint"]) as saved:
        for name, value in initialize_model(initial_key).items():
            np.testing.assert_array_equal(saved[name], value)
    with np.load(run / stages[-1]["checkpoint"]) as saved:
        final = {name: jnp.array(saved[name]) for name in saved.files}
    evaluation_key = jax.random.wrap_key_data(jnp.array(settings["keys"]["evaluation"], dtype=jnp.uint32))
    batch = training_batch(evaluation_key, settings["evaluation_samples"])
    np.testing.assert_allclose(noise_loss(final, *batch), float(stages[-1]["evaluation_loss"]), rtol=2e-6)
    sampling_key = jax.random.wrap_key_data(jnp.array(settings["keys"]["sampling"], dtype=jnp.uint32))
    reverse_key, initial_key = jax.random.split(sampling_key)
    initial = jax.random.normal(initial_key, (128, 2), dtype=jnp.float32)
    with np.load(run / "samples.npz") as saved:
        samples = saved["generated"]
        assert samples.shape == (128, 2) and np.isfinite(samples).all()
        np.testing.assert_array_equal(samples, ddpm_trajectory(final, initial, reverse_key)[-1])
    from PIL import Image
    with Image.open(figure) as image:
        image.verify()
