import csv
import json
import subprocess
import sys

from scripts.noise_model import ddpm_trajectory, initialize_model, predict_noise
from scripts.diffusion import ALPHA_BAR, BETA
from scripts.training_movie import training_frames

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np


def test_reverse_trajectories_share_initial_points_and_step_noise_across_predictors():
    first = jax.tree.map(jnp.zeros_like, initialize_model(jax.random.key(31)))
    second = dict(first, b3=jnp.array([0.5, -0.25]))
    key = jax.random.key(32)
    paths = [np.array(ddpm_trajectory(params, key, 10)) for params in (first, second)]
    assert paths[0].shape == (1001, 10, 2)
    np.testing.assert_array_equal(paths[0][0], paths[1][0])
    _, initial_key = jax.random.split(key)
    np.testing.assert_array_equal(paths[0][0], jax.random.normal(initial_key, (10, 2)))
    for i in [0, 500, 998]:
        t = 1000 - i
        innovations, scales = [], []
        for params, path in zip((first, second), paths):
            beta, a = float(BETA[t]), float(ALPHA_BAR[t])
            mean = (path[i] - beta / np.sqrt(1 - a) * np.array(predict_noise(params, jnp.array(path[i]), t))) / np.sqrt(1 - beta)
            innovations.append(path[i + 1] - mean)
            scales.append(np.max(np.abs(path[i + 1])) + np.max(np.abs(mean)))
        # Recovering noise by subtracting large float32 states incurs roundoff.
        roundoff = 8 * np.finfo(np.float32).eps * sum(scales)
        np.testing.assert_allclose(innovations[0], innovations[1], atol=roundoff)
    assert not np.allclose(paths[0][-1], paths[1][-1])


def test_stage_frames_show_ten_full_paths_and_no_future_losses():
    rng = np.random.default_rng(33)
    paths = rng.normal(size=(10, 1001, 10, 2))
    paths[0, -1, 0] = [100., -80.]
    updates = np.array([0, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 25000])
    history = np.column_stack((np.arange(1, 25001), 1 / np.sqrt(np.arange(1, 25001))))
    evaluation = np.linspace(1., 0.25, 10)
    fig, update = training_frames(paths, updates, history, evaluation)
    zoom, overview, loss_ax = fig.axes
    fixed_limits = [(ax.get_xlim(), ax.get_ylim()) for ax in fig.axes]
    colours = [line.get_color() for line in zoom.lines]
    for stage in [0, 1, 9]:
        update(stage)
        for ax in [zoom, overview]:
            for chain, line in enumerate(ax.lines):
                np.testing.assert_array_equal(line.get_xydata(), paths[stage, :, chain])
        np.testing.assert_array_equal(zoom.collections[-1].get_offsets(), paths[stage, -1])
        raw_history = loss_ax.lines[0].get_xydata()
        np.testing.assert_array_equal(raw_history, history[:updates[stage]])
        assert [line.get_color() for line in zoom.lines] == colours
        assert [(ax.get_xlim(), ax.get_ylim()) for ax in fig.axes] == fixed_limits
        visible_arrows = [text for text in zoom.texts if text.get_visible()]
        if stage == 0:
            assert len(visible_arrows) == 1
            assert "128.1" in visible_arrows[0].get_text()
            np.testing.assert_allclose(visible_arrows[0].xy, [5., -4.])
        else:
            assert not visible_arrows
    assert zoom.get_xlim() == (-5., 5.) and zoom.get_ylim() == (-5., 5.)
    assert overview.get_xlim()[1] > 100
    plt.close(fig)


def test_offscreen_annotations_cover_all_edges_and_clear_between_stages():
    paths = np.zeros((10, 1001, 10, 2))
    outside = np.array([[20., 10.], [-12., 6.], [3., 15.], [-6., -18.], [30., 3.]])
    paths[0, -1, :5] = outside
    paths[1, -1, :, 0] = np.linspace(-4., 4., 10)
    paths[1, -1, :, 1] = np.linspace(20., 100., 10)
    updates = np.array([0, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 25000])
    history = np.column_stack((np.arange(1, 25001), np.ones(25000)))
    fig, update = training_frames(paths, updates, history, np.ones(10))
    zoom = fig.axes[0]
    update(0)
    arrows = [text for text in zoom.texts if text.get_visible()]
    assert len(arrows) == 5
    for chain, arrow in enumerate(arrows):
        expected_edge = outside[chain] * 5 / np.max(np.abs(outside[chain]))
        np.testing.assert_allclose(arrow.xy, expected_edge)
        assert f"{np.linalg.norm(outside[chain]):.1f}" in arrow.get_text()
        np.testing.assert_array_equal(zoom.lines[chain].get_xydata(), paths[0, :, chain])
    assert len({tuple(text.get_position()) for text in arrows}) == 5
    update(1)
    fig.canvas.draw()
    boxes = [text.get_bbox_patch().get_window_extent() for text in zoom.texts if text.get_visible()]
    assert len(boxes) == 10
    assert not any(first.overlaps(second) for i, first in enumerate(boxes) for second in boxes[i + 1:])
    update(9)
    assert not any(text.get_visible() for text in zoom.texts)
    plt.close(fig)


def test_training_movie_command_replays_all_checkpoints_with_shared_randomness(tmp_path):
    run = tmp_path / "artifacts"
    run.mkdir()
    updates = [0, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 25000]
    params = initialize_model(jax.random.key(34))
    with (run / "stages.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["stage", "update", "evaluation_loss", "checkpoint"])
        for stage, count in enumerate(updates):
            filename = f"checkpoint-{count:06d}.npz"
            np.savez(run / filename, **params)
            writer.writerow([stage, count, 1., filename])
    with (run / "loss.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["update", "learning_rate", "training_loss"])
        writer.writerows((i, 0.001, 1.) for i in range(1, 25001))
    key = jax.random.key(35)
    (run / "settings.json").write_text(json.dumps({"keys": {"sampling": np.array(jax.random.key_data(key)).tolist()}}))
    output = tmp_path / "evidence" / "stages.mp4"
    subprocess.run([sys.executable, "-m", "scripts.training_movie", "--run", str(run),
                    "--output", str(output), "--frames-per-stage", "1"],
                   check=True, capture_output=True, text=True)
    with np.load(run / "training-paths.npz") as saved:
        paths = saved["trajectories"]
        assert paths.shape == (10, 1001, 10, 2)
        assert np.isfinite(paths).all()
        np.testing.assert_array_equal(saved["updates"], updates)
        np.testing.assert_array_equal(saved["levels"], np.arange(1000, -1, -1))
        expected = np.array(ddpm_trajectory(params, key, 10))
        for path in paths:
            np.testing.assert_array_equal(path, expected)
    settings = json.loads((run / "training-movie.json").read_text())
    assert settings["sampling_key"] == np.array(jax.random.key_data(key)).tolist()
    assert settings["frames_per_stage"] == 1
    info = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=nb_frames,width,height",
                           "-of", "json", str(output)], check=True, capture_output=True, text=True)
    video = json.loads(info.stdout)["streams"][0]
    assert int(video["nb_frames"]) == 10
    assert (video["width"], video["height"]) == (1500, 500)
    subprocess.run(["ffmpeg", "-v", "error", "-xerror", "-i", str(output), "-f", "null", "-"], check=True, capture_output=True)
